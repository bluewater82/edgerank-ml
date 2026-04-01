#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_model.py

Combines per-ticker model with portfolio architecture

Author: Andre
Created: 3/31/2026
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

from src.utils.paths import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "model_features.parquet"

FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "dist_from_sma_20",
    "dist_from_sma_50",
    "volatility_5d",
    "volatility_20d",
    "volume_ratio_20d",
]

TARGET_COL = "target_next_day_up"
RETURN_COL = "next_day_return"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def get_candidate_tickers(
    df: pd.DataFrame,
    min_rows: int = 2000,
    allowed_tickers: list[str] | None = None,
) -> list[str]:
    counts = df["ticker"].value_counts()
    tickers = sorted(counts[counts >= min_rows].index.tolist())

    if allowed_tickers is not None:
        allowed_set = set(allowed_tickers)
        tickers = [t for t in tickers if t in allowed_set]

    return tickers


def time_split_ticker(df_ticker: pd.DataFrame, split_ratio: float = 0.8):
    split_index = int(len(df_ticker) * split_ratio)
    train = df_ticker.iloc[:split_index].copy()
    test = df_ticker.iloc[split_index:].copy()
    return train, test


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def generate_ticker_predictions(
    df_ticker: pd.DataFrame,
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
) -> pd.DataFrame:
    ticker = df_ticker["ticker"].iloc[0]

    train, test = time_split_ticker(df_ticker)

    X_train, y_train = prepare_xy(train)
    model = train_model(X_train, y_train)

    out = test.copy()
    probs_up = model.predict_proba(out[FEATURE_COLS])[:, 1]

    out["prob_up"] = probs_up
    out["signal"] = 0
    out.loc[out["prob_up"] >= upper_threshold, "signal"] = 1
    out.loc[out["prob_up"] <= lower_threshold, "signal"] = -1
    out["conviction"] = (out["prob_up"] - 0.5).abs()
    out["model_type"] = "per_ticker"

    return out


def collect_all_predictions(
    df: pd.DataFrame,
    tickers: list[str],
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        df_ticker = (
            df[df["ticker"] == ticker]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        preds = generate_ticker_predictions(
            df_ticker,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
        )
        all_frames.append(preds)

        acted = (preds["signal"] != 0).sum()
        print(
            f"{ticker}: test_rows={len(preds)}, "
            f"signals={acted}, "
            f"avg_prob={preds['prob_up'].mean():.4f}"
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    return combined


def rank_and_select_signals(
    df: pd.DataFrame,
    top_n_per_day: int = 3,
) -> pd.DataFrame:
    out = df.copy()

    signaled = out[out["signal"] != 0].copy()

    if signaled.empty:
        return out

    keep_index = (
        signaled.sort_values(["date", "conviction"], ascending=[True, False])
        .groupby("date", group_keys=False)
        .head(top_n_per_day)
        .index
    )

    out.loc[~out.index.isin(keep_index), "signal"] = 0
    out.loc[out["signal"] == 0, "conviction"] = 0.0

    return out


def apply_conviction_weighted_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["raw_strategy_return"] = 0.0
    out.loc[out["signal"] == 1, "raw_strategy_return"] = out["next_day_return"]
    out.loc[out["signal"] == -1, "raw_strategy_return"] = -out["next_day_return"]

    out["weight"] = 0.0
    out["strategy_return"] = 0.0

    acted = out[out["signal"] != 0].copy()

    if acted.empty:
        return out

    # Conviction-squared weighting
    acted["weight"] = acted["conviction"] ** 2
    acted["weight"] = acted["weight"] / acted.groupby("date")["weight"].transform("sum")

    acted["strategy_return"] = acted["raw_strategy_return"] * acted["weight"]

    out.loc[acted.index, "weight"] = acted["weight"]
    out.loc[acted.index, "strategy_return"] = acted["strategy_return"]

    return out


def summarize_backtest(df: pd.DataFrame) -> None:
    acted = df[df["signal"] != 0].copy()

    print("\n=== Hybrid Backtest Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Acted rows: {len(acted)}")
    print(f"Coverage: {len(acted) / len(df):.4f}")

    if acted.empty:
        print("No trades were taken.")
        return

    win_rate = (acted["strategy_return"] > 0).mean()
    avg_trade_return = acted["strategy_return"].mean()
    median_trade_return = acted["strategy_return"].median()

    print(f"Win rate: {win_rate:.4f}")
    print(f"Average trade return: {avg_trade_return:.6f}")
    print(f"Median trade return: {median_trade_return:.6f}")

    print("\nSignal counts:")
    print(acted["signal"].value_counts().sort_index())

    print("\nWeight summary:")
    print(acted["weight"].describe())

    print("\nReturn summary:")
    print(acted["strategy_return"].describe())

    daily = acted.groupby("date", as_index=True)["strategy_return"].mean().sort_index()
    equity = (1.0 + daily).cumprod()

    print(f"\nCumulative return multiple: {equity.iloc[-1]:.4f}")
    print(f"Approx total return: {(equity.iloc[-1] - 1.0):.4%}")

    print("\n=== Average Return by Year ===")
    yearly = acted.groupby(acted["date"].dt.year)["strategy_return"].mean()
    print(yearly)

    print("\n=== Performance by Ticker ===")
    by_ticker = acted.groupby("ticker").agg(
        trades=("strategy_return", "count"),
        win_rate=("strategy_return", lambda s: (s > 0).mean()),
        avg_return=("strategy_return", "mean"),
        median_return=("strategy_return", "median"),
        total_return=("strategy_return", "sum"),
        avg_weight=("weight", "mean"),
    ).sort_values("avg_return", ascending=False)
    print(by_ticker)

    print("\n=== Top 10 Trades ===")
    print(acted.sort_values("strategy_return", ascending=False).head(10))

    print("\n=== Bottom 10 Trades ===")
    print(acted.sort_values("strategy_return").head(10))


def get_summary_metrics(df: pd.DataFrame) -> dict:
    acted = df[df["signal"] != 0].copy()

    if acted.empty:
        return {}

    return {
        "trades": len(acted),
        "coverage": len(acted) / len(df),
        "win_rate": (acted["strategy_return"] > 0).mean(),
        "avg_return": acted["strategy_return"].mean(),
        "median_return": acted["strategy_return"].median(),
        "total_return": acted["strategy_return"].sum(),
        "longs": int((acted["signal"] == 1).sum()),
        "shorts": int((acted["signal"] == -1).sum()),
        "avg_weight": acted["weight"].mean(),
        "max_gain": acted["strategy_return"].max(),
        "max_loss": acted["strategy_return"].min(),
    }


def log_experiment(name: str, metrics: dict):
    import csv

    file_exists = False
    try:
        with open("hybrid_experiments.csv", "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    fieldnames = [
        "name",
        "trades",
        "coverage",
        "win_rate",
        "avg_return",
        "median_return",
        "total_return",
        "longs",
        "shorts",
        "avg_weight",
        "max_gain",
        "max_loss",
    ]

    with open("hybrid_experiments.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {"name": name, **metrics}
        writer.writerow(row)


def run_hybrid_backtest(
    min_rows: int = 2000,
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
    top_n_per_day: int = 3,
    allowed_tickers: list[str] | None = None,
    experiment_name: str = "hybrid_top3",
) -> pd.DataFrame:
    df = load_data()

    tickers = get_candidate_tickers(
        df,
        min_rows=min_rows,
        allowed_tickers=allowed_tickers,
    )
    print(f"Candidate tickers ({len(tickers)}): {tickers}")

    preds = collect_all_predictions(
        df,
        tickers,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )

    selected = rank_and_select_signals(preds, top_n_per_day=top_n_per_day)
    results = apply_conviction_weighted_returns(selected)

    summarize_backtest(results)

    metrics = get_summary_metrics(results)
    log_experiment(experiment_name, metrics)

    return results


if __name__ == "__main__":
    run_hybrid_backtest()