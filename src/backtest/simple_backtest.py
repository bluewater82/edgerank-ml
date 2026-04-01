#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_backtest.py

A brief description of what this module does.

Author: Andre
Created: 3/30/2026
"""

from __future__ import annotations

import csv
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
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, split_ratio: float = 0.8):
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
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


def generate_signals(
    model,
    test_df: pd.DataFrame,
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
    allowed_tickers: list[str] | None = None,
    top_n_per_day: int | None = 1,
) -> pd.DataFrame:
    out = test_df.copy()

    if allowed_tickers is not None:
        out = out[out["ticker"].isin(allowed_tickers)].copy()

    probs_up = model.predict_proba(out[FEATURE_COLS])[:, 1]
    out["prob_up"] = probs_up

    out["signal"] = 0
    out.loc[out["prob_up"] >= upper_threshold, "signal"] = 1
    out.loc[out["prob_up"] <= lower_threshold, "signal"] = -1

    out["conviction"] = (out["prob_up"] - 0.5).abs()

    if top_n_per_day is not None:
        signaled = out[out["signal"] != 0].copy()

        if not signaled.empty:
            keep_index = (
                signaled.sort_values(["date", "conviction"], ascending=[True, False])
                .groupby("date", group_keys=False)
                .head(top_n_per_day)
                .index
            )

            out.loc[~out.index.isin(keep_index), "signal"] = 0

    return out


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


def apply_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Raw signed return based on direction
    out["raw_strategy_return"] = 0.0
    out.loc[out["signal"] == 1, "raw_strategy_return"] = out["next_day_return"]
    out.loc[out["signal"] == -1, "raw_strategy_return"] = -out["next_day_return"]

    # Default weight = 0 for rows with no signal
    out["weight"] = 0.0

    acted_mask = out["signal"] != 0
    acted = out.loc[acted_mask].copy()

    if not acted.empty:
        # Normalize conviction within each day so total daily exposure sums to 1
        daily_conviction_sum = acted.groupby("date")["conviction"].transform("sum")
        acted["weight"] = acted["conviction"] / daily_conviction_sum

        # Weighted return
        acted["strategy_return"] = acted["raw_strategy_return"] * acted["weight"]

        # Write back into main frame
        out.loc[acted.index, "weight"] = acted["weight"]
        out.loc[acted.index, "strategy_return"] = acted["strategy_return"]

    # Any flat rows should contribute zero
    out["strategy_return"] = out["strategy_return"].fillna(0.0)

    return out


def summarize_backtest(df: pd.DataFrame) -> None:
    acted = df[df["signal"] != 0].copy()

    print("\n=== Backtest Summary ===")
    print(f"Total test rows: {len(df)}")
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

    print("\nReturn summary:")
    print(acted["strategy_return"].describe())

    daily = acted.groupby("date", as_index=True)["strategy_return"].mean().sort_index()
    equity = (1.0 + daily).cumprod()

    print(f"\nCumulative return multiple: {equity.iloc[-1]:.4f}")
    print(f"Approx total return: {(equity.iloc[-1] - 1.0):.4%}")

    print("\n=== Top 10 Trades ===")
    print(acted.sort_values("strategy_return", ascending=False).head(10))

    print("\n=== Bottom 10 Trades ===")
    print(acted.sort_values("strategy_return").head(10))

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
    ).sort_values("avg_return", ascending=False)

    print(by_ticker)

    import csv


def log_experiment(name: str, metrics: dict):
    file_exists = False

    try:
        with open("experiments.csv", "r"):
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

    with open("experiments.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {"name": name, **metrics}
        writer.writerow(row)


def run_backtest():
    df = load_data()
    train, test = time_split(df)

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    X_train, y_train = prepare_xy(train)
    model = train_model(X_train, y_train)

    results = generate_signals(
        model,
        test,
        upper_threshold=0.60,
        lower_threshold=0.40,
        allowed_tickers=["AAPL", "NVDA", "IWN", "XLK", "SPY", "NFLX", "GOOG"],
        top_n_per_day=3,
    )
    results = apply_strategy_returns(results)
    summarize_backtest(results)
    metrics = get_summary_metrics(results)
    log_experiment("top3", metrics)

    results.to_parquet("reports/backtest_results.parquet", index=False)

    return results


if __name__ == "__main__":
    run_backtest()