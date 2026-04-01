#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_ticker_model.py

New model that is used per-ticker to address that individual tickers
deviate in a way that dragged down the generalized model.

Author: Andre
Created: 3/31/2026
"""

from __future__ import annotations

import csv
from pathlib import Path

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
) -> list[str]:
    counts = df["ticker"].value_counts()
    tickers = sorted(counts[counts >= min_rows].index.tolist())
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


def generate_signals(
    model,
    test_df: pd.DataFrame,
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
) -> pd.DataFrame:
    out = test_df.copy()

    probs_up = model.predict_proba(out[FEATURE_COLS])[:, 1]
    out["prob_up"] = probs_up

    out["signal"] = 0
    out.loc[out["prob_up"] >= upper_threshold, "signal"] = 1
    out.loc[out["prob_up"] <= lower_threshold, "signal"] = -1

    out["conviction"] = (out["prob_up"] - 0.5).abs()

    return out


def apply_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["raw_strategy_return"] = 0.0
    out.loc[out["signal"] == 1, "raw_strategy_return"] = out["next_day_return"]
    out.loc[out["signal"] == -1, "raw_strategy_return"] = -out["next_day_return"]

    # For per-ticker models, acted rows get full weight.
    out["weight"] = 0.0
    out.loc[out["signal"] != 0, "weight"] = 1.0

    out["strategy_return"] = out["raw_strategy_return"] * out["weight"]

    return out


def summarize_ticker_results(
    ticker: str,
    results: pd.DataFrame,
) -> dict:
    acted = results[results["signal"] != 0].copy()

    summary = {
        "ticker": ticker,
        "rows_total": len(results),
        "trades": len(acted),
        "coverage": len(acted) / len(results) if len(results) else 0.0,
        "win_rate": 0.0,
        "avg_return": 0.0,
        "median_return": 0.0,
        "total_return": 0.0,
        "longs": 0,
        "shorts": 0,
        "avg_prob_up": results["prob_up"].mean() if "prob_up" in results.columns else 0.0,
        "max_gain": 0.0,
        "max_loss": 0.0,
    }

    if acted.empty:
        return summary

    summary.update(
        {
            "win_rate": (acted["strategy_return"] > 0).mean(),
            "avg_return": acted["strategy_return"].mean(),
            "median_return": acted["strategy_return"].median(),
            "total_return": acted["strategy_return"].sum(),
            "longs": int((acted["signal"] == 1).sum()),
            "shorts": int((acted["signal"] == -1).sum()),
            "max_gain": acted["strategy_return"].max(),
            "max_loss": acted["strategy_return"].min(),
        }
    )

    return summary


def backtest_single_ticker(
    df_ticker: pd.DataFrame,
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
) -> tuple[dict, pd.DataFrame]:
    ticker = df_ticker["ticker"].iloc[0]

    train, test = time_split_ticker(df_ticker)

    X_train, y_train = prepare_xy(train)
    model = train_model(X_train, y_train)

    results = generate_signals(
        model,
        test,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
    results = apply_strategy_returns(results)

    summary = summarize_ticker_results(ticker, results)
    return summary, results


def log_summaries_to_csv(
    summaries_df: pd.DataFrame,
    filepath: str = "per_ticker_results.csv",
) -> None:
    summaries_df.to_csv(filepath, index=False)


def print_ranked_summary(summaries_df: pd.DataFrame) -> None:
    print("\n=== Per-Ticker Results (sorted by avg_return) ===")
    cols = [
        "ticker",
        "trades",
        "coverage",
        "win_rate",
        "avg_return",
        "median_return",
        "total_return",
        "longs",
        "shorts",
        "max_gain",
        "max_loss",
    ]
    print(summaries_df[cols].sort_values("avg_return", ascending=False))


def run_all_tickers(
    min_rows: int = 2000,
    upper_threshold: float = 0.65,
    lower_threshold: float = 0.35,
) -> pd.DataFrame:
    df = load_data()
    tickers = get_candidate_tickers(df, min_rows=min_rows)

    print(f"Candidate tickers ({len(tickers)}): {tickers}")

    summaries: list[dict] = []

    for ticker in tickers:
        df_ticker = df[df["ticker"] == ticker].copy().sort_values("date").reset_index(drop=True)

        summary, _ = backtest_single_ticker(
            df_ticker,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
        )
        summaries.append(summary)

        print(
            f"{ticker}: trades={summary['trades']}, "
            f"win_rate={summary['win_rate']:.4f}, "
            f"avg_return={summary['avg_return']:.6f}, "
            f"total_return={summary['total_return']:.6f}"
        )

    summaries_df = pd.DataFrame(summaries)

    log_summaries_to_csv(summaries_df, filepath="per_ticker_results.csv")
    print_ranked_summary(summaries_df)

    return summaries_df


if __name__ == "__main__":
    run_all_tickers()