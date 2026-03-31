#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py

A brief description of what this module does.

Author: Andre
Created: 3/29/2026
"""

from __future__ import annotations

import pandas as pd

from src.utils.paths import PROCESSED_DIR, ensure_directories


INPUT_FILE = PROCESSED_DIR / "daily_prices.parquet"
OUTPUT_FILE = PROCESSED_DIR / "model_features.parquet"


def load_prices() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run the price loader first."
        )

    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    grouped = df.groupby("ticker", group_keys=False)

    # Returns
    df["return_1d"] = grouped["close"].pct_change(1)
    df["return_5d"] = grouped["close"].pct_change(5)
    df["return_20d"] = grouped["close"].pct_change(20)

    # Simple moving averages
    df["sma_5"] = grouped["close"].transform(lambda s: s.rolling(5).mean())
    df["sma_20"] = grouped["close"].transform(lambda s: s.rolling(20).mean())
    df["sma_50"] = grouped["close"].transform(lambda s: s.rolling(50).mean())

    # Distance from trend
    df["dist_from_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["dist_from_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # Rolling volatility based on 1-day returns
    df["volatility_5d"] = grouped["return_1d"].transform(lambda s: s.rolling(5).std())
    df["volatility_20d"] = grouped["return_1d"].transform(lambda s: s.rolling(20).std())

    # Volume features
    df["volume_avg_20d"] = grouped["volume"].transform(lambda s: s.rolling(20).mean())
    df["volume_ratio_20d"] = df["volume"] / df["volume_avg_20d"]

    # Actual next-day return for backtesting
    df["next_day_return"] = grouped["close"].shift(-1) / df["close"] - 1.0

    # Prediction target: whether next day's close is higher than today's close
    df["target_next_day_up"] = (
        grouped["close"].shift(-1) > df["close"]
    ).astype("int64")

    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "sma_5",
        "sma_20",
        "sma_50",
        "dist_from_sma_20",
        "dist_from_sma_50",
        "volatility_5d",
        "volatility_20d",
        "volume_avg_20d",
        "volume_ratio_20d",
        "next_day_return",
    ]

    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def save_features(df: pd.DataFrame) -> None:
    ensure_directories()
    df.to_parquet(OUTPUT_FILE, index=False)


def build_features() -> pd.DataFrame:
    df = load_prices()
    df = add_features(df)
    df = drop_warmup_rows(df)
    save_features(df)
    return df