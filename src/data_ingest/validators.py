#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validators.py

A brief description of what this module does.

Author: Andre
Created: 3/29/2026
"""

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_duplicates(df: pd.DataFrame) -> None:
    dupes = df.duplicated(subset=["date", "ticker"])
    if dupes.any():
        duplicate_rows = df.loc[dupes, ["date", "ticker"]]
        raise ValueError(
            f"Duplicate (date, ticker) rows found:\n{duplicate_rows.head(10)}"
        )


def validate_no_nulls(df: pd.DataFrame) -> None:
    core = ["date", "ticker", "open", "high", "low", "close", "volume"]
    null_counts = df[core].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Null values found in required columns:\n{bad}")


def validate_price_relationships(df: pd.DataFrame) -> None:
    if (df["high"] < df["low"]).any():
        bad_rows = df.loc[df["high"] < df["low"], ["date", "ticker", "high", "low"]]
        raise ValueError(f"Found rows where high < low:\n{bad_rows.head(10)}")

    price_cols = ["open", "high", "low", "close", "adjusted_close"]
    for col in price_cols:
        if (df[col] <= 0).any():
            bad_rows = df.loc[df[col] <= 0, ["date", "ticker", col]]
            raise ValueError(f"Found nonpositive values in {col}:\n{bad_rows.head(10)}")


def validate_volume(df: pd.DataFrame) -> None:
    if (df["volume"] < 0).any():
        bad_rows = df.loc[df["volume"] < 0, ["date", "ticker", "volume"]]
        raise ValueError(f"Found negative volume values:\n{bad_rows.head(10)}")


def run_all_validations(df: pd.DataFrame) -> None:
    validate_required_columns(df)
    validate_no_duplicates(df)
    validate_no_nulls(df)
    validate_price_relationships(df)
    validate_volume(df)