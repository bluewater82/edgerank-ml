#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_features.py

A brief description of what this module does.

Author: Andre
Created: 3/29/2026
"""

from src.features.features import build_features

if __name__ == "__main__":
    df = build_features()

    print(df.head())
    print()
    print(df.tail())
    print()
    print(df.dtypes)
    print()
    print(f"Rows: {len(df)}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")