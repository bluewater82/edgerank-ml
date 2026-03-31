#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_loader.py

A brief description of what this module does.

Author: Andre
Created: 3/29/2026
"""

from src.data_ingest.prices import PriceLoader

if __name__ == "__main__":
    loader = PriceLoader()
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

    df = loader.fetch_many(
        tickers=tickers,
        force_refresh=True,
        save_processed=True,
    )

    print(df.head())
    print()
    print(df.tail())
    print()
    print(df.dtypes)
    print()
    print(f"Rows: {len(df)}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")