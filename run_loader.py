#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_loader.py

Used in conjuction with prices getter module to pull specific ticker prices

Author: Andre
Created: 3/29/2026
"""

from src.data_ingest.prices import PriceLoader

TICKERS = [
    # existing
    "AAPL", "NVDA", "SPY", "QQQ",

    # high vol
    "AMD", "TSLA", "META", "NFLX",

    # mega cap
    "MSFT", "GOOG", "AMZN",

    # ETFs
    "IWM", "DIA",

    # sectors
    "XLF", "XLK", "XLE"
]

if __name__ == "__main__":
    loader = PriceLoader()
    tickers = TICKERS

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