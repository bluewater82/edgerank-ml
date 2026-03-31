#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

A brief description of what this module does.

Author: Andre
Created: 3/29/2026
"""

import os
from dotenv import load_dotenv

load_dotenv()

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

BASE_URL = "https://api.twelvedata.com/time_series"

DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
]

RAW_FILE_TEMPLATE = "daily_{ticker}.csv"
PROCESSED_FILE_NAME = "daily_prices.parquet"