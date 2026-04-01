#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_per_ticker.py

Runner file for per-ticker model.

Author: Andre
Created: 3/31/2026
"""

from src.models.per_ticker_model import run_all_tickers

if __name__ == "__main__":
    run_all_tickers(
        min_rows=2000,
        upper_threshold=0.65,
        lower_threshold=0.35,
    )