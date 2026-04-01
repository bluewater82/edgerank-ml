#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hybrid.py

Runner for hybrid model

Author: Andre
Created: 3/31/2026
"""

from src.models.hybrid_model import run_hybrid_backtest

if __name__ == "__main__":
    run_hybrid_backtest(
        min_rows=2000,
        upper_threshold=0.70,
        lower_threshold=0.30,
        top_n_per_day=1,
        allowed_tickers=[
            "AAPL",
            "GOOG",
            "NFLX",
            "NVDA",
            "SPY",
            "XLK",
        ],
        experiment_name="hybrid_top3_base",
    )