#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prices.py

A brief description of what this module does.

Author: Andre
Created: 3/29/2026
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

from src.data_ingest.config import (
    TWELVE_DATA_API_KEY,
    BASE_URL,
    DEFAULT_TICKERS,
    PROCESSED_FILE_NAME,
    RAW_FILE_TEMPLATE,
)
from src.data_ingest.validators import run_all_validations
from src.utils.paths import RAW_DIR, PROCESSED_DIR, ensure_directories


class PriceLoader:
    def __init__(self, api_key: str | None = None, pause_seconds: float = 10.0) -> None:
        self.api_key = api_key or TWELVE_DATA_API_KEY
        self.pause_seconds = pause_seconds

        if not self.api_key:
            raise ValueError(
                "Twelve Data API key not found. Add TWELVE_DATA_API_KEY to your .env file."
            )

        ensure_directories()

    def fetch_daily(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        ticker = ticker.upper()
        raw_path = RAW_DIR / RAW_FILE_TEMPLATE.format(ticker=ticker)

        if raw_path.exists() and not force_refresh:
            df_raw = pd.read_csv(raw_path)
        else:
            df_raw = self._download_daily_csv(ticker)
            df_raw.to_csv(raw_path, index=False)

        df_clean = self._standardize_twelve_data_daily(df_raw, ticker)
        return df_clean

    def fetch_many(
        self,
        tickers: Iterable[str] | None = None,
        force_refresh: bool = False,
        save_processed: bool = True,
    ) -> pd.DataFrame:
        tickers = list(tickers) if tickers is not None else DEFAULT_TICKERS
        all_frames: list[pd.DataFrame] = []

        for i, ticker in enumerate(tickers):
            df_ticker = self.fetch_daily(ticker, force_refresh=force_refresh)
            all_frames.append(df_ticker)

            if force_refresh and i < len(tickers) - 1:
                time.sleep(self.pause_seconds)

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        run_all_validations(combined)

        if save_processed:
            processed_path = PROCESSED_DIR / PROCESSED_FILE_NAME
            combined.to_parquet(processed_path, index=False)

        return combined

    def load_processed(self) -> pd.DataFrame:
        processed_path = PROCESSED_DIR / PROCESSED_FILE_NAME
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed file not found at {processed_path}. Run fetch_many() first."
            )
        return pd.read_parquet(processed_path)

    def _download_daily_csv(self, ticker: str) -> pd.DataFrame:
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 5000,
            "format": "JSON",
            "apikey": self.api_key,
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if "status" in payload and payload["status"] == "error":
            raise ValueError(f"Twelve Data error for {ticker}: {payload}")

        if "values" not in payload:
            raise ValueError(f"Unexpected Twelve Data response for {ticker}: {payload}")

        return pd.DataFrame(payload["values"])

    @staticmethod
    def _standardize_twelve_data_daily(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        column_map = {
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        missing = [col for col in column_map if col not in df.columns]
        if missing:
            raise ValueError(
                f"Twelve Data response for {ticker} is missing expected columns: {missing}"
            )

        out = df[list(column_map.keys())].rename(columns=column_map).copy()
        out["ticker"] = ticker.upper()

        out["date"] = pd.to_datetime(out["date"], errors="raise")

        float_cols = ["open", "high", "low", "close"]
        for col in float_cols:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)

        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")
        out["adjusted_close"] = out["close"]

        out = out[
            [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]
        ]

        return out.sort_values("date").reset_index(drop=True)