#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_model.py

Basic linear regression model used as a baseline. This is the floor model.

Author: Andre
Created: 3/30/2026
"""

from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, split_ratio: float = 0.8):
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def inspect_coefficients(model):
    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)

    print("\n=== Feature Coefficients ===")
    print(coef_df)


def evaluate_standard(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n=== Standard Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def evaluate_binary_threshold(model, X_test, y_test, threshold: float = 0.55):
    probs_up = model.predict_proba(X_test)[:, 1]
    y_pred = (probs_up >= threshold).astype("int64")

    print(f"\n=== Binary Threshold Evaluation (threshold = {threshold:.2f}) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    pred_up_rate = y_pred.mean()
    print(f"Predicted UP rate: {pred_up_rate:.4f}")


def evaluate_selective_signals(
    model,
    X_test,
    y_test,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
):
    probs_up = model.predict_proba(X_test)[:, 1]

    eval_df = pd.DataFrame(
        {
            "prob_up": probs_up,
            "actual": y_test.to_numpy(),
        }
    )

    eval_df["signal"] = -1
    eval_df.loc[eval_df["prob_up"] >= upper_threshold, "signal"] = 1
    eval_df.loc[eval_df["prob_up"] <= lower_threshold, "signal"] = 0

    acted = eval_df[eval_df["signal"] != -1].copy()

    print(
        f"\n=== Selective Signal Evaluation "
        f"(upper = {upper_threshold:.2f}, lower = {lower_threshold:.2f}) ==="
    )

    total_rows = len(eval_df)
    acted_rows = len(acted)
    coverage = acted_rows / total_rows if total_rows else 0.0

    print(f"Total test rows: {total_rows}")
    print(f"Rows with signal: {acted_rows}")
    print(f"Coverage: {coverage:.4f}")

    if acted.empty:
        print("No rows met the threshold criteria.")
        return

    accuracy = accuracy_score(acted["actual"], acted["signal"])
    print(f"Accuracy on acted rows: {accuracy:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(acted["actual"], acted["signal"]))
    print()

    print("Classification Report:")
    print(classification_report(acted["actual"], acted["signal"]))

    print("Signal counts:")
    print(acted["signal"].value_counts().sort_index())

    print("\nProbability summary on acted rows:")
    print(acted["prob_up"].describe())


def probability_summary(model, X_test):
    probs_up = model.predict_proba(X_test)[:, 1]
    s = pd.Series(probs_up, name="prob_up")

    print("\n=== Probability Summary ===")
    print(s.describe())

    print("\nProbability bucket counts:")
    bins = pd.cut(
        s,
        bins=[0.0, 0.4, 0.45, 0.55, 0.6, 1.0],
        include_lowest=True,
    )
    print(bins.value_counts().sort_index())


def run_pipeline():
    df = load_data()
    train, test = time_split(df)

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    model = train_model(X_train, y_train)

    evaluate_standard(model, X_test, y_test)
    probability_summary(model, X_test)
    evaluate_binary_threshold(model, X_test, y_test, threshold=0.55)
    evaluate_selective_signals(
        model,
        X_test,
        y_test,
        upper_threshold=0.55,
        lower_threshold=0.45,
    )

    inspect_coefficients(model)


if __name__ == "__main__":
    run_pipeline()