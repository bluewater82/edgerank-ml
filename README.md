# EdgeRank

A machine learning-based system for ranking high-confidence trading signals.

## Overview
EdgeRank is a machine learning-driven trading signal system designed to identify high-confidence short-term price movement in equities. The project evolves from a simple data pipeline into a structured, experiment-driven framework for generating, filtering, and evaluating trading signals.

Rather than attempt to predict every market movement, this system focuses on:
- Identifying and acting only on the highest-confidence opportunities

## Project Goals
- Build a clean, modular ML pipeline for financial data
- Explore whether simple technical features contain predictive signal
- Move from raw predictions -> actionable trading decisions
- Develop a disciplined, experiment-driven workflow
- Avoid overfitting and premature scaling

## Core Philosophy
This project is built around a key principle:
- **Train boadly, act selectively**
- The model learns from multiple assets
- The system filters and ranks signals
- Only the strongest opportunities are executed

## System Architecture
### 1. Data Ingestion
- Source: free market data
- Multi-ticker historical OHLCV data
- Cleaned and stardardized format
### 2. Feature Engineering
Features are designed to capture:
- Returns
  - 1-day, 5-day, 20-day returns
- Trend
  - SMA (5, 20, 50)
  - Distance from SMA
- Volatility
  - Rolling standard deviation (5d, 20d)
- Volume
  - Rolling average volume
  - Volume ration
- Target
  - Binary: whether next day closes higher
  - Continuous: next-day return (for back-testing)

## Modeling Approach
**Baseline Model**
- Logistic Regression
- Result: weak signal, poor confidence separation

**Final Model (Current)**
- XGBoost (gradient boosted trees)

**Why XGBoost?**
- Captures nonlinear relationships
- Handles feature interactions
- Produces meaningful probability distributions

## Key Breakthrough
Initial model output:
- ~98% predictions were near 0.50 -> unusable

After XGBoost:
- Wide probability distribution (0.15 -> 0.86)
- Enabled meaningful confidence-based decisions

## Signal Generation
Signals are generated using probability thresholds:
- Long (1): prob_up >= upper threshold
- Short (-1): prob_up <= lower threshold
- No Trade (0): otherwise

Current working threshold strategy:
- upper = 0.65
- lower = 0.35

## Ranking System (Major Upgrade)
Instead of taking all signals that passed the thresholds, a ranking system was implemented to isolate the strongest signals.

**Conviction Score**
- conviction = abs(prob_up - 0.50)

**Daily Selection**
- Rank signals by conviction
- Select top N per day

**Final Configuration**
- top_n_per_day = 3

## Backtesting Framework

**Trade Logic**
- Long -> earn next-day return
- Short -> earn inverse return
- No signal -> no trade

**Metrics Tracked**
- Win rate
- Average return
- Median return
- Total return
- Coverage
- Max gain/loss
- Long vs short distribution

## Experimental Results

**Ranking Sensitivity (Top N)**
- **Top N**:
  - 1: 94 trades, 58.5% win rate, 0.00829 avg return, 0.779 total return
  - 2: 99 trades, 59.6% win rate, 0.00852 avg return, 0.843 total return
  - 3: 100 trades, 60.0% win rate, 0.00887 avg return, 0.887 total return
- **Conclusion**: Signal depth exists beyond just the top candidate

**Threshold Sensitivity (Top 3)**
- **Threshold**
  - 0.63/0.37: 201 trades, 59.3% win rate, 0.00676 avg return, **1.379** total return
  - 0.65/0.35: 100 trades, 60.0% win rate, 0.00887 avg return, 0.887 total return
  - 0.67/0.33: 58 trades, **62.1%** win rate, **0.01176** avg return, 0.682 total return

Tradeoff identified:
- Looser -> more trades, more total return
- Stricter -> higher quality, fewer trades