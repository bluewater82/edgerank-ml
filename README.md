# 🚀 EdgeRank

> A machine learning system for identifying and executing high-confidence trading opportunities.

---

## 🧠 Overview

**EdgeRank** is a machine learning-driven trading system designed to identify **high-probability short-term opportunities in equities**.

Instead of predicting everything, EdgeRank focuses on:

> **Acting only when confidence is high and the best opportunity is clear.**

The system has evolved into a **hybrid architecture** combining:
- 🎯 Per-ticker specialization  
- 🏆 Cross-asset competition  
- ⚖️ Strict portfolio selection  

---

## 🎯 Core Philosophy

### Train broadly. Act selectively.

- Models learn from historical data per asset  
- Signals are filtered by confidence thresholds  
- Opportunities compete across the market  
- Only the **best trade(s) per day** are executed  

---

## 🏗️ System Architecture

```
Data → Features → Models → Signals → Ranking → Trades → Backtest
```

### 1. 📥 Data Ingestion
- OHLCV market data (free sources)
- Multi-ticker dataset
- Cleaned + standardized

---

### 2. 🧮 Feature Engineering

#### Returns
- 1d, 5d, 20d

#### Trend
- SMA (5, 20, 50)
- Distance from SMA

#### Volatility
- Rolling std (5d, 20d)

#### Volume
- Volume averages
- Volume ratios

#### Targets
- Binary: next-day up/down  
- Continuous: next-day return  

---

## 🤖 Modeling

### Baseline
- Logistic Regression  
- ❌ Result: weak signal (~0.50 collapse)

### Current Model
- 🌳 **XGBoost**

#### Why?
- Captures nonlinear patterns  
- Handles interactions  
- Produces useful probability spread  

---

## 💡 Key Breakthrough

Before:
```
Probabilities ~0.50 → unusable
```

After XGBoost:
```
Probabilities ~0.15 → 0.86
```

✅ Enabled confidence-based trading decisions

---

## 🔄 System Evolution

### Phase 1 — Pooled Model
- One model for all tickers  
- Ranked signals globally  

---

### Phase 2 — Asset Filtering
- Removed weak contributors  
- Improved stability  

---

### Phase 3 — Per-Ticker Models
- One model per asset  
- Revealed asset-specific behavior  

> 💡 Not all tickers behave the same

---

### Phase 4 — Hybrid System (Current)

- Per-ticker models generate signals  
- Signals compete globally  
- Only the **top opportunity per day** is selected  

---

## 📊 Signal Logic

```
Long  → prob_up ≥ 0.70
Short → prob_up ≤ 0.30
Else  → no trade
```

---

## 🏆 Ranking & Selection

### Conviction
```
conviction = |prob_up - 0.50|
```

### Selection Rule
```
top_n_per_day = 1
```

> The system acts as a **daily best-opportunity selector**

---

## 📈 Backtesting

### Trade Logic
- Long → next-day return  
- Short → inverse return  

### Metrics
- Win rate  
- Avg / median return  
- Total return  
- Coverage  
- Per-ticker stats  

---

## 📊 Current Performance

**Configuration**
- Thresholds: 0.70 / 0.30  
- Top trades per day: 1  

**Results**
- Coverage: ~5%  
- Win rate: ~55%  
- Avg return: ~0.24%  
- Total return: ~90%+  

---

## 🧠 Asset Insights

### 🟢 Strong
- NVDA (primary alpha driver)
- AAPL (consistent)
- SPY (stable)

### 🔴 Weak
- GOOG
- XLK
- NFLX

> 💡 Not all assets deserve equal participation

---

## ⚙️ System Behavior

### Strengths
- Highly selective  
- Robust over time  
- Interpretable decisions  

### Observations
- Regime-dependent performance  
- Strong in trending markets  
- Flat periods exist  

---

## 🧪 Experiment Tracking

- CSV logging system  
- Tracks:
  - thresholds  
  - ranking depth  
  - performance  

---

## 🧠 What This Demonstrates

- Technical features contain signal  
- Nonlinear models outperform linear ones  
- **Selectivity > prediction volume**  
- Ranking is critical  
- Specialization improves results  

---

## ⚠️ Limitations

- No transaction costs  
- No slippage  
- Limited tickers  
- No regime detection (yet)  
- No macro inputs  

---

## 🚀 Future Work

### Short-Term
- Refine ticker set  
- Test top_n = 2  
- Add conviction filters  

### Medium-Term
- Regime detection  
- Dynamic thresholds  
- Risk controls  

### Long-Term
- Expand universe  
- Ensemble models  
- Optional macro features  

---

## 🧭 Summary

EdgeRank is not just a model.

> It is a **decision system** that identifies, ranks, and executes the highest-confidence opportunities in the market.

---

## 👤 Author

Andre — Computer Science @ UNM  
Focus: AI / ML, Systems, Data Engineering

