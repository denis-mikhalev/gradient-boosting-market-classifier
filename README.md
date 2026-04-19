# Gradient Boosting Market Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Trains XGBoost classifiers to predict directional price movement (LONG / SHORT / HOLD) on financial time-series data, with dynamic risk controls and real-time Telegram alerts.

An end-to-end XGBoost-based trading signal system for spot and derivatives markets. Trains multi-class direction classifiers (LONG / SHORT / HOLD), runs live predictions on multiple assets in parallel, and aggregates signals with risk metrics — all connected to Telegram for real-time alerts.

**Example signal output:**
```
📈 LONG  |  BTCUSDT 15m
Confidence : 0.74  ·  SMC score: 3 / 5 ✓
Entry : 83,420  ·  SL : 82,800  ·  TP : 84,500
RRR   : 1 : 2.1   ·  EV : +0.41%
```

---

## Tech Stack

- **XGBoost** — gradient boosted tree classifier
- **scikit-learn** — time-series cross-validation, metrics
- **pandas / numpy** — data manipulation
- **ta** — technical analysis indicators
- **ccxt** — Binance OHLCV data fetching
- **python-dotenv** — environment variable management
- **matplotlib** — training diagnostics and efficient frontier plots
- **psutil** — process monitoring for parallel model processes

---

## Overview

| Component | File | Description |
|-----------|------|-------------|
| **Model training** | `train_model.py` | Downloads OHLCV history from Binance, engineers 51+ features (single + multi-timeframe), labels with TP/SL horizon, trains XGBoost with time-series CV |
| **Live prediction** | `predict.py` | Loads a trained model, runs inference on the latest candle, sends Telegram alerts on signal changes |
| **Signal aggregator** | `signal_aggregator.py` | Polls signal files from all running models, computes ensemble consensus, forwards to Telegram |
| **SMC filter** | `smc_filter.py` | Smart Money Concepts overlay: filters out signals that conflict with order blocks, BOS/CHOCH, and liquidity sweeps |
| **Auto launcher** | `AutoLauncher.py` | Spawns and monitors parallel prediction processes for all models in the registry |
| **Watchlist** | `WatchlistJSONManager.py` | Manages the JSON watchlist of active symbols/models |
| **Risk metrics** | `risk_metrics.py` | Expected value, profit factor, max drawdown, Sharpe ratio calculations |

---

## Features

- **51+ engineered features** across four groups — momentum (RSI, MACD, Stochastic), volatility (Bollinger Bands, ATR, Keltner), volume (OBV, VWAP, volume delta), and multi-timeframe confirmation (15m / 1h / 4h)
- **3-class XGBoost classifier** (LONG / SHORT / HOLD) with time-series cross-validation, stratified holdout backtest, and bootstrap confidence intervals on Profit Factor
- **Threshold optimization** via efficient frontier scatter — Profit Factor vs. max drawdown across the full probability grid; selects the plateau with the best PF/DD trade-off
- **SMC confluence filter** — 6 independent Smart Money checks (order blocks, BOS/CHOCH, liquidity sweeps, VWAP, volume, BTC dominance context); signal approved only when score ≥ configurable minimum
- **Optimal lookback search** — grid search from 50 to 3 000 days (configurable step) per symbol; auto-updates `watchlist.json` with the winning period
- **Parallel multi-asset live inference** — all models run as isolated processes with heartbeat monitoring and automatic restart on failure
- **Ensemble signal aggregation** — `signal_aggregator.py` polls all active model outputs, computes consensus direction, and forwards a portfolio-level summary to Telegram
- **Per-trade risk metrics** — Expected Value, Risk:Reward ratio, break-even probability, and `edge_ok` / `prob_ok` gate computed from configurable `risk_config.json`

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your Telegram bot token + chat ID
```

### 3. Train a model
```bash
python train_model.py --symbol BTC/USDT --timeframe 15m --days 1500 --backtest-days 120
```

### 4. Run live predictions
```bash
python predict.py \
  --meta models_v2/meta_BTCUSDT_15m_1500d_bt120d_20251031_122314.json \
  --model models_v2/xgb_BTCUSDT_15m_1500d_bt120d_20251031_122314.json \
  --symbol BTCUSDT --tf 15m
```

### 5. Launch all models in parallel
```bash
python LaunchAllModels.py
```

### 6. Start signal aggregator (with Telegram)
```bash
python signal_aggregator.py
```

---

## Project Structure

```
├── train_model.py              # XGBoost model training pipeline
├── predict.py                  # Live inference engine
├── signal_aggregator.py        # Multi-model signal aggregation
├── smc_filter.py               # Smart Money Concepts filter
├── risk_metrics.py             # Risk / EV calculations
├── telegram_sender.py          # Telegram notification helper
├── AutoLauncher.py             # Parallel model launcher + watchdog
├── LaunchAllModels.py          # Launch all registry models (one-shot)
├── LaunchAllModelsV2.py        # Launcher v2 with enhanced logging
├── BatchTrainModelsV2.py       # Batch-train a list of symbol configs
├── ModelTrainingMediator.py    # Coordinates training across timeframes
├── ModelLaunchNotifier.py      # Notify on model start / error
├── HeartbeatManager.py         # Heartbeat ping for live processes
├── ModelVersionManager.py      # Version tracking for trained models
├── CleanupNonOptimalModels.py  # Remove models below quality threshold
├── CleanupUnprofitableModels.py# Remove models with negative EV
├── WatchlistJSONManager.py     # CRUD for watchlist.json
├── WatchlistParser.py          # Watchlist format parsing utilities
├── WatchlistAdapter.py         # Legacy ↔ new watchlist adapter
├── WatchlistAutoUpdaterJSON.py # Auto-populate watchlist from model metadata
├── UpdateWatchlistWithModelInfo.py # Sync model stats into watchlist
├── OptimalPeriodSearch.py      # Grid search for best training period
├── ExtendedPeriodSearch.py     # Extended period sweep with more candidates
├── CompareOptimalPeriodSelection.py # Compare period search strategies
├── analyze_market_conditions.py # Market regime analysis tool
├── compare_timeframes.py       # Side-by-side timeframe performance comparison
├── models_v2/                  # Model metadata JSON files
│   ├── xgb_*.json              # XGBoost model feature/threshold data
│   └── meta_*.json             # Training metrics and config snapshots
├── risk_config.json            # Risk parameters (TP%, SL%, position sizing)
├── model_thresholds.json       # Per-symbol signal probability thresholds
└── watchlist.json              # Active symbols and model registry
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes (for alerts) | Telegram bot API token |
| `TELEGRAM_CHAT_ID` | Yes (for alerts) | Telegram chat / channel ID |
| `BINANCE_API_KEY` | No | Binance API key (public data works without it) |
| `BINANCE_API_SECRET` | No | Binance API secret |

Copy `.env.example` to `.env` and fill in your values. **Never commit `.env` to version control.**


