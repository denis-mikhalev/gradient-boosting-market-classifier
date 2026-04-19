# Gradient Boosting Market Classifier

An end-to-end XGBoost-based trading signal system for cryptocurrency markets. Trains multi-class direction classifiers (LONG / SHORT / HOLD), runs live predictions on multiple assets in parallel, and aggregates signals with risk metrics — all connected to Telegram for real-time alerts.

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

- **51+ engineered features**: RSI, MACD, Bollinger Bands, ATR, volume profile, price action patterns, multi-timeframe confirmation (15m / 1h / 4h)
- **3-class XGBoost classifier** with time-series cross-validation and holdout evaluation
- **Threshold optimization** via efficient frontier (max drawdown vs. profit factor)
- **SMC confluence filter**: minimum confluence score configurable, filters low-quality signals
- **Parallel multi-asset live trading**: all models run concurrently with heartbeat monitoring
- **Watchlist auto-update**: model metadata (accuracy, feature importance, thresholds) flows into the watchlist
- **Optimal lookback period search**: grid search over historical period length to maximize model quality
- **Telegram integration**: signal alerts, model launch notifications, ensemble summaries

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
python train_model.py --symbol BTCUSDT --tf 1h --days 500 --horizon 12
```

### 4. Run live predictions
```bash
python predict.py --model models_v2/xgb_BTCUSDT_1h_500d.json
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
