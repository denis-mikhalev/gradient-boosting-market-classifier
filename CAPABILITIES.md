# Capabilities Reference

Full reference of all CLI commands, parameters, and system features.

---

## Table of Contents

- [Model Training — `train_model.py`](#model-training--train_modelpy)
- [Live Prediction — `predict.py`](#live-prediction--predictpy)
- [Auto Launcher — `AutoLauncher.py` / `LaunchAllModels.py`](#auto-launcher--autolauncherpy--launchallmodelspy)
- [Signal Aggregator — `signal_aggregator.py`](#signal-aggregator--signal_aggregatorpy)
- [SMC Filter — `smc_filter.py`](#smc-filter--smc_filterpy)
- [Optimal Period Search — `OptimalPeriodSearch.py`](#optimal-period-search--optimalperiodsearchpy)
- [Batch Training — `BatchTrainModelsV2.py`](#batch-training--batchtrainmodelsv2py)
- [Model Cleanup — `CleanupNonOptimalModels.py`](#model-cleanup--cleanupnonoptimalmodelspy)
- [Risk Metrics — `risk_metrics.py`](#risk-metrics--risk_metricspy)
- [Watchlist Management](#watchlist-management)
- [Configuration Files](#configuration-files)
- [VS Code Tasks](#vs-code-tasks)

---

## Model Training — `train_model.py`

Fetches OHLCV data from Binance, engineers 51+ features, labels candles with TP/SL horizon, trains a 3-class XGBoost classifier (LONG / SHORT / HOLD) with time-series cross-validation and holdout backtest.

```bash
python train_model.py [OPTIONS]
```

### Core Parameters

| Flag | Default | Description |
|---|---|---|
| `--symbol` | `BTC/USDT` | Trading pair (e.g. `ETH/USDT`, `SOL/USDT`) |
| `--timeframe` | `30m` | Candle timeframe (`15m`, `30m`, `1h`, `4h`) |
| `--days` | `1825` | Historical lookback in days |
| `--tp_pct` | `0.008` | Take-profit target as a fraction (0.008 = 0.8%) |
| `--sl_pct` | `0.006` | Stop-loss level as a fraction (0.006 = 0.6%) |
| `--train-split` | `0.8` | Train/test split ratio |
| `--backtest-days` | — | Fixed holdout window in days (alternative to `--train-split`) |

### Geometry & Labeling Overrides

| Flag | Description |
|---|---|
| `--geom-tp START,END,STEP` | Override TP multiplier grid scan (e.g. `1.0,6.0,0.25`) |
| `--geom-sl START,END,STEP` | Override SL multiplier grid scan (e.g. `0.4,2.0,0.2`) |
| `--horizon N` | Override forward horizon in bars for labeling |
| `--dynamic-tp-percentile P` | Adjust TP target to MFE percentile after calibration (experimental) |
| `--dynamic-tp-auto-select` | `recommend` or `apply` — auto-select most robust dynamic TP percentile |
| `--apply-dynamic-tp` | Apply dynamic TP adjustment immediately |
| `--no-geom-auto-expand` | Disable automatic TP range expansion |
| `--no-anti-saturation` | Disable anti-saturation penalty in TP selection |

### Threshold & Plateau Detection

| Flag | Default | Description |
|---|---|---|
| `--plateau-pf-margin` | `0.05` | Include thresholds within this fraction of max PF (5% = within 5% of best) |
| `--plateau-min-trades` | `30` | Minimum trades to include a threshold in plateau analysis |
| `--sensitivity-min-trades` | `30` | Minimum trades per percentile for sensitivity pass |
| `--sensitivity-thresholds` | — | Override threshold subset for sensitivity scan (e.g. `0.73,0.75,0.78`) |
| `--sensitivity-bootstrap-pf` | `0` | Bootstrap samples for PF confidence interval per sensitivity percentile |

### Bootstrap & Statistics

| Flag | Default | Description |
|---|---|---|
| `--bootstrap-pf` | `0` | Bootstrap samples for PF confidence interval after backtest |
| `--baseline-no-cap` | — | Run a no-cap baseline threshold scan alongside dynamic TP run |
| `--dynamic-tp-sensitivity` | — | Comma-separated alternative percentiles for sensitivity pass |

### Output & Visualisation

| Flag | Description |
|---|---|
| `--silent` | Suppress matplotlib graph display |
| `--no-versioning` | Overwrite existing model files instead of timestamping |
| `--no-debug-trades` | Disable saving per-threshold trade logs |
| `--frontier-from-meta PATH` | Plot PF vs max-drawdown efficient frontier from a saved meta JSON and exit |
| `--frontier-show` | Force show frontier plot even in `--silent` mode |
| `--save-frontier [NAME]` | Auto-save frontier plot after training (`auto` = auto-named) |

### Example

```bash
# Train BTC 15-minute model with 1500-day lookback and 120-day holdout
python train_model.py --symbol BTC/USDT --timeframe 15m --days 1500 --backtest-days 120

# Research run: custom geometry, bootstrap CI, save frontier
python train_model.py --symbol ETH/USDT --days 2000 \
    --geom-tp 1.0,6.0,0.25 --geom-sl 0.4,2.0,0.2 \
    --bootstrap-pf 500 --save-frontier
```

**Output artifacts** saved to `models_v2/`:
- `xgb_<SYMBOL>_<TF>_<DAYS>d_bt<BACKTEST>d_<TIMESTAMP>.json` — model weights
- `meta_<SYMBOL>_<TF>_<DAYS>d_bt<BACKTEST>d_<TIMESTAMP>.json` — full metadata (CV scores, backtest stats, threshold grid, feature importances)

---

## Live Prediction — `predict.py`

Loads a trained model, polls Binance for the latest candle on a configurable interval, runs inference, optionally filters through the SMC layer, and sends Telegram alerts on signal changes.

```bash
python predict.py --meta META_JSON --model MODEL_JSON [OPTIONS]
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--meta` | *(required)* | Path to `meta_*.json` artifact |
| `--model` | *(required)* | Path to `xgb_*.json` model file |
| `--symbol` | `BTCUSDT` | Trading symbol |
| `--tf` | `30m` | Candle timeframe |
| `--limit` | `1000` | Number of candles to fetch per inference cycle |
| `--min_conf` | `0.66` | Minimum model confidence to emit a signal |
| `--atr_sl_mul` | `1.5` | ATR multiplier for dynamic stop-loss |
| `--atr_tp_mul` | `2.5` | ATR multiplier for dynamic take-profit |
| `--interval` | `60` | Polling interval in seconds |
| `--enable_smc_filter` | off | Enable Smart Money Concepts confluence filter |
| `--smc_min_confluence` | `3` | Minimum SMC confluence score required (1–5) |

### Signal Output

Each emitted signal includes:
- Direction: **LONG** / **SHORT** / **HOLD**
- Model confidence probability
- Dynamic ATR-based SL / TP levels
- Expected Value and Risk:Reward ratio (from `risk_metrics.py`)
- SMC confluence score and detailed reasons (when filter enabled)
- Telegram notification with full signal card

### Example

```bash
# Run live prediction with SMC filter, polling every 60 s
python predict.py \
    --meta models_v2/meta_BTCUSDT_15m_1500d_bt120d_20251031_122314.json \
    --model models_v2/xgb_BTCUSDT_15m_1500d_bt120d_20251031_122314.json \
    --symbol BTCUSDT --tf 15m \
    --enable_smc_filter --smc_min_confluence 3
```

---

## Auto Launcher — `AutoLauncher.py` / `LaunchAllModels.py`

Scans `models_v2/` for all valid model pairs, spawns a separate `predict.py` process for each model, and monitors them with automatic restart on crash.

### `LaunchAllModels.py`

```bash
python LaunchAllModels.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--list` | — | Print discovered models without launching |
| `--interval` | `60` | Prediction polling interval passed to each `predict.py` process |
| `--enable_smc_filter` | off | Enable SMC filter for all spawned processes |
| `--smc_min_confluence` | `3` | SMC confluence threshold (`2`–`5`) |

### `AutoLauncher.py`

Advanced launcher that additionally:
- Runs quality checks on each model before launch
- Generates `cleanup_recommendations.json` for `ModelVersionManager.py`
- Only launches the latest version of each symbol/timeframe combination
- Emits Telegram notifications on model launch and errors via `ModelLaunchNotifier`

### Examples

```bash
# List all available models
python LaunchAllModels.py --list

# Launch all models with SMC filter (confluence ≥ 4)
python LaunchAllModels.py --enable_smc_filter --smc_min_confluence 4

# Launch with quality check and version management
python AutoLauncher.py
```

---

## Signal Aggregator — `signal_aggregator.py`

Polls `signals/*.json` written by running `predict.py` processes every 5 minutes. Computes ensemble consensus across all active models and forwards a consolidated summary to Telegram. Also calculates portfolio-level risk metrics (Expected Value, Profit Factor) using `risk_metrics.py`.

```bash
python signal_aggregator.py
```

- Reads all signal files younger than 30 minutes
- Deduplicates already-processed signals
- Sends ensemble summary: active model count, consensus direction, aggregate EV
- Logs all activity to `signal_aggregator.log`

---

## SMC Filter — `smc_filter.py`

Smart Money Concepts overlay that validates XGBoost signals against institutional price action logic before a signal is forwarded to Telegram.

### Confluence Checks (each scores +1)

| Check | Logic |
|---|---|
| **VWAP alignment** | Price is above VWAP for LONG / below for SHORT |
| **Order Block** | Recent bullish/bearish order block present near current price |
| **Liquidity Sweep** | Prior swing high/low was swept before the move |
| **Break of Structure (BOS/CHOCH)** | Higher-high / lower-low structure confirmed |
| **Volume Confirmation** | Current volume exceeds rolling average |
| **BTC Dominance context** | Altcoin signals filtered by BTC trend alignment |

### Usage

```python
from smc_filter import SMCFilter

smc = SMCFilter(min_confluence_score=3)
result = smc.validate_signal(df_ohlcv, signal="LONG", confidence=0.74, symbol="ETHUSDT")
# result: { approved, confluence_score, reasons, recommendation }
```

Signal is approved only when `confluence_score >= min_confluence_score`.

---

## Optimal Period Search — `OptimalPeriodSearch.py`

Grid-searches over historical lookback lengths to find the period that maximises model quality (Profit Factor, Win Rate) for a given symbol.

```bash
python OptimalPeriodSearch.py [SYMBOLS ...] [OPTIONS]
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--symbols` | all active | Space-separated list of symbols to search |
| `--watchlist`, `-w` | `watchlist.json` | Watchlist file |
| `--timeframe`, `-t` | `30m` | Candle timeframe |
| `--start-days`, `-s` | `50` | Minimum period to test (days) |
| `--max-days`, `-m` | `1000` | Maximum period to test (days) |
| `--step` | `50` | Step size between periods (days) |
| `--backtest-days`, `-b` | `14` | Holdout window per candidate model |
| `--min-trades` | `5` | Minimum backtest trades to consider a result valid |
| `--keep-top-n` | `1` | Number of top models to retain per symbol |
| `--quality-preset` | `balanced` | `conservative` / `balanced` / `aggressive` |
| `--auto-update`, `-u` | — | Automatically write best period back to `watchlist.json` |
| `--no-cleanup` | — | Keep all intermediate model files |
| `--results-dir` | `optimal_period_analysis` | Output directory for result JSONs |
| `--geom-tp`, `--geom-sl` | — | Pass-through geometry overrides to each training run |
| `--horizon` | — | Pass-through horizon override |
| `--silent`, `-q` | — | Suppress output |
| `--verbose`, `-v` | — | Verbose output |

### Example

```bash
# Search BTCUSDT and ETHUSDT, periods 200–3000 days, step 100, auto-update watchlist
python OptimalPeriodSearch.py BTCUSDT ETHUSDT \
    --start-days 200 --max-days 3000 --step 100 \
    --backtest-days 30 --min-trades 20 \
    --quality-preset conservative --auto-update
```

---

## Batch Training — `BatchTrainModelsV2.py`

Trains models for all symbols in `watchlist.json` sequentially, reading optimal periods and parameters from the watchlist.

```bash
python BatchTrainModelsV2.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--watchlist` | `watchlist.json` | Watchlist source file |
| `--period N` | — | Override period for all symbols |
| `--backtest-days N` | — | Override holdout window for all symbols |
| `--active-only` | — | Skip inactive symbols |
| `--silent` | — | Suppress graph display during training |

### Example

```bash
# Retrain all active symbols using periods from watchlist
python BatchTrainModelsV2.py --active-only --silent

# Force all to a fixed 1500-day period
python BatchTrainModelsV2.py --period 1500 --backtest-days 90
```

---

## Model Cleanup — `CleanupNonOptimalModels.py`

Removes model files whose period does not match the optimal period recorded in `watchlist.json`. Keeps `models_v2/` tidy after period search runs.

```bash
python CleanupNonOptimalModels.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--models-dir` | `models_v2` | Directory to scan |
| `--watchlist` | `watchlist.json` | Reference watchlist |
| `--dry-run` | — | Print what would be deleted without deleting |
| `--force` | — | Delete without confirmation prompt |

```bash
# Preview what would be removed
python CleanupNonOptimalModels.py --dry-run

# Execute cleanup
python CleanupNonOptimalModels.py --force
```

---

## Risk Metrics — `risk_metrics.py`

Utility module consumed by `predict.py` and `signal_aggregator.py`. Calculates per-signal edge metrics from `risk_config.json`.

### Computed Values

| Metric | Description |
|---|---|
| `rrr` | Risk:Reward ratio (target_pct / stop_pct) |
| `p_be` | Break-even probability — minimum win rate needed for positive EV |
| `ev_naive_pct` | Expected value in % using model confidence as win probability proxy |
| `edge_ok` | Whether net move (after fees) exceeds `min_edge_pct` |
| `prob_ok` | Whether model confidence ≥ break-even probability |

### `risk_config.json` Structure

```json
{
  "deposit": 10000,
  "risk_pct": 0.01,
  "leverage": 10,
  "default_round_trip_cost_pct": 0.003,
  "min_edge_pct": 0.005,
  "symbols": {
    "BTCUSDT": { "min_edge_pct": 0.007, "round_trip_cost_pct": 0.0022 }
  }
}
```

All fields are optional — sensible defaults apply if the file is missing.

---

## Watchlist Management

The `watchlist.json` file is the central registry of active symbols and their optimal model parameters. Several scripts maintain it:

| Script | Purpose |
|---|---|
| `WatchlistJSONManager.py` | Core CRUD operations with full history tracking |
| `WatchlistParser.py` | Reads watchlist entries for batch training |
| `WatchlistAdapter.py` | Compatibility layer for legacy callers |
| `WatchlistAutoUpdaterJSON.py` | Writes optimal period results back after `OptimalPeriodSearch` runs |
| `UpdateWatchlistWithModelInfo.py` | Enriches entries with trained model metadata (threshold, PF, win rate) |
| `CompareOptimalPeriodSelection.py` | Audits period selection logic across analysis runs |

### Watchlist Entry Schema

```json
{
  "symbol": "BTCUSDT",
  "period": 1500,
  "active": true,
  "best_model_file": "xgb_BTCUSDT_15m_1500d_bt120d_20251031_122314.json",
  "optimization_history": [ ... ],
  "models": [
    {
      "period": 1500,
      "model_file": "xgb_BTCUSDT_15m_1500d_bt120d_20251031_122314.json",
      "threshold": 0.62,
      "profit_factor": 2.3,
      "win_rate": 65.0,
      "trades": 36
    }
  ]
}
```

---

## Configuration Files

| File | Purpose |
|---|---|
| `.env` | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `BINANCE_API_KEY`, `BINANCE_API_SECRET` |
| `.env.example` | Template — copy to `.env` and fill in values |
| `risk_config.json` | Per-symbol risk/fee parameters for EV calculation |
| `model_thresholds.json` | Override confidence thresholds per symbol outside model metadata |
| `watchlist.json` | Active symbol registry with optimal periods and model history |

---

## VS Code Tasks

Pre-configured tasks in `.vscode/tasks.json` for one-click launch:

| Task | Description |
|---|---|
| **Start All Available Models** | Launch `LaunchAllModels.py` — spawns all models in `models_v2/` |
| **Start All Available Models (SMC Filter)** | Same with `--enable_smc_filter --smc_min_confluence 3` |
| **Start Advanced Models** | Launch `AutoLauncher.py` with quality checks |
| **Start Signal Aggregator** | Run `signal_aggregator.py` in background |
| **Start Trading System** | Sequential: Advanced Models → Signal Aggregator |
| **Start All Available Models + Aggregator** | Parallel: all models + aggregator |
| **Start All Available Models + Aggregator (SMC Filter)** | Same with SMC filter enabled |
| **Start All Predictions** | Alias for all models via `LaunchAllModels.py` |
| **Start Trading System (Legacy)** | Sequential: All Predictions → Signal Aggregator |
