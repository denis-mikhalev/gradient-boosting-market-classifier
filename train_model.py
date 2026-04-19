# CreateModel-2.py
# — Качает расширенную историю с Binance, строит фичи (вкл. мультитаймфрейм),
# — формирует target по TP/SL за фиксированный горизонт,
# — обучает XGBoost (3 класса: LONG/SHORT/HOLD),
# — делает time-series CV и holdout-оценку,
# — сохраняет артефакты (модель + метаданные/фичлисты).

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
import ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

# =====================
# Конфиг
# =====================
def plot_efficient_frontier(meta_path: str, show_plot: bool = True, save_path: Optional[str] = None):
    """Efficient frontier: X=max_drawdown_pct, Y=profit_factor, size=total_trades."""
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    all_results = None
    # 1) New primary location: threshold_optimization.all_results
    if isinstance(meta.get('threshold_optimization'), dict) and 'all_results' in meta['threshold_optimization']:
        all_results = meta['threshold_optimization']['all_results']
    # 2) Legacy: labeling.all_results
    if all_results is None and isinstance(meta.get('labeling'), dict) and 'all_results' in meta['labeling']:
        all_results = meta['labeling']['all_results']
    # 3) Flat top-level (very early versions)
    if all_results is None:
        all_results = meta.get('all_results')
    if not all_results:
        # Fallback: attempt to reconstruct from per-threshold trades CSV files
        base_dir = os.path.dirname(meta_path)
        base_symbol = meta.get('config', {}).get('symbol') or meta.get('symbol') or ''
        tf = (meta.get('config', {}) or {}).get('tf_train') or '30m'
        # pattern trades_<SYMBOL>_<tf>_thrXpp.csv
        import glob, re
        pattern = os.path.join(base_dir, f"trades_{base_symbol.replace('/','')}*thr*.csv")
        files = glob.glob(pattern)
        recon = []
        for fp in files:
            try:
                df_t = pd.read_csv(fp)
                if 'pnl' not in df_t.columns and 'pnl_pct' in df_t.columns:
                    # derive USD pnl relative to first equity if possible
                    df_t['pnl'] = df_t['pnl_pct']
                threshold_match = re.search(r'thr(\d+p\d+)', fp)
                if not threshold_match:
                    continue
                thr_str = threshold_match.group(1).replace('p','.')
                thr_val = float(thr_str)
                if 'pnl_pct' in df_t.columns:
                    gains = df_t[df_t['pnl_pct'] > 0]['pnl_pct'].sum()
                    losses = -df_t[df_t['pnl_pct'] < 0]['pnl_pct'].sum()
                else:
                    gains = df_t[df_t['pnl'] > 0]['pnl'].sum()
                    losses = -df_t[df_t['pnl'] < 0]['pnl'].sum()
                profit_factor = (gains / losses) if losses > 0 else np.nan
                win_rate = (df_t['pnl'] > 0).mean() if 'pnl' in df_t.columns else (df_t['pnl_pct'] > 0).mean()
                # max drawdown from equity columns if present
                if 'equity_after' in df_t.columns:
                    eq = df_t['equity_after'].values
                    peak = -np.inf
                    dd = 0.0
                    for v in eq:
                        peak = max(peak, v)
                        dd = max(dd, (peak - v) / peak * 100 if peak > 0 else 0)
                    mdd = dd
                else:
                    mdd = np.nan
                recon.append({
                    'threshold': thr_val,
                    'total_trades': len(df_t),
                    'profit_factor': float(profit_factor) if profit_factor == profit_factor else None,
                    'win_rate': float(win_rate),
                    'max_drawdown_pct': float(mdd),
                })
            except Exception:
                continue
        if not recon:
            raise ValueError('No all_results found and reconstruction from trades_* failed')
        all_results = recon
    df_front = pd.DataFrame(all_results)
    for col in ['threshold','profit_factor','max_drawdown_pct','total_trades']:
        if col not in df_front.columns:
            raise ValueError(f'Missing column {col} in all_results')
    df_front = df_front.dropna(subset=['profit_factor','max_drawdown_pct','total_trades'])
    if df_front.empty:
        raise ValueError('Empty after filtering all_results')
    if 'win_rate' in df_front.columns:
        df_front['pf_wr_ratio'] = (df_front['profit_factor'] * df_front['win_rate'] / df_front['max_drawdown_pct'].replace(0,np.nan))
    df_front['pf_over_dd'] = df_front['profit_factor'] / df_front['max_drawdown_pct'].replace(0,np.nan)
    plt.figure(figsize=(10,7))
    sizes = 40 + 180 * (df_front['total_trades'] / df_front['total_trades'].max())
    sc = plt.scatter(df_front['max_drawdown_pct'], df_front['profit_factor'], s=sizes, c=df_front['total_trades'], cmap='viridis', alpha=0.78, edgecolor='k')
    opt_thr = (
        meta.get('optimal_threshold') or
        (meta.get('threshold_optimization') or {}).get('optimal_threshold') or
        (meta.get('labeling') or {}).get('optimal_threshold') or
        None
    )
    if opt_thr is not None:
        sel = df_front[df_front['threshold'] == opt_thr]
        if not sel.empty:
            plt.scatter(sel['max_drawdown_pct'], sel['profit_factor'], s=300, c='red', marker='*', edgecolor='black', label=f'Optimal {opt_thr}')
    for _, r in df_front.iterrows():
        if r['threshold'] in (df_front['threshold'].min(), df_front['threshold'].max(), opt_thr):
            plt.text(r['max_drawdown_pct'], r['profit_factor'], f"{r['threshold']:.2f}", fontsize=8)
    plt.xlabel('Max Drawdown %')
    plt.ylabel('Profit Factor')
    plt.title(f'Efficient Frontier (PF vs DD)\n{os.path.basename(meta_path)}')
    cb = plt.colorbar(sc)
    cb.set_label('Total Trades')
    plt.grid(alpha=0.3)
    if opt_thr is not None:
        plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'Saved efficient frontier to {save_path}')
    if show_plot:
        plt.show()
    else:
        plt.close()
@dataclass
class Config:
    symbol: str = "BTC/USDT"            # формат для ccxt
    tf_train: str = "30m"               # ОПТИМАЛЬНЫЙ таймфрейм на основе результатов
    lookback_days: int = 365 * 5         # сколько дней истории качать
    limit_per_call: int = 1000           # лимит ccxt за вызов

    # Target: triple-barrier style (оптимизировано для 30m)
    horizon_bars: int = 12               # увеличено с 6 до 12 (6 часов вместо 3) для реалистичности
    tp_pct: float = 0.008                # +0.8% в пределах горизонта => класс LONG
    sl_pct: float = 0.006                # -0.6% в пределах горизонта => класс SHORT
    tp_mult: float = 1.0                 # равное соотношение показало лучший результат
    sl_mult: float = 1.0                 # равное соотношение TP:SL

    # Фичи: окна для индикаторов
    rsi_len: int = 14
    ema_fast: int = 21
    ema_slow: int = 55
    atr_len: int = 14

    # Мульти-таймфрейм (агрегация из train TF в H1, можно выключить = None)
    higher_tf: str = "1h"
    higher_rsi_len: int = 14
    higher_ema: int = 50

    # Обучение
    test_size_frac: float = 0.2          # доля holdout-теста по хвосту
    n_splits_cv: int = 4                 # TimeSeries CV сплитов
    random_state: int = 42

    # Модель XGBClassifier
    xgb_params: Dict = None

    out_dir: str = "models_v2"
    
    # Версионирование моделей
    use_versioning: bool = True          # Добавлять timestamp к именам файлов
    version_format: str = "%Y%m%d_%H%M%S"  # Формат timestamp

    # --- Geometry (asymmetric grid) advanced options ---
    # Более консервативные диапазоны для реальной торговли
    geometry_tp_range: Tuple[float, float, float] = (1.2, 3.0, 0.25)  # более реалистичные TP
    geometry_sl_range: Tuple[float, float, float] = (0.6, 1.5, 0.2)   # более консервативные SL
    geometry_auto_expand: bool = True
    geometry_saturation_threshold: float = 0.60  # если >= доля tp_mult у верхней границы – расширяем
    geometry_max_tp_mult: float = 7.0            # абсолютный потолок расширения
    geometry_expansion_step: float = 1.0         # шаг увеличения верхней границы
    # Адаптивный старт SL (если почти всё упирается в нижнюю границу)
    geometry_sl_adaptive: bool = True            # включить повторный прогон с повышенным SL-минимумом
    geometry_sl_saturation_threshold: float = 0.7 # доля выбранных sl_mult на нижней границе для адаптации
    geometry_sl_enforced_min: float = 0.5        # новый нижний порог SL при адаптации

    # --- Anti-saturation & dynamic TP ---
    geometry_anti_saturation_enable: bool = True         # включить штраф при чрезмерной доле верхнего TP
    geometry_anti_saturation_target: float = 0.55        # желаемая максимальная доля верхнего уровня
    geometry_anti_saturation_alpha: float = 0.6          # сила штрафа (множитель влияния)
    dynamic_tp_apply: bool = False                       # применять динамический TP на основе MFE перцентиля
    dynamic_tp_percentile: float = 0.65                  # целевой перцентиль MFE/ATR
    dynamic_tp_min_observations: int = 100               # минимально trades для оценки динамики (было 200)

    # --- Debug / research ---
    debug_save_all_thresholds: bool = True       # по умолчанию включаем сохранение trades по всем threshold
    debug_trades_prefix: Optional[str] = None    # можно переопределить директорию/префикс для debug trades

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Train cryptocurrency trading model')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', 
                       help='Trading symbol (e.g., BTC/USDT, ETH/USDT)')
    parser.add_argument('--timeframe', type=str, default='30m',
                       help='Timeframe (e.g., 15m, 30m, 1h)')
    parser.add_argument('--days', type=int, default=1825,
                       help='History period in days')
    parser.add_argument('--tp_pct', type=float, default=0.008,
                       help='Take Profit percentage')
    parser.add_argument('--sl_pct', type=float, default=0.006,
                       help='Stop Loss percentage')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training data split ratio (0.8 = 80%% train, 20%% test)')
    parser.add_argument('--backtest-days', type=int,
                       help='Fixed number of days for backtest (alternative to --train-split)')
    parser.add_argument('--no-versioning', action='store_true',
                       help='Disable model versioning (overwrite existing files)')
    parser.add_argument('--silent', action='store_true',
                       help='Silent mode (suppress graph display)')
    # --- Research / overrides ---
    parser.add_argument('--geom-tp', type=str,
                       help='Override geometry TP range: start,end,step (e.g. 1.0,6.0,0.25)')
    parser.add_argument('--geom-sl', type=str,
                       help='Override geometry SL range: start,end,step (e.g. 0.4,2.0,0.2)')
    parser.add_argument('--horizon', type=int,
                       help='Override horizon_bars (e.g. 12). If large ( > 30 ) warns about runtime & overfitting risk.')
    parser.add_argument('--dynamic-tp-percentile', type=float,
                       help='After initial labeling/backtest calibration, adjust chosen tp_mult target to this MFE percentile (e.g. 0.65). Experimental.')
    parser.add_argument('--baseline-no-cap', action='store_true', help='Run a baseline threshold scan without TP cap for comparison (only meaningful when dynamic TP applied)')
    parser.add_argument('--bootstrap-pf', type=int, default=0, help='If >0: number of bootstrap samples for PF CI (applied after optimal threshold backtest)')
    # Sensitivity/auto-select controls
    # --dynamic-tp-sensitivity defined once below with extended help
    parser.add_argument('--sensitivity-min-trades', type=int, default=30, help='Minimum trades required for a percentile to be eligible in sensitivity selection')
    parser.add_argument('--sensitivity-thresholds', type=str, default='', help='Override sensitivity threshold subset, e.g. 0.73,0.75,0.78,0.79,0.80,0.81,0.82')
    parser.add_argument('--sensitivity-bootstrap-pf', type=int, default=0, help='If >0: bootstrap PF samples for best threshold per sensitivity percentile (fast CI)')
    parser.add_argument('--dynamic-tp-auto-select', type=str, default='', choices=['', 'recommend', 'apply'], help='Auto-select the most robust dynamic TP percentile (recommendation or apply). Apply mode currently records recommendation and sets needs_rerun flag.')
    # Plateau detection
    parser.add_argument('--plateau-pf-margin', type=float, default=0.05, help='Plateau margin: include thresholds with PF within this fraction of max PF (e.g., 0.05 = within 5%)')
    parser.add_argument('--plateau-min-trades', type=int, default=30, help='Minimum trades to include a threshold in plateau detection')
    parser.add_argument('--dynamic-tp-sensitivity', type=str, default='', help='Comma-separated list of alternative dynamic TP percentiles for sensitivity pass (e.g. 0.60,0.65,0.70)')
    parser.add_argument('--no-debug-trades', action='store_true',
                       help='Disable saving trades for all thresholds (default is enabled)')
    parser.add_argument('--frontier-from-meta', type=str,
                       help='Path to meta JSON to plot efficient frontier (PF vs DD) and exit')
    parser.add_argument('--frontier-show', action='store_true',
                       help='Force show frontier plot even if --silent is set')
    parser.add_argument('--save-frontier', nargs='?', const='auto',
                       help='Automatically save efficient frontier plot after training; optional custom filename or "auto"')
    parser.add_argument('--no-geom-auto-expand', action='store_true',
                       help='Disable automatic TP range expansion (geometry_auto_expand) for experimentation')
    parser.add_argument('--no-anti-saturation', action='store_true',
                       help='Disable anti-saturation penalty for TP selection')
    parser.add_argument('--apply-dynamic-tp', action='store_true',
                       help='Apply dynamic TP percentile adjustment (overrides suggest-only mode)')
    
    args = parser.parse_args()
    # --- Centralized training defaults (optional) ---
    # If Info/TrainingDefaults.json exists, load defaults and apply them when corresponding CLI flags are not provided.
    try:
        defaults_path = os.path.join('Info', 'TrainingDefaults.json')
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r', encoding='utf-8') as f:
                _td = json.load(f)
            # Compose effective defaults from root and symbol-specific overrides
            eff = dict(_td.get('defaults') or {})
            sym_key = (args.symbol or '').replace('/', '')
            sym_over = (_td.get('symbols') or {}).get(sym_key, {})
            eff.update(sym_over)

            # Helper: apply if flag not present on CLI
            def apply_if_missing(flag_name: str, attr: str, value):
                try:
                    if flag_name not in sys.argv and value is not None:
                        setattr(args, attr, value)
                except Exception:
                    pass

            # Geometry ranges
            apply_if_missing('--geom-tp', 'geom_tp', eff.get('geom_tp'))
            apply_if_missing('--geom-sl', 'geom_sl', eff.get('geom_sl'))
            # Horizon bars override
            apply_if_missing('--horizon', 'horizon', eff.get('horizon_bars'))
            # Auto expand toggle (default wants to disable expansion)
            if eff.get('no_geom_auto_expand', False) and ('--no-geom-auto-expand' not in sys.argv):
                args.no_geom_auto_expand = True
            # Anti-saturation toggle (keep enabled by default unless explicitly disabled)
            if eff.get('no_anti_saturation', False) and ('--no-anti-saturation' not in sys.argv):
                args.no_anti_saturation = True
            # Dynamic TP controls
            if eff.get('apply_dynamic_tp', False) and ('--apply-dynamic-tp' not in sys.argv):
                args.apply_dynamic_tp = True
            apply_if_missing('--dynamic-tp-percentile', 'dynamic_tp_percentile', eff.get('dynamic_tp_percentile'))
            apply_if_missing('--dynamic-tp-sensitivity', 'dynamic_tp_sensitivity', eff.get('dynamic_tp_sensitivity'))
            apply_if_missing('--sensitivity-min-trades', 'sensitivity_min_trades', eff.get('sensitivity_min_trades'))
            apply_if_missing('--sensitivity-bootstrap-pf', 'sensitivity_bootstrap_pf', eff.get('sensitivity_bootstrap_pf'))
            # Baseline (no-cap) comparison
            if eff.get('baseline_no_cap', False) and ('--baseline-no-cap' not in sys.argv):
                args.baseline_no_cap = True
            # Plateau detection
            apply_if_missing('--plateau-pf-margin', 'plateau_pf_margin', eff.get('plateau_pf_margin'))
            apply_if_missing('--plateau-min-trades', 'plateau_min_trades', eff.get('plateau_min_trades'))
            print(f"⚙️ Training defaults loaded from {defaults_path} (symbol override: {'yes' if sym_over else 'no'})")
    except Exception as _ex:
        print(f"⚠️ TrainingDefaults load error: {str(_ex)[:160]}")
    
    # Проверяем, что не переданы оба параметра одновременно
    # Определяем, был ли --train-split явно передан (отличается от дефолтного значения)
    train_split_provided = '--train-split' in sys.argv
    backtest_days_provided = args.backtest_days is not None
    
    if train_split_provided and backtest_days_provided:
        print("❌ ОШИБКА: Нельзя указывать одновременно --train-split и --backtest-days")
        print("   Используйте один из параметров:")
        print("   --train-split 0.85 (процентное разделение)")
        print("   --backtest-days 30 (фиксированное количество дней)")
        exit(1)
    
    return args

cfg = Config()

def create_versioned_basename(symbol: str, timeframe: str, lookback_days: int, 
                              backtest_info: str = "", use_versioning: bool = True, 
                              version_format: str = "%Y%m%d_%H%M%S") -> tuple:
    """
    Создает базовое имя файла с опциональным версионированием
    
    Args:
        symbol: Торговый символ
        timeframe: Таймфрейм
        lookback_days: Общее количество дней данных
        backtest_info: Информация о бэктесте (например, "bt7d" или "bt20pct")
        use_versioning: Включить ли версионирование
        version_format: Формат временной метки
    
    Returns:
        tuple: (base_name, timestamp)
    """
    base = f"{symbol.replace('/','')}_{timeframe}_{lookback_days}d"
    if backtest_info:
        base += f"_{backtest_info}"
    timestamp = datetime.now().strftime(version_format) if use_versioning else ""
    
    if use_versioning and timestamp:
        versioned_base = f"{base}_{timestamp}"
    else:
        versioned_base = base
    
    return versioned_base, timestamp

if cfg.xgb_params is None:
    cfg.xgb_params = dict(
        n_estimators=800,                # ОПТИМАЛЬНО (проверено 3 экспериментами)
        max_depth=4,                     # ОПТИМАЛЬНО (баланс: сложность vs переобучение)
        learning_rate=0.05,              # умеренная скорость обучения
        subsample=0.7,                   # регуляризация против переобучения
        colsample_bytree=0.8,            # выбор случайных фичей
        min_child_weight=3,              # минимальный вес в листе (важно!)
        gamma=0.1,                       # порог для разделения узла (важно!)
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,        # остановка при стагнации
        n_jobs=-1,
        random_state=cfg.random_state,
    )

os.makedirs(cfg.out_dir, exist_ok=True)

# =====================
# Адаптивные параметры
# =====================

def detect_market_regime(df: pd.DataFrame) -> str:
    """Определение режима рынка"""
    
    # Убеждаемся что у нас есть ATR
    if 'atr' not in df.columns:
        # Рассчитываем ATR если его нет
        high_low_diff = df['high'] - df['low']
        high_close_diff = abs(df['high'] - df['close'].shift())
        low_close_diff = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low_diff, high_close_diff, low_close_diff], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
    
    # Trend strength (упрощенный ADX)
    high_low_diff = df['high'] - df['low']
    high_close_diff = abs(df['high'] - df['close'].shift())
    low_close_diff = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low_diff, high_close_diff, low_close_diff], axis=1).max(axis=1)
    adx = true_range.rolling(14).mean()
    
    # Direction
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    
    # Volatility
    vol_percentile = df['atr'].rolling(100).rank(pct=True)
    
    # Получаем последние значения (защита от NaN)
    try:
        current_adx = adx.dropna().iloc[-1] if len(adx.dropna()) > 0 else 15
        current_vol = vol_percentile.dropna().iloc[-1] if len(vol_percentile.dropna()) > 0 else 0.5
        is_uptrend = sma_20.dropna().iloc[-1] > sma_50.dropna().iloc[-1] if len(sma_20.dropna()) > 0 and len(sma_50.dropna()) > 0 else True
    except:
        # Fallback значения
        current_adx = 15
        current_vol = 0.5
        is_uptrend = True
    
    # Классификация режима
    if current_adx > 25:  # Trending market
        if is_uptrend:
            return 'trending_bull'
        else:
            return 'trending_bear'
    elif current_vol > 0.8:
        return 'high_volatility'
    else:
        return 'ranging'

def get_adaptive_parameters(regime: str, symbol: str) -> dict:
    """Получение адаптивных параметров для режима рынка"""
    
    # Единые параметры для всех монет
    base_tp, base_sl = 1.3, 1.3  # Можно скорректировать по вашему опыту
    base_horizon = 8
    base_threshold = 0.65
    
    # Модификация по режиму рынка (сбалансированные параметры)
    regime_params = {
        'trending_bull': {
            'tp_mult': base_tp * 1.0,      # TP: 1.2x 
            'sl_mult': base_sl * 1.0,      # SL: 1.2x (равные)
            'horizon_bars': base_horizon + 2,   # 10 баров
            'confidence_threshold': base_threshold + 0.05  # 0.65
        },
        'trending_bear': {
            'tp_mult': base_tp * 1.1,      # TP: 1.32x
            'sl_mult': base_sl * 1.1,      # SL: 1.32x (равные)
            'horizon_bars': base_horizon + 1,   # 9 баров
            'confidence_threshold': base_threshold + 0.02  # 0.62
        },
        'ranging': {
            'tp_mult': base_tp * 1.0,      # TP: 1.2x
            'sl_mult': base_sl * 1.0,      # SL: 1.2x (равные для баланса)
            'horizon_bars': base_horizon,       # 8 баров
            'confidence_threshold': base_threshold + 0.05  # 0.65
        },
        'high_volatility': {
            'tp_mult': base_tp * 1.3,      # TP: 1.56x 
            'sl_mult': base_sl * 1.3,      # SL: 1.56x (равные)
            'horizon_bars': base_horizon + 3,   # 11 баров
            'confidence_threshold': base_threshold + 0.08   # 0.68
        }
    }
    
    return regime_params.get(regime, regime_params['ranging'])

def get_time_based_threshold_adjustment(hour: int) -> float:
    """Корректировка порога по времени суток"""
    
    # Неликвидные часы - более строгие пороги
    if hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Азиатская сессия/ночь
        return 0.05  # +5% к порогу
    elif hour in [6, 7]:  # Переход к Лондону
        return 0.02  # +2% к порогу
    else:  # Ликвидные часы (Лондон + Нью-Йорк)
        return -0.02  # -2% к порогу (менее строго)

# =====================
# Утилиты
# =====================

def make_advanced_exit_labels_atr(df: pd.DataFrame, tp_mult=1.0, sl_mult=1.0, atr_window=14, horizon_bars=8, 
                                  trailing_stop=True, trailing_mult=0.5, min_profit_for_trail=0.3):
    """
    Продвинутая стратегия выхода с trailing stop и временными ограничениями:
    - LONG, если позиция прибыльна
    - SHORT, если позиция убыточна  
    - HOLD, если неопределенно
    
    Параметры:
    - trailing_stop: использовать ли trailing stop
    - trailing_mult: множитель ATR для trailing stop (например, 0.5 = 0.5*ATR)
    - min_profit_for_trail: минимальная прибыль в долях ATR для активации trailing stop
    """

    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"], 
        close=df["close"],
        window=atr_window
    ).average_true_range()

    labels = []
    closes = df["close"].values
    opens = df["open"].values  # Добавляем массив open для реалистичного входа
    highs = df["high"].values
    lows = df["low"].values
    atr_vals = atr.values

    for i in range(len(closes)):
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вход на следующей свече (open[i+1])
        # Модель видит индикаторы на закрытии свечи i, вход происходит на открытии i+1
        if i + 1 >= len(opens):
            labels.append("HOLD")
            continue
            
        entry = opens[i + 1]  # Реалистичный вход: open следующей свечи после сигнала
        atr_val = atr_vals[i]
        if np.isnan(atr_val) or np.isnan(entry):
            labels.append("HOLD")
            continue

        # Начальные уровни
        initial_tp = entry + tp_mult * atr_val
        initial_sl = entry - sl_mult * atr_val
        
        # Переменные для trailing stop
        current_sl = initial_sl
        max_profitable_price = entry
        trailing_active = False
        
        outcome = "HOLD"
        exit_price = None
        
        # ИСПРАВЛЕНИЕ: Начинаем проверку с свечи i+1 (вход на open[i+1])
        for j in range(1, horizon_bars + 1):
            check_idx = i + j
            if check_idx >= len(closes):
                break

            current_high = highs[check_idx]
            current_low = lows[check_idx]
            current_close = closes[check_idx]

            # Проверка достижения начального TP
            if current_high >= initial_tp:
                outcome = "LONG"
                exit_price = initial_tp
                break
                
            # Обновление trailing stop для лонг позиций
            if trailing_stop and current_high > entry:
                profit_in_atr = (current_high - entry) / atr_val
                
                # Активация trailing stop при достижении минимальной прибыли
                if profit_in_atr >= min_profit_for_trail:
                    trailing_active = True
                    max_profitable_price = max(max_profitable_price, current_high)
                    
                    # Обновление trailing stop level
                    new_trailing_sl = max_profitable_price - trailing_mult * atr_val
                    current_sl = max(current_sl, new_trailing_sl)

            # Проверка stop loss (обычного или trailing)
            if current_low <= current_sl:
                if trailing_active and current_sl > initial_sl:
                    # Trailing stop сработал - частичная прибыль
                    profit_in_atr = (current_sl - entry) / atr_val
                    outcome = "LONG" if profit_in_atr > 0.1 else "SHORT"  # Минимальная прибыль для лонга
                else:
                    # Обычный stop loss
                    outcome = "SHORT"
                exit_price = current_sl
                break
                
            # Временной выход - анализируем текущую позицию
            if j == horizon_bars:
                profit_in_atr = (current_close - entry) / atr_val
                
                if profit_in_atr > 0.2:  # Хорошая прибыль
                    outcome = "LONG"
                elif profit_in_atr < -0.15:  # Значительный убыток
                    outcome = "SHORT"
                else:
                    outcome = "HOLD"  # Неопределенная ситуация
                    
                exit_price = current_close

        labels.append(outcome)

    return labels


def make_triple_barrier_labels_atr(df: pd.DataFrame, tp_mult=1.0, sl_mult=1.0, atr_window=14, horizon_bars=8):
    """
    ИСПРАВЛЕННАЯ симметричная triple-barrier разметка на основе ATR:
    - LONG: если цена достигнет TP вверх раньше, чем любой из SL
    - SHORT: если цена достигнет TP вниз раньше, чем любой из SL
    - HOLD: если не достигнуты ни один из TP в пределах горизонта
    
    Это ПРАВИЛЬНАЯ логика для предсказания направления движения цены!
    """

    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=atr_window
    ).average_true_range()

    labels = []
    closes = df["close"].values
    opens = df["open"].values  # Добавляем массив open для реалистичного входа
    highs = df["high"].values
    lows = df["low"].values
    atr_vals = atr.values

    for i in range(len(closes)):
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вход на следующей свече (open[i+1])
        if i + 1 >= len(opens):
            labels.append("HOLD")
            continue
            
        entry = opens[i + 1]  # Реалистичный вход: open следующей свечи после сигнала
        atr_val = atr_vals[i]
        if np.isnan(atr_val) or np.isnan(entry):
            labels.append("HOLD")
            continue

        # Уровни для LONG позиции
        tp_long = entry + tp_mult * atr_val  # TP вверх
        sl_long = entry - sl_mult * atr_val  # SL вниз
        
        # Уровни для SHORT позиции  
        tp_short = entry - tp_mult * atr_val  # TP вниз
        sl_short = entry + sl_mult * atr_val  # SL вверх

        long_tp_hit = False
        short_tp_hit = False
        long_tp_bar = None
        short_tp_bar = None
        
        outcome = "HOLD"
        
        for j in range(1, horizon_bars + 1):
            check_idx = i + j
            if check_idx >= len(closes):
                break

            high = highs[check_idx]
            low = lows[check_idx]

            # Проверяем достижение TP для LONG (цена вверх)
            if not long_tp_hit and high >= tp_long:
                long_tp_hit = True
                long_tp_bar = j
                
            # Проверяем достижение TP для SHORT (цена вниз)
            if not short_tp_hit and low <= tp_short:
                short_tp_hit = True
                short_tp_bar = j

        # Определяем метку на основе того, что произошло раньше
        if long_tp_hit and short_tp_hit:
            # Оба направления достигли TP - выбираем более быстрое
            if long_tp_bar < short_tp_bar:
                outcome = "LONG"
            elif short_tp_bar < long_tp_bar:
                outcome = "SHORT"
            else:
                outcome = "HOLD"  # Одновременно - редкий случай
        elif long_tp_hit:
            outcome = "LONG"
        elif short_tp_hit:
            outcome = "SHORT"
        # else: outcome остается "HOLD"

        labels.append(outcome)

    return labels

# =============================
# Variant B: Asymmetric grid relabeling
# =============================
def make_asymmetric_grid_labels(
    df: pd.DataFrame,
    atr_window: int = 14,
    horizon_bars: int = 8,
    tp_mult_range: Tuple[float, float, float] = (1.0, 5.0, 0.25),
    sl_mult_range: Tuple[float, float, float] = (0.4, 2.0, 0.2),
    min_net_rr: Optional[float] = None,
    round_trip_cost_pct: float = 0.002,
    selection: str = 'first_hit_best_rr',
    prefer: str = 'rr_then_speed',
    auto_expand: bool = False,
    saturation_threshold: float = 0.6,
    expansion_step: float = 1.0,
    max_tp_cap: float = 7.0,
) -> Tuple[List[str], Dict[str, any]]:
    """Generate labels by scanning a grid of (tp_mult, sl_mult) asymmetries per bar.

    For each entry bar i:
      - For each (tp_mult, sl_mult) compute barrier prices using ATR[i].
      - Simulate up to horizon_bars forward bars, recording which barrier is hit first (TP→LONG, SL→SHORT).
      - If neither barrier hit within horizon: HOLD.
      - Optionally enforce feasibility wrt min_net_rr using net_rr = (tp_pct - cost)/(sl_pct + cost).

    Selection policy (if multiple geometries produce different outcomes):
      selection='first_hit_best_rr' & prefer='rr_then_speed':
        1) Filter geometries that yield acceptable net_rr (>= min_net_rr) if min_net_rr set; else all.
        2) Among those producing at least one directional outcome (LONG/SHORT), choose the geometry with highest net_rr.
        3) If tie on net_rr, choose one with earliest barrier hit (fewest bars).
        4) If still tie, choose smallest sl_mult (tighter risk).
        5) If no geometry yields directional (all HOLD) -> HOLD.

    Returns:
      labels: list of class strings per row
      stats: diagnostics dict with distributions & infeasibility counts
    """
    import math
    atr_series = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=atr_window
    ).average_true_range()

    closes = df["close"].values
    opens = df["open"].values  # Добавляем массив open для реалистичного входа
    highs = df["high"].values
    lows = df["low"].values
    atr_vals = atr_series.values

    def _run_single_pass(tp_mult_range_local: Tuple[float, float, float]):
        tp_start, tp_end, tp_step = tp_mult_range_local
        sl_start, sl_end, sl_step = sl_mult_range
        tp_grid_local = np.arange(tp_start, tp_end + 1e-9, tp_step)
        sl_grid_local = np.arange(sl_start, sl_end + 1e-9, sl_step)

        labels_local: List[str] = []
        chosen_tp_mult_local: List[float] = []
        chosen_sl_mult_local: List[float] = []
        chosen_net_rr_local: List[float] = []
        infeasible_count_local = 0
        all_infeasible_rows_local = 0

        import math
        for i in range(len(closes)):
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вход на следующей свече (open[i+1])
            if i + 1 >= len(opens):
                labels_local.append("HOLD")
                chosen_tp_mult_local.append(float('nan'))
                chosen_sl_mult_local.append(float('nan'))
                chosen_net_rr_local.append(float('nan'))
                continue
                
            entry = opens[i + 1]  # Реалистичный вход: open следующей свечи после сигнала
            atr_val = atr_vals[i]
            if math.isnan(atr_val) or atr_val <= 0 or math.isnan(entry):
                labels_local.append("HOLD")
                chosen_tp_mult_local.append(float('nan'))
                chosen_sl_mult_local.append(float('nan'))
                chosen_net_rr_local.append(float('nan'))
                continue
            best = None  # (net_rr, bars_to_hit, outcome, tp_mult, sl_mult)
            any_feasible = False
            # Precompute penalty for saturation (if enabled)
            anti_sat_enable = getattr(cfg, 'geometry_anti_saturation_enable', False)
            anti_sat_target = getattr(cfg, 'geometry_anti_saturation_target', 0.55)
            anti_sat_alpha = getattr(cfg, 'geometry_anti_saturation_alpha', 0.6)
            # NOTE: Heuristic pre-penalty removed. We rely solely on precise second-pass adjustment after measuring saturation.
            max_tp_candidate = tp_grid_local[-1]
            for tp_m in tp_grid_local:
                for sl_m in sl_grid_local:
                    tp_level = entry + tp_m * atr_val
                    sl_level = entry - sl_m * atr_val
                    target_pct = (tp_level - entry) / entry
                    stop_pct = (entry - sl_level) / entry
                    if target_pct <= 0 or stop_pct <= 0:
                        continue
                    net_target = target_pct - round_trip_cost_pct
                    net_stop = stop_pct + round_trip_cost_pct
                    if net_target <= 0 or net_stop <= 0:
                        continue
                    net_rr = net_target / net_stop
                    # (anti-saturation pre-penalty disabled)
                    feasible = (min_net_rr is None) or (net_rr >= min_net_rr)
                    if feasible:
                        any_feasible = True
                    # Simulate horizon - ИСПРАВЛЕНО: симметричная разметка для LONG и SHORT
                    # Для LONG: TP вверх, SL вниз
                    # Для SHORT: TP вниз (entry - tp_m * atr), SL вверх (entry + sl_m * atr)
                    tp_level_long = tp_level  # entry + tp_m * atr
                    sl_level_long = sl_level  # entry - sl_m * atr
                    tp_level_short = entry - tp_m * atr_val  # TP для SHORT - цена вниз
                    sl_level_short = entry + sl_m * atr_val  # SL для SHORT - цена вверх
                    
                    outcome = "HOLD"
                    bars_to_hit = horizon_bars + 1  # sentinel bigger than horizon
                    long_hit = False
                    short_hit = False
                    long_bars = horizon_bars + 1
                    short_bars = horizon_bars + 1
                    
                    for j in range(1, horizon_bars + 1):
                        check_idx = i + j
                        if check_idx >= len(closes):
                            break
                        high = highs[check_idx]
                        low = lows[check_idx]
                        
                        # Проверяем достижение TP для LONG (цена вверх)
                        if not long_hit and high >= tp_level_long:
                            long_hit = True
                            long_bars = j
                            
                        # Проверяем достижение TP для SHORT (цена вниз)
                        if not short_hit and low <= tp_level_short:
                            short_hit = True
                            short_bars = j
                    
                    # Определяем метку на основе того, что произошло раньше
                    if long_hit and short_hit:
                        if long_bars < short_bars:
                            outcome = "LONG"
                            bars_to_hit = long_bars
                        elif short_bars < long_bars:
                            outcome = "SHORT"
                            bars_to_hit = short_bars
                        else:
                            outcome = "HOLD"  # Одновременно - редкий случай
                            bars_to_hit = horizon_bars + 1
                    elif long_hit:
                        outcome = "LONG"
                        bars_to_hit = long_bars
                    elif short_hit:
                        outcome = "SHORT"
                        bars_to_hit = short_bars
                    if min_net_rr is not None and not feasible:
                        continue  # skip infeasible geometry entirely
                    # Consider only directional outcomes for selection; HOLD kept if nothing else
                    if outcome == "HOLD":
                        continue
                    if best is None:
                        best = (net_rr, bars_to_hit, outcome, tp_m, sl_m)
                    else:
                        # Compare by net_rr, then speed, then tighter SL
                        if net_rr > best[0] + 1e-9:
                            best = (net_rr, bars_to_hit, outcome, tp_m, sl_m)
                        elif abs(net_rr - best[0]) <= 1e-9:
                            if bars_to_hit < best[1]:
                                best = (net_rr, bars_to_hit, outcome, tp_m, sl_m)
                            elif bars_to_hit == best[1]:
                                # Новый приоритет: более низкий tp_mult (для снижения скрытой сатурации верхнего уровня)
                                if tp_m < best[3]:
                                    best = (net_rr, bars_to_hit, outcome, tp_m, sl_m)
                                elif tp_m == best[3] and sl_m < best[4]:
                                    best = (net_rr, bars_to_hit, outcome, tp_m, sl_m)
            if best is None:
                labels_local.append("HOLD")
                chosen_tp_mult_local.append(float('nan'))
                chosen_sl_mult_local.append(float('nan'))
                chosen_net_rr_local.append(float('nan'))
                if not any_feasible:
                    infeasible_count_local += 1
                all_infeasible_rows_local += (0 if any_feasible else 1)
            else:
                labels_local.append(best[2])
                chosen_tp_mult_local.append(best[3])
                chosen_sl_mult_local.append(best[4])
                chosen_net_rr_local.append(best[0])
        return (labels_local, chosen_tp_mult_local, chosen_sl_mult_local, chosen_net_rr_local,
                infeasible_count_local, all_infeasible_rows_local, tp_mult_range_local)

    # Первичный прогон
    labels_pass, tp_list, sl_list, rr_list, infeas_cnt, no_dir_cnt, used_tp_range = _run_single_pass(tp_mult_range)

    saturation_flag = False
    expansion_rounds = []
    if auto_expand:
        # считаем долю directional выборок на верхней границе
        arr_tp = np.array(tp_list)
        arr_labels = np.array(labels_pass)
        if np.isfinite(arr_tp).any():
            tp_max = used_tp_range[1]
            directional_mask = (arr_labels == 'LONG') | (arr_labels == 'SHORT')
            if directional_mask.any():
                pct_at_max = float(np.sum((arr_tp == tp_max) & directional_mask) / np.sum(directional_mask))
            else:
                pct_at_max = 0.0
            if pct_at_max >= saturation_threshold and used_tp_range[1] < max_tp_cap:
                saturation_flag = True
                current_end = used_tp_range[1]
                # Расширяем пока есть насыщение и не превышен max_tp_cap
                while pct_at_max >= saturation_threshold and current_end < max_tp_cap:
                    new_end = min(current_end + expansion_step, max_tp_cap)
                    new_range = (used_tp_range[0], new_end, used_tp_range[2])
                    labels_pass, tp_list, sl_list, rr_list, infeas_cnt, no_dir_cnt, used_tp_range = _run_single_pass(new_range)
                    arr_tp = np.array(tp_list)
                    arr_labels = np.array(labels_pass)
                    directional_mask = (arr_labels == 'LONG') | (arr_labels == 'SHORT')
                    if directional_mask.any():
                        pct_at_max = float(np.sum((arr_tp == new_end) & directional_mask) / np.sum(directional_mask))
                    else:
                        pct_at_max = 0.0
                    expansion_rounds.append({'new_end': new_end, 'pct_at_max': pct_at_max})
                    current_end = new_end

    labels = labels_pass
    chosen_tp_mult = tp_list
    chosen_sl_mult = sl_list
    chosen_net_rr = rr_list
    tp_mult_range_final = used_tp_range
    # Рассчёт saturation на финальном диапазоне
    arr_tp_final = np.array(chosen_tp_mult)
    arr_labels_final = np.array(labels)
    directional_mask_final = (arr_labels_final == 'LONG') | (arr_labels_final == 'SHORT')
    if directional_mask_final.any():
        tp_max_final = tp_mult_range_final[1]
        pct_at_max_final = float(np.sum((arr_tp_final == tp_max_final) & directional_mask_final) / np.sum(directional_mask_final))
    else:
        pct_at_max_final = 0.0

    # ===== Second pass anti-saturation (refined) =====
    anti_sat_second_pass = None
    if getattr(cfg, 'geometry_anti_saturation_enable', False):
        target_pct = getattr(cfg, 'geometry_anti_saturation_target', 0.55)
        alpha_pen = getattr(cfg, 'geometry_anti_saturation_alpha', 0.6)
        if pct_at_max_final > target_pct and (tp_mult_range_final[1] - tp_mult_range_final[0]) > 0.0:
            # compute penalty factor based on actual exceed amount
            exceed = (pct_at_max_final - target_pct) / max(1e-9, (1 - target_pct))
            penalty_factor = 1 - alpha_pen * exceed
            penalty_factor = float(min(max(penalty_factor, 0.05), 0.99))
            # re-run selection with penalty applied precisely to max tp level
            tp_start2, tp_end2, tp_step2 = tp_mult_range_final
            sl_start2, sl_end2, sl_step2 = sl_mult_range
            tp_grid2 = np.arange(tp_start2, tp_end2 + 1e-9, tp_step2)
            sl_grid2 = np.arange(sl_start2, sl_end2 + 1e-9, sl_step2)
            new_tp_list = []
            new_sl_list = []
            new_rr_list = []
            new_labels = []
            import math
            for i in range(len(closes)):
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вход на следующей свече (open[i+1])
                if i + 1 >= len(opens):
                    new_labels.append(labels[i])
                    new_tp_list.append(chosen_tp_mult[i])
                    new_sl_list.append(chosen_sl_mult[i])
                    new_rr_list.append(chosen_net_rr[i])
                    continue
                    
                entry = opens[i + 1]  # Реалистичный вход: open следующей свечи после сигнала
                atr_val = atr_vals[i]
                if math.isnan(atr_val) or atr_val <= 0 or math.isnan(entry):
                    new_labels.append(labels[i])
                    new_tp_list.append(chosen_tp_mult[i])
                    new_sl_list.append(chosen_sl_mult[i])
                    new_rr_list.append(chosen_net_rr[i])
                    continue
                best2 = None
                for tp_m in tp_grid2:
                    for sl_m in sl_grid2:
                        tp_level = entry + tp_m * atr_val
                        sl_level = entry - sl_m * atr_val
                        target_pct = (tp_level - entry) / entry
                        stop_pct = (entry - sl_level) / entry
                        if target_pct <= 0 or stop_pct <= 0:
                            continue
                        net_target = target_pct - round_trip_cost_pct
                        net_stop = stop_pct + round_trip_cost_pct
                        if net_target <= 0 or net_stop <= 0:
                            continue
                        net_rr2 = net_target / net_stop
                        if tp_m == tp_grid2[-1]:
                            net_rr2 *= penalty_factor
                        # simulate horizon (fast re-use prior horizon_bars) - ИСПРАВЛЕНО: симметричная разметка
                        tp_level_long = tp_level  # entry + tp_m * atr
                        sl_level_long = sl_level  # entry - sl_m * atr
                        tp_level_short = entry - tp_m * atr_val  # TP для SHORT - цена вниз
                        sl_level_short = entry + sl_m * atr_val  # SL для SHORT - цена вверх
                        
                        outcome = "HOLD"
                        bars_to_hit = horizon_bars + 1
                        long_hit = False
                        short_hit = False
                        long_bars = horizon_bars + 1
                        short_bars = horizon_bars + 1
                        
                        for j in range(1, horizon_bars + 1):
                            check_idx = i + j
                            if check_idx >= len(closes):
                                break
                            high = highs[check_idx]
                            low = lows[check_idx]
                            
                            # Проверяем достижение TP для LONG (цена вверх)
                            if not long_hit and high >= tp_level_long:
                                long_hit = True
                                long_bars = j
                                
                            # Проверяем достижение TP для SHORT (цена вниз)
                            if not short_hit and low <= tp_level_short:
                                short_hit = True
                                short_bars = j
                        
                        # Определяем метку на основе того, что произошло раньше
                        if long_hit and short_hit:
                            if long_bars < short_bars:
                                outcome = "LONG"
                                bars_to_hit = long_bars
                            elif short_bars < long_bars:
                                outcome = "SHORT"
                                bars_to_hit = short_bars
                            else:
                                outcome = "HOLD"  # Одновременно
                                bars_to_hit = horizon_bars + 1
                        elif long_hit:
                            outcome = "LONG"
                            bars_to_hit = long_bars
                        elif short_hit:
                            outcome = "SHORT"
                            bars_to_hit = short_bars
                        if outcome == 'HOLD':
                            continue
                        if best2 is None:
                            best2 = (net_rr2, bars_to_hit, outcome, tp_m, sl_m)
                        else:
                            if net_rr2 > best2[0] + 1e-9:
                                best2 = (net_rr2, bars_to_hit, outcome, tp_m, sl_m)
                            elif abs(net_rr2 - best2[0]) <= 1e-9:
                                if bars_to_hit < best2[1]:
                                    best2 = (net_rr2, bars_to_hit, outcome, tp_m, sl_m)
                                elif bars_to_hit == best2[1] and sl_m < best2[4]:
                                    best2 = (net_rr2, bars_to_hit, outcome, tp_m, sl_m)
                if best2 is None:
                    # fallback – keep original
                    new_labels.append(labels[i])
                    new_tp_list.append(chosen_tp_mult[i])
                    new_sl_list.append(chosen_sl_mult[i])
                    new_rr_list.append(chosen_net_rr[i])
                else:
                    new_labels.append(best2[2])
                    new_tp_list.append(best2[3])
                    new_sl_list.append(best2[4])
                    new_rr_list.append(best2[0])
            # recompute pct_at_max after second pass
            arr_tp_second = np.array(new_tp_list)
            arr_labels_second = np.array(new_labels)
            directional_mask_second = (arr_labels_second == 'LONG') | (arr_labels_second == 'SHORT')
            if directional_mask_second.any():
                tp_max_final2 = tp_mult_range_final[1]
                pct_after = float(np.sum((arr_tp_second == tp_max_final2) & directional_mask_second) / np.sum(directional_mask_second))
            else:
                pct_after = 0.0
            anti_sat_second_pass = {
                'applied': True,
                'pct_before': pct_at_max_final,
                'pct_after': pct_after,
                'penalty_factor': penalty_factor
            }
            # adopt new results
            labels = new_labels
            chosen_tp_mult = new_tp_list
            chosen_sl_mult = new_sl_list
            chosen_net_rr = new_rr_list
            pct_at_max_final = pct_after

    stats = {
        'grid_tp_range': tp_mult_range_final,
        'grid_sl_range': sl_mult_range,
        'min_net_rr': min_net_rr,
        'round_trip_cost_pct': round_trip_cost_pct,
        'chosen_tp_mult_dist': {
            'mean': float(np.nanmean(chosen_tp_mult)) if len(chosen_tp_mult) else None,
            'median': float(np.nanmedian(chosen_tp_mult)) if len(chosen_tp_mult) else None,
        },
        'chosen_sl_mult_dist': {
            'mean': float(np.nanmean(chosen_sl_mult)) if len(chosen_sl_mult) else None,
            'median': float(np.nanmedian(chosen_sl_mult)) if len(chosen_sl_mult) else None,
        },
        'chosen_net_rr_dist': {
            'mean': float(np.nanmean(chosen_net_rr)) if len(chosen_net_rr) else None,
            'median': float(np.nanmedian(chosen_net_rr)) if len(chosen_net_rr) else None,
        },
        'infeasible_rows': infeas_cnt,
        'total_rows_no_direction': no_dir_cnt,
        'saturation_auto_expanded': auto_expand,
        'saturation_triggered': saturation_flag,
        'saturation_threshold': saturation_threshold,
        'pct_at_tp_max_final': pct_at_max_final,
        'expansion_rounds': expansion_rounds,
        'anti_saturation': {
            'enabled': getattr(cfg, 'geometry_anti_saturation_enable', False),
            'alpha': getattr(cfg, 'geometry_anti_saturation_alpha', None),
            'target': getattr(cfg, 'geometry_anti_saturation_target', None)
        },
        'anti_saturation_second_pass': anti_sat_second_pass
    }
    df['grid_tp_mult_chosen'] = chosen_tp_mult
    df['grid_sl_mult_chosen'] = chosen_sl_mult
    df['grid_net_rr_chosen'] = chosen_net_rr
    return labels, stats

def check_data_availability(exchange: object, symbol: str, timeframe: str, required_days: int, exchange_name: str) -> Tuple[bool, int, Optional[str]]:
    """
    Проверяет доступность исторических данных для символа на конкретной бирже
    
    Args:
        exchange: Экземпляр биржи (ccxt)
        symbol: Символ в формате ccxt (BTC/USDT)
        timeframe: Таймфрейм (30m, 1h, etc.)
        required_days: Требуемое количество дней данных
        exchange_name: Название биржи для логирования
    
    Returns:
        Tuple[bool, int, Optional[str]]: (достаточно_данных, доступно_дней, дата_листинга)
    """
    try:
        # Специальная проверка для Binance через API
        if exchange_name.lower() == 'binance':
            try:
                # Конвертируем символ обратно в формат Binance
                binance_symbol = symbol.replace('/', '')
                
                response = requests.get(
                    'https://api.binance.com/api/v3/exchangeInfo',
                    params={'symbol': binance_symbol},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'symbols' in data and len(data['symbols']) > 0:
                        # Получаем время листинга из onboardDate
                        symbol_info = data['symbols'][0]
                        if 'onboardDate' in symbol_info and symbol_info['onboardDate']:
                            listing_timestamp = int(symbol_info['onboardDate'])
                            listing_date = datetime.fromtimestamp(listing_timestamp / 1000)
                            
                            # Вычисляем доступные дни
                            days_available = (datetime.now() - listing_date).days
                            listing_date_str = listing_date.strftime('%Y-%m-%d')
                            
                            sufficient = days_available >= required_days
                            
                            if not sufficient:
                                print(f"⚠️ {exchange_name}: {symbol} - недостаточно данных!")
                                print(f"   📅 Доступно: {days_available} дней (требуется: {required_days})")
                                print(f"   📍 Данные с: {listing_date_str}")
                                print(f"   💡 Рекомендуемый максимум: {max(30, days_available - 30)} дней")
                            
                            return sufficient, days_available, listing_date_str
                        
            except Exception as e:
                print(f"⚠️ Ошибка проверки данных Binance API: {e}")
                # Продолжаем с общей проверкой ccxt
        
        # Общая проверка через ccxt для всех бирж
        # Пробуем получить максимум данных для проверки доступности
        try:
            # Получаем данные начиная с максимально раннего времени
            # Используем время 10 лет назад как стартовую точку для поиска всей доступной истории
            ten_years_ago = int((datetime.now() - timedelta(days=3650)).timestamp() * 1000)
            
            test_data = exchange.fetch_ohlcv(symbol, timeframe, since=ten_years_ago, limit=1000)
            if not test_data or len(test_data) == 0:
                print(f"❌ {exchange_name}: Нет данных для {symbol}")
                return False, 0, None
            
            # Получаем самые старые доступные данные
            oldest_timestamp = test_data[0][0]
            oldest_date = datetime.fromtimestamp(oldest_timestamp / 1000)
            
            # Вычисляем доступные дни
            days_available = (datetime.now() - oldest_date).days
            oldest_date_str = oldest_date.strftime('%Y-%m-%d')
            
            sufficient = days_available >= required_days
            
            if not sufficient:
                print(f"⚠️ {exchange_name}: {symbol} - недостаточно данных!")
                print(f"   📅 Доступно: {days_available} дней (требуется: {required_days})")
                print(f"   📍 Данные с: {oldest_date_str}")
                print(f"   💡 Рекомендуемый максимум: {max(30, days_available - 30)} дней")
            else:
                print(f"✅ {exchange_name}: {symbol} - данных достаточно ({days_available} дней)")
            
            return sufficient, days_available, oldest_date_str
            
        except Exception as e:
            print(f"❌ {exchange_name}: Ошибка получения данных для {symbol}: {e}")
            return False, 0, None
            
    except Exception as e:
        print(f"❌ {exchange_name}: Критическая ошибка проверки данных: {e}")
        return False, 0, None

def find_best_exchange(symbol: str, required_days: int = 365) -> tuple:
    """
    Находит лучшую биржу для символа с fallback механизмом и проверкой доступности данных
    Принимает символ в формате BTC/USDT или BTCUSDT
    Возвращает (exchange_instance, exchange_name, ccxt_symbol)
    """
    # Конвертируем в правильный формат для ccxt
    if '/' in symbol:
        # Уже в формате BTC/USDT - оставляем как есть
        ccxt_symbol = symbol
    elif symbol.endswith('USDT'):
        base = symbol[:-4]
        ccxt_symbol = f"{base}/USDT"
    elif symbol.endswith('USDC'):
        base = symbol[:-4]
        ccxt_symbol = f"{base}/USDC"
    elif symbol.endswith('BTC'):
        base = symbol[:-3]
        ccxt_symbol = f"{base}/BTC"
    else:
        ccxt_symbol = symbol  # Оставляем как есть для нестандартных форматов
    
    print(f"Символ {symbol} -> {ccxt_symbol} для ccxt")
    print(f"🔍 Поиск биржи с достаточными данными (≥{required_days} дней)...")
    
    # Приоритет бирж: от лучшей к запасным
    exchange_priority = [
        ('binance', ccxt.binance),
        ('kucoin', ccxt.kucoin),
        ('gate', ccxt.gate),
        ('mexc', ccxt.mexc)
    ]
    
    for exchange_name, exchange_class in exchange_priority:
        try:
            exchange = exchange_class()
            markets = exchange.load_markets()
            
            if ccxt_symbol in markets:
                # Проверяем что это спот (не фьючерсы)
                market_info = markets[ccxt_symbol]
                is_spot = not market_info.get('swap') and not market_info.get('future')
                
                if is_spot:
                    # Тест получения данных
                    test_data = exchange.fetch_ohlcv(ccxt_symbol, '1h', limit=2)
                    if test_data and len(test_data) > 0:
                        price = test_data[-1][4]
                        print(f"🔍 {exchange_name}: найден {ccxt_symbol} (спот), цена: {price}")
                        
                        # Проверяем доступность данных
                        sufficient, available_days, listing_date = check_data_availability(
                            exchange, ccxt_symbol, '30m', required_days, exchange_name
                        )
                        
                        if sufficient:
                            print(f"✅ Выбран {exchange_name} с {available_days} днями данных")
                            return exchange, exchange_name, ccxt_symbol
                        else:
                            print(f"⏭️ {exchange_name}: недостаточно данных, пробуем следующую биржу...")
                            continue
                else:
                    print(f"❌ {exchange_name}: {ccxt_symbol} это фьючерсы, не спот")
            else:
                print(f"❌ {exchange_name}: символ {ccxt_symbol} не найден")
                    
            time.sleep(0.2)  # Rate limit между биржами
            
        except Exception as e:
            print(f"❌ {exchange_name}: {str(e)[:100]}")
            continue
    
    raise ValueError(f"❌ Символ {symbol} -> {ccxt_symbol} не найден с достаточными данными (≥{required_days} дней) ни на одной из поддерживаемых бирж")

def fetch_ohlcv_ccxt(symbol: str, timeframe: str, since_ms: int, limit: int, required_days: int = 365) -> pd.DataFrame:
    # Автоматически выбираем лучшую биржу с достаточными данными
    exchange, exchange_name, actual_symbol = find_best_exchange(symbol, required_days)
    print(f"🏛️ Используем биржу: {exchange_name} с символом: {actual_symbol}")
    
    all_rows = []
    last_since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(actual_symbol, timeframe=timeframe, since=last_since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        # Обновляем since: последний open_time + 1 мс
        last_open = batch[-1][0]
        last_since = last_open + 1
        # anti-rate-limit
        time.sleep(0.2)
        # Остановимся, если пришло меньше лимита (конец истории)
        if len(batch) < limit:
            break
    df = pd.DataFrame(all_rows, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy().sort_values("time").set_index("time")

    # Базовые индикаторы
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=cfg.rsi_len).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=cfg.ema_fast).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=cfg.ema_slow).ema_indicator()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=cfg.atr_len)
    df["atr"] = atr.average_true_range()

    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]

    # Доходности и волатильность
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["rvol_20"] = df["ret_1"].rolling(20).std()

    # Мульти-таймфрейм (агрегируем в 1h и тащим обратно через asof)
    if cfg.higher_tf:
        agg = df[["open","high","low","close","volume"]].resample(cfg.higher_tf).agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna()
        agg["rsi_h"] = ta.momentum.RSIIndicator(agg["close"], window=cfg.higher_rsi_len).rsi()
        agg["ema_h"] = ta.trend.EMAIndicator(agg["close"], window=cfg.higher_ema).ema_indicator()
        # мержим как "последнее известное значение" на момент бара train TF
        df = df.merge(agg[["rsi_h","ema_h"]], left_index=True, right_index=True, how="left")
        df[["rsi_h","ema_h"]] = df[["rsi_h","ema_h"]].ffill()

    df = add_advanced_features(df)

    df = df.dropna()
    df = df.reset_index()  # Сохраняем время как колонку "time"
    return df

# Добавьте эту функцию после строки 158 (после функции add_features):

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить продвинутые фичи для улучшения модели"""
    
    # ===== MARKET STRUCTURE FEATURES =====
    # Higher Highs / Lower Lows patterns
    df['higher_highs'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['lower_lows'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['higher_lows'] = (df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))
    df['lower_highs'] = (df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2))
    
    # Price Action Patterns
    df['inside_bar'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
    df['outside_bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    df['doji'] = abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1
    df['hammer'] = ((df['close'] > df['open']) & 
                   ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) & 
                   ((df['high'] - df['close']) < 0.1 * (df['close'] - df['open'])))
    
    # ===== VOLATILITY REGIME FEATURES =====
    df['vol_percentile'] = df['atr'].rolling(100).rank(pct=True)
    df['high_vol_regime'] = df['vol_percentile'] > 0.8
    df['low_vol_regime'] = df['vol_percentile'] < 0.2
    df['vol_expanding'] = df['atr'].rolling(10).mean() > df['atr'].rolling(20).mean()
    
    # ===== TIME-BASED FEATURES =====
    # Add hour information (assuming index is datetime)
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
    else:
        df['hour'] = pd.to_datetime(df['time']).dt.hour if 'time' in df.columns else 12  # Default
    
    df['is_asia_session'] = df['hour'].isin([0,1,2,3,4,5,6,7,8])
    df['is_london_session'] = df['hour'].isin([8,9,10,11,12,13,14,15,16])
    df['is_ny_session'] = df['hour'].isin([13,14,15,16,17,18,19,20,21])
    df['is_liquid_hours'] = df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    # ===== ORIGINAL FEATURES (улучшенные) =====
    # Волатильность breakouts
    df['vol_breakout'] = (df['volume'] / df['volume'].rolling(20).mean()) > 2.0
    df['vol_breakout_strong'] = (df['volume'] / df['volume'].rolling(20).mean()) > 3.0
    
    # Price momentum (расширенный)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # Support/Resistance levels (улучшенные)
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['high_50'] = df['high'].rolling(50).max()
    df['low_50'] = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] / df['high_20']) > 0.98
    df['near_support'] = (df['close'] / df['low_20']) < 1.02
    df['near_major_resistance'] = (df['close'] / df['high_50']) > 0.99
    df['near_major_support'] = (df['close'] / df['low_50']) < 1.01
    
    # Volume-Price Trend
    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
    df['vpt_sma'] = df['vpt'].rolling(14).mean()
    
    # ===== TREND STRENGTH FEATURES =====
    # ADX calculation (simplified)
    high_low_diff = df['high'] - df['low']
    high_close_diff = abs(df['high'] - df['close'].shift())
    low_close_diff = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low_diff, high_close_diff, low_close_diff], axis=1).max(axis=1)
    df['adx'] = true_range.rolling(14).mean()  # Simplified ADX
    
    # ===== ДОПОЛНИТЕЛЬНЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ =====
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    # Commodity Channel Index
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    
    # ===== PRICE VELOCITY AND ACCELERATION =====
    df['price_velocity'] = df['close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # ===== RELATIVE STRENGTH =====
    df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(10).mean()
    
    return df


def make_triple_barrier_labels(df: pd.DataFrame, tp_pct=0.01, sl_pct=0.01, horizon_bars=8):
    """
    Triple-barrier разметка:
    - LONG, если цена достигла TP раньше SL в пределах horizon
    - SHORT, если цена достигла SL раньше TP
    - HOLD, если не достигнуты ни TP, ни SL до конца горизонта
    """

    labels = []
    closes = df["close"].values
    opens = df["open"].values  # Добавляем массив open для реалистичного входа

    for i in range(len(closes)):
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вход на следующей свече (open[i+1])
        if i + 1 >= len(opens):
            labels.append("HOLD")
            continue
            
        entry = opens[i + 1]  # Реалистичный вход: open следующей свечи после сигнала
        tp_level = entry * (1 + tp_pct)
        sl_level = entry * (1 - sl_pct)

        outcome = "HOLD"  # по умолчанию
        for j in range(1, horizon_bars + 1):
            check_idx = i + j
            if check_idx >= len(closes):
                break

            high = df["high"].iloc[check_idx]
            low = df["low"].iloc[check_idx]

            # Проверка: какое событие произошло первым
            if high >= tp_level:
                outcome = "LONG"
                break
            elif low <= sl_level:
                outcome = "SHORT"
                break

        labels.append(outcome)

    return pd.Series(labels, index=df.index, name="target")



def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    # обратная пропорция частоте класса
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: total/(len(classes)*cnt) for c, cnt in zip(classes, counts)}
    return weights


def calculate_position_size(prediction_proba, max_position=1.0):
    """Scale position size based on model confidence"""
    # Get probability of the predicted class
    confidence = np.max(prediction_proba)
    # Scale position (e.g., 0.5-1.0 range)
    position_scale = min(1.0, max(0.5, (confidence - 0.33) * 1.5))
    return max_position * position_scale

def detailed_backtest(df: pd.DataFrame,
                     predictions: np.ndarray,
                     probabilities: np.ndarray,
                     cfg: Config,
                     confidence_threshold: float = 0.58,
                     p_hit_map: dict | None = None) -> Dict:
    """Новая версия детального бэктеста, согласованная с live-логикой risk_metrics.

    Особенности:
      - Использует те же фильтры качества: edge_ok, prob_ok, net_rr_ok, conf_edge_ok, cal_ev_ok
      - Поддерживает калиброванный p_hit (ev_calibrated)
      - Позиционирование через sizing_mode (risk или margin) как в live
      - Стоимость сделки (round-trip) из risk_config; половина применяется на входе и выходе
      - Динамическое обновление депозита для risk-сайзинга (компаундинг)
      - Отчет по отклоненным сигналам с причинами
    """
    if 'time' not in df.columns:
        # Попытаемся восстановить столбец времени из индекса
        df = df.copy()
        df['time'] = df.index

    try:
        from risk_metrics import compute_signal_metrics, load_risk_config
    except Exception as e:
        raise RuntimeError(f"Не удалось импортировать risk_metrics: {e}")

    risk_cfg_static = load_risk_config()
    # Начальный баланс из risk_config (если есть) иначе 10k
    starting_equity = float(risk_cfg_static.get('deposit', 10_000))
    equity = starting_equity

    horizon = cfg.horizon_bars
    symbol_key = cfg.symbol.replace('/', '') if getattr(cfg, 'symbol', None) else ''
    symbol_over = (risk_cfg_static.get('symbols') or {}).get(symbol_key, {})
    round_trip_cost = symbol_over.get('round_trip_cost_pct', risk_cfg_static.get('default_round_trip_cost_pct', 0.002))
    half_cost = round_trip_cost / 2.0

    trades: list[dict] = []
    equity_curve: list[float] = [equity]
    open_position = None
    total_signals = 0  # сгенерированные (до фильтров)
    accepted_signals = 0
    filtered_reason_counts: dict[str, int] = {}

    # Функция фильтра причин
    def record_filter(reason: str):
        filtered_reason_counts[reason] = filtered_reason_counts.get(reason, 0) + 1

    # Основной проход по барам (до последнего, где хватает горизонта)
    last_entry_index = len(df) - horizon - 1
    for i in range(max(0, last_entry_index)):
        row = df.iloc[i]

        # Закрытие открытой позиции на этом шаге? (выход уже моделировался при открытии — мы симулируем сразу)
        # В новой логике мы моделируем исход сделки сразу после открытия, поэтому нет поэтапного обновления open_position.
        # Оставлено для совместимости если решим моделировать trailing в будущем.

        # Проверяем потенциальный сигнал
        pred = predictions[i]
        probs = probabilities[i]
        confidence = float(np.max(probs))

        if pred not in (1, 2) or confidence < confidence_threshold:
            equity_curve.append(equity)
            continue

        total_signals += 1

        side = 'LONG' if pred == 1 else 'SHORT'
        price = float(row['close'])
        atr = float(row.get('atr', np.nan))
        if not np.isfinite(atr) or atr <= 0 or price <= 0:
            record_filter('invalid_atr_or_price')
            equity_curve.append(equity)
            continue

        # Формируем уровни SL/TP (сырые, без комиссий) с приоритетом сеточных (Variant B)
        if 'grid_tp_mult_chosen' in df.columns and np.isfinite(row.get('grid_tp_mult_chosen', np.nan)):
            g_tp = float(row['grid_tp_mult_chosen'])
            g_sl = float(row.get('grid_sl_mult_chosen', cfg.sl_mult))
            tp_price = price + g_tp * atr if side == 'LONG' else price - g_tp * atr
            sl_price = price - g_sl * atr if side == 'LONG' else price + g_sl * atr
        else:
            tp_price = price + cfg.tp_mult * atr if side == 'LONG' else price - cfg.tp_mult * atr
            sl_price = price - cfg.sl_mult * atr if side == 'LONG' else price + cfg.sl_mult * atr

        raw_signal = {
            'symbol': symbol_key,
            'signal': side,
            'price': price,  # до комиссии
            'take_profit': tp_price,
            'stop_loss': sl_price,
            'confidence': confidence,
        }

        # Динамическое обновление депозита для вычисления размера позиции
        # Копируем risk config и заменяем deposit на текущее equity
        risk_cfg_dynamic = json.loads(json.dumps(risk_cfg_static))  # deep copy простым способом
        risk_cfg_dynamic['deposit'] = equity

        metrics = compute_signal_metrics(raw_signal, risk_cfg_dynamic)

        # Добавим калибровку если есть карта
        if p_hit_map:
            # ищем подходящий бин
            p_hit_cal = None
            bins = p_hit_map.get('bins', [])
            for b in bins:
                # поддержка форматов {'conf_min','conf_max'} или {'low','high'}
                low = b.get('conf_min', b.get('low'))
                high = b.get('conf_max', b.get('high'))
                if low is not None and high is not None and low <= confidence < high:
                    p_hit_cal = b.get('p_hit')
                    break
            if p_hit_cal is None:
                p_hit_cal = p_hit_map.get('overall_p_hit', confidence)
            # повторно пересчитаем EV(cal)
            if 0 <= p_hit_cal <= 1:
                net_target = metrics['net_target_pct']
                net_stop = metrics['net_stop_pct']
                ev_cal_pct = p_hit_cal * net_target - (1 - p_hit_cal) * net_stop
                metrics['p_hit_cal'] = p_hit_cal
                metrics['ev_calibrated_pct'] = ev_cal_pct
                metrics['ev_cal_flag'] = ev_cal_pct >= 0
                if metrics.get('position_notional'):
                    metrics['ev_calibrated_usd'] = metrics['position_notional'] * ev_cal_pct
                # Применяем правило отклонения если включено
                if metrics.get('reject_negative_calibrated_ev') and ev_cal_pct < 0:
                    metrics['cal_ev_ok'] = False

        # Фильтрация
        if not metrics.get('edge_ok', True):
            record_filter('edge_fail')
            equity_curve.append(equity)
            continue
        if not metrics.get('prob_ok', True):
            record_filter('prob_fail')
            equity_curve.append(equity)
            continue
        if metrics.get('net_rr_ok') is False:
            record_filter('net_rr_fail')
            equity_curve.append(equity)
            continue
        if metrics.get('conf_edge_ok') is False:
            record_filter('conf_edge_fail')
            equity_curve.append(equity)
            continue
        if metrics.get('cal_ev_ok') is False:
            record_filter('cal_ev_fail')
            equity_curve.append(equity)
            continue

        accepted_signals += 1

        position_notional = metrics.get('position_notional')
        if not position_notional or position_notional <= 0:
            record_filter('no_position_size')
            equity_curve.append(equity)
            continue

        # Симуляция исхода (triple barrier) — ищем первое достижение tp/sl в пределах горизонта
        tp_level = tp_price
        sl_level = sl_price
        outcome = 'time_exit'
        exit_price_raw = df['close'].iloc[i + horizon]  # по времени если не достигнуты уровни
        bars_to_exit = horizon
        hit_tp_flag = False
        hit_sl_flag = False
        # Tracking max favorable / adverse excursion in pct relative to entry (raw, before costs)
        max_fav_pct = 0.0
        max_adv_pct = 0.0
        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break
            high = df['high'].iloc[i + j]
            low = df['low'].iloc[i + j]
            if side == 'LONG':
                hit_tp = high >= tp_level
                hit_sl = low <= sl_level
                # Favorable excursion: high vs entry
                fav = (high - price) / price * 100 if price > 0 else 0
                adv = (low - price) / price * 100 if price > 0 else 0  # будет отрицательным или меньшим
                if fav > max_fav_pct:
                    max_fav_pct = fav
                if adv < max_adv_pct:
                    max_adv_pct = adv  # adv более отрицательный – худшее
            else:
                hit_tp = low <= tp_level
                hit_sl = high >= sl_level
                # For SHORT, favorable = entry - low, adverse = entry - high
                fav = (price - low) / price * 100 if price > 0 else 0
                adv = (price - high) / price * 100 if price > 0 else 0  # adverse станет отрицательным или меньшим
                if fav > max_fav_pct:
                    max_fav_pct = fav
                if adv < max_adv_pct:
                    max_adv_pct = adv
            if hit_tp and hit_sl:
                outcome = 'time_exit'
                bars_to_exit = j
                break
            if hit_tp:
                exit_price_raw = tp_level
                outcome = 'take_profit'
                bars_to_exit = j
                hit_tp_flag = True
                break
            if hit_sl:
                exit_price_raw = sl_level
                outcome = 'stop_loss'
                bars_to_exit = j
                hit_sl_flag = True
                break

        # Исполнение с учетом половинных затрат
        if side == 'LONG':
            entry_exec = price * (1 + half_cost)
            exit_exec = exit_price_raw * (1 - half_cost)
            pnl_usd = (exit_exec - entry_exec) * (position_notional / price)
        else:
            entry_exec = price * (1 - half_cost)
            exit_exec = exit_price_raw * (1 + half_cost)
            pnl_usd = (entry_exec - exit_exec) * (position_notional / price)

        equity += pnl_usd
        equity_curve.append(equity)

        tp_mult_used_val = float(row.get('grid_tp_mult_chosen')) if row.get('grid_tp_mult_chosen') else cfg.tp_mult
        sl_mult_used_val = float(row.get('grid_sl_mult_chosen')) if row.get('grid_sl_mult_chosen') else cfg.sl_mult
        atr_at_entry = float(row.get('atr', np.nan))
        if side == 'LONG':
            per_unit_risk = price - sl_level
        else:
            per_unit_risk = sl_level - price
        per_unit_risk = max(per_unit_risk, 1e-12)
        position_units = position_notional / price if price > 0 else 0
        risk_usd = position_units * per_unit_risk
        realized_rr = pnl_usd / risk_usd if risk_usd > 0 else np.nan
        if side == 'LONG':
            gross_pnl_before_cost = (exit_price_raw - price) * position_units
        else:
            gross_pnl_before_cost = (price - exit_price_raw) * position_units
        gross_return_pct_before_cost = gross_pnl_before_cost / starting_equity * 100
        # saturation flag (в рамках текущего df) – ближе к макс tp в сетке
        max_tp_obs = np.nanmax(df.get('grid_tp_mult_chosen')) if 'grid_tp_mult_chosen' in df.columns else tp_mult_used_val
        saturation_flag = bool(np.isfinite(max_tp_obs) and tp_mult_used_val >= 0.98 * max_tp_obs)
        trades.append({
            'entry_time': row['time'],
            'exit_time': df['time'].iloc[i + horizon] if (i + horizon) < len(df) else df['time'].iloc[-1],
            'side': side,
            'entry_price_raw': price,
            'exit_price_raw': exit_price_raw,
            'entry_price_exec': entry_exec,
            'exit_price_exec': exit_exec,
            'tp_level': tp_level,
            'sl_level': sl_level,
            'horizon_bars': horizon,
            'bars_to_exit': bars_to_exit,
            'confidence': confidence,
            'p_hit_cal': metrics.get('p_hit_cal'),
            'ev_calibrated_pct': metrics.get('ev_calibrated_pct'),
            'ev_naive_pct': metrics.get('ev_naive_pct'),
            'position_notional': position_notional,
            'pnl_usd': pnl_usd,
            'return_pct': pnl_usd / starting_equity * 100,
            'gross_return_pct_before_cost': gross_return_pct_before_cost,
            'realized_rr': realized_rr,
            'risk_usd': risk_usd,
            'exit_reason': outcome,
            'flag_hit_tp': hit_tp_flag,
            'flag_hit_sl': hit_sl_flag,
            'flag_time_exit': outcome == 'time_exit',
            'geometry_source': row.get('grid_tp_mult_chosen') and 'grid' or 'static',
            'tp_mult_used': tp_mult_used_val,
            'sl_mult_used': sl_mult_used_val,
            'saturation_flag': saturation_flag,
            'threshold_used': confidence_threshold,
            'label_original': pred,
            'net_rr_chosen': float(row.get('grid_net_rr_chosen')) if row.get('grid_net_rr_chosen') else None,
            'atr_at_entry': atr_at_entry,
            'equity_before': equity - pnl_usd,
            'equity_after': equity,
            'mfe_pct': max_fav_pct,              # Max Favorable Excursion (% от entry)
            'mae_pct': max_adv_pct,              # Max Adverse Excursion (% - обычно отрицательная величина)
        })

    # === Аггрегация метрик ===
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        total_pnl = trades_df['pnl_usd'].sum()
        wins = trades_df[trades_df['pnl_usd'] > 0]
        losses = trades_df[trades_df['pnl_usd'] < 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) else 0
        avg_win_pct = wins['return_pct'].mean() if not wins.empty else 0
        avg_loss_pct = losses['return_pct'].mean() if not losses.empty else 0
        loss_sum = abs(losses['pnl_usd'].sum())
        profit_sum = wins['pnl_usd'].sum()
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series / peak - 1) * 100
        max_drawdown = abs(drawdown.min())
        total_return_pct = (equity - starting_equity) / starting_equity * 100
        trades_by_exit = trades_df['exit_reason'].value_counts().to_dict()
        ev_cal_sum = trades_df['ev_calibrated_pct'].sum(skipna=True) if 'ev_calibrated_pct' in trades_df else 0
        ev_naive_sum = trades_df['ev_naive_pct'].sum(skipna=True) if 'ev_naive_pct' in trades_df else 0
        metrics = {
            'total_trades': int(len(trades_df)),
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            # Итоговая прибыль в USDT по всем сделкам
            'total_pnl_usd': float(total_pnl),
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'avg_win_pct': avg_win_pct if np.isfinite(avg_win_pct) else 0,
            'avg_loss_pct': avg_loss_pct if np.isfinite(avg_loss_pct) else 0,
            # Балансы для наглядности
            'starting_equity': float(starting_equity),
            'final_balance': equity,
            'trades_by_exit_reason': trades_by_exit,
            'ev_calibrated_sum': ev_cal_sum,
            'ev_naive_sum': ev_naive_sum,
            # Geometry stats (accepted trades only)
            'geometry_usage': trades_df['geometry_source'].value_counts().to_dict() if 'geometry_source' in trades_df else {},
            'tp_mult_used_dist': {
                'mean': float(trades_df['tp_mult_used'].mean()) if 'tp_mult_used' in trades_df else None,
                'median': float(trades_df['tp_mult_used'].median()) if 'tp_mult_used' in trades_df else None,
                'p90': float(trades_df['tp_mult_used'].quantile(0.9)) if 'tp_mult_used' in trades_df else None,
            },
            'sl_mult_used_dist': {
                'mean': float(trades_df['sl_mult_used'].mean()) if 'sl_mult_used' in trades_df else None,
                'median': float(trades_df['sl_mult_used'].median()) if 'sl_mult_used' in trades_df else None,
                'p10': float(trades_df['sl_mult_used'].quantile(0.1)) if 'sl_mult_used' in trades_df else None,
            },
        }

        # --- Exit PnL Breakdown ---
        try:
            exit_groups = trades_df.groupby('exit_reason')
            gross_profit_total = wins['pnl_usd'].sum()
            gross_loss_total = abs(losses['pnl_usd'].sum())
            contrib = {}
            for reason, grp in exit_groups:
                gp = grp[grp['pnl_usd'] > 0]['pnl_usd'].sum()
                gl = abs(grp[grp['pnl_usd'] < 0]['pnl_usd'].sum())
                contrib[reason] = {
                    'count': int(len(grp)),
                    'gross_profit': float(gp),
                    'gross_loss': float(gl),
                    'avg_pnl_usd': float(grp['pnl_usd'].mean()) if len(grp) else 0.0,
                    'avg_return_pct': float(grp['return_pct'].mean()) if len(grp) else 0.0,
                    'profit_trades': int((grp['pnl_usd'] > 0).sum()),
                    'loss_trades': int((grp['pnl_usd'] < 0).sum()),
                }
            # Средние по типам
            tp_df = trades_df[trades_df['exit_reason'] == 'take_profit']
            sl_df = trades_df[trades_df['exit_reason'] == 'stop_loss']
            te_df = trades_df[trades_df['exit_reason'] == 'time_exit']
            breakdown_block = {
                'average_tp_gain_pct': float(tp_df['return_pct'].mean()) if not tp_df.empty else 0.0,
                'average_sl_loss_pct': float(sl_df['return_pct'].mean()) if not sl_df.empty else 0.0,
                'average_time_exit_gain_pct': float(te_df[te_df['pnl_usd'] > 0]['return_pct'].mean()) if not te_df.empty and (te_df['pnl_usd'] > 0).any() else 0.0,
                'average_time_exit_loss_pct': float(te_df[te_df['pnl_usd'] < 0]['return_pct'].mean()) if not te_df.empty and (te_df['pnl_usd'] < 0).any() else 0.0,
                'profitable_time_exit_count': int((te_df['pnl_usd'] > 0).sum()) if not te_df.empty else 0,
                'losing_time_exit_count': int((te_df['pnl_usd'] < 0).sum()) if not te_df.empty else 0,
                'gross_profit_contribution_pct': {},
                'gross_loss_contribution_pct': {},
                'by_exit_reason': contrib,
                # Суммарные значения по причинам выхода
                'gross_profit_total': float(gross_profit_total),
                'gross_loss_total': float(gross_loss_total),
                'net_pnl_usd': float(gross_profit_total - gross_loss_total)
            }
            if gross_profit_total > 0:
                for r,v in contrib.items():
                    if v['gross_profit'] > 0:
                        breakdown_block['gross_profit_contribution_pct'][r] = float(v['gross_profit'] / gross_profit_total * 100)
            if gross_loss_total > 0:
                for r,v in contrib.items():
                    if v['gross_loss'] > 0:
                        breakdown_block['gross_loss_contribution_pct'][r] = float(v['gross_loss'] / gross_loss_total * 100)
            metrics['exit_pnl_breakdown'] = breakdown_block

            # === MFE/MAE агрегаты и метрики эффективности ===
            if 'mfe_pct' in trades_df.columns:
                try:
                    # Общие агрегаты
                    metrics['overall_mfe_mean_pct'] = float(trades_df['mfe_pct'].mean())
                    metrics['overall_mae_mean_pct'] = float(trades_df['mae_pct'].mean()) if 'mae_pct' in trades_df.columns else None
                    # Нереализованный потенциал (только если MFE > фактического результата)
                    unrealized_series = (trades_df['mfe_pct'] - trades_df['return_pct']).clip(lower=0)
                    metrics['overall_avg_unrealized_left_pct'] = float(unrealized_series.mean())
                    # По time_exit
                    if not te_df.empty:
                        metrics['time_exit_mfe_mean_pct'] = float(te_df['mfe_pct'].mean())
                        metrics['time_exit_mae_mean_pct'] = float(te_df['mae_pct'].mean()) if 'mae_pct' in te_df.columns else None
                        metrics['time_exit_mfe_median_pct'] = float(te_df['mfe_pct'].median())
                        metrics['time_exit_mae_median_pct'] = float(te_df['mae_pct'].median()) if 'mae_pct' in te_df.columns else None
                        metrics['time_exit_avg_unrealized_left_pct'] = float(((te_df['mfe_pct'] - te_df['return_pct']).clip(lower=0)).mean())
                    # Эффективность: насколько процента MFE реализовано (return_pct / mfe_pct)
                    # Добавляем колонку для дальнейшего анализа
                    trades_df['efficiency_ratio'] = trades_df.apply(lambda r: (r['return_pct'] / r['mfe_pct']) if r['mfe_pct'] and r['mfe_pct'] != 0 else np.nan, axis=1)
                    # Сырые (могут быть >1 или отрицательные) и обрезанные [0,1]
                    eff_raw = trades_df['efficiency_ratio']
                    eff_clipped = eff_raw.clip(lower=0, upper=1)
                    metrics['overall_efficiency_ratio_mean'] = float(eff_raw.mean(skipna=True))
                    metrics['overall_efficiency_ratio_median'] = float(eff_raw.median(skipna=True))
                    metrics['overall_efficiency_ratio_mean_clipped'] = float(eff_clipped.mean(skipna=True))
                    metrics['overall_efficiency_ratio_median_clipped'] = float(eff_clipped.median(skipna=True))
                    # По time_exit
                    if not te_df.empty:
                        te_eff_raw = trades_df.loc[trades_df['exit_reason']=='time_exit','efficiency_ratio']
                        te_eff_clip = te_eff_raw.clip(lower=0, upper=1)
                        metrics['time_exit_efficiency_ratio_mean'] = float(te_eff_raw.mean(skipna=True))
                        metrics['time_exit_efficiency_ratio_median'] = float(te_eff_raw.median(skipna=True))
                        metrics['time_exit_efficiency_ratio_mean_clipped'] = float(te_eff_clip.mean(skipna=True))
                        metrics['time_exit_efficiency_ratio_median_clipped'] = float(te_eff_clip.median(skipna=True))
                except Exception as mfe_err:
                    metrics['mfe_aggregation_error'] = str(mfe_err)[:200]
        except Exception as _br_err:
            metrics['exit_pnl_breakdown_error'] = str(_br_err)[:200]
    else:
        metrics = {
            'total_trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'final_balance': starting_equity,
            'trades_by_exit_reason': {},
            'ev_calibrated_sum': 0,
            'ev_naive_sum': 0
        }

    metrics['total_signals_generated'] = total_signals
    metrics['signals_after_filtering'] = accepted_signals
    metrics['filter_ratio'] = accepted_signals / total_signals if total_signals > 0 else 0
    metrics['filtered_reason_counts'] = filtered_reason_counts
    metrics['acceptance_rate'] = metrics['filter_ratio']
    # If there are trades, add simple R:R realized distribution proxies
    if not trades_df.empty and 'tp_mult_used' in trades_df and 'sl_mult_used' in trades_df:
        try:
            rr_series = trades_df['tp_mult_used'] / trades_df['sl_mult_used']
            metrics['realized_geometry_rr'] = {
                'mean': float(rr_series.mean()),
                'median': float(rr_series.median()),
                'p90': float(rr_series.quantile(0.9))
            }
        except Exception:
            pass

    return {
        'metrics': metrics,
        'trades': trades_df,
        'equity_curve': equity_curve
    }

# =====================
# p_hit calibration utilities
# =====================

def _simulate_tp_sl_outcome(df: pd.DataFrame, idx: int, side: str, tp_level: float, sl_level: float, horizon: int) -> str:
    """Сканирует вперед до horizon баров и возвращает 'take_profit', 'stop_loss' или 'none'.
    Порядок проверки: если в баре достигаются оба уровня (high>=tp и low<=sl), считается 'none' (неопределённо) чтобы не искажать статистику."""
    last = min(idx + horizon, len(df) - 1)
    for j in range(idx + 1, last + 1):
        high = df['high'].iloc[j]
        low = df['low'].iloc[j]
        hit_tp = False
        hit_sl = False
        if side == 'LONG':
            hit_tp = high >= tp_level
            hit_sl = low <= sl_level
        else:  # SHORT
            hit_tp = low <= tp_level
            hit_sl = high >= sl_level
        if hit_tp and hit_sl:
            return 'none'  # неоднозначно
        if hit_tp:
            return 'take_profit'
        if hit_sl:
            return 'stop_loss'
    return 'none'

def calibrate_p_hit(calib_df: pd.DataFrame, model, feature_cols: List[str], adaptive_params: dict, cfg: Config) -> dict:
    """Строит бин-карту p_hit на первой половине тестового периода.
    Бины только по confidence (p_dir)."""
    if len(calib_df) < 50:
        return {'bins': [], 'overall_p_hit': 0.5, 'note': 'not_enough_data'}
    X_calib = calib_df[feature_cols].values
    proba = model.predict_proba(X_calib)
    horizon = adaptive_params.get('horizon_bars', cfg.horizon_bars)
    tp_mult = adaptive_params.get('tp_mult', cfg.tp_mult)
    sl_mult = adaptive_params.get('sl_mult', cfg.sl_mult)
    candidates = []
    closes = calib_df['close'].values
    atrs = calib_df['atr'].values if 'atr' in calib_df.columns else np.zeros(len(calib_df))
    for i in range(len(calib_df) - horizon):
        p_hold, p_long, p_short = proba[i]
        if max(p_long, p_short) <= 0.5:  # минимальный базовый порог включения
            continue
        side = 'LONG' if p_long > p_short else 'SHORT'
        confidence = max(p_long, p_short)
        price = closes[i]
        atr_val = atrs[i]
        if atr_val <= 0 or price <= 0:
            continue
        if side == 'LONG':
            tp_level = price + tp_mult * atr_val
            sl_level = price - sl_mult * atr_val
        else:
            tp_level = price - tp_mult * atr_val
            sl_level = price + sl_mult * atr_val
        outcome = _simulate_tp_sl_outcome(calib_df, i, side, tp_level, sl_level, horizon)
        if outcome in ('take_profit', 'stop_loss'):
            candidates.append({'confidence': confidence, 'outcome': outcome})
    if not candidates:
        return {'bins': [], 'overall_p_hit': 0.5, 'note': 'no_candidates'}
    cand_df = pd.DataFrame(candidates)
    overall = (cand_df['outcome'] == 'take_profit').mean()
    # Бины
    edges = [round(x,2) for x in np.arange(0.50, 0.801, 0.02)]
    bins = []
    for b_start, b_end in zip(edges[:-1], edges[1:]):
        mask = (cand_df['confidence'] >= b_start) & (cand_df['confidence'] < b_end)
        sub = cand_df[mask]
        if len(sub) >= 5:
            p_hit = (sub['outcome'] == 'take_profit').mean()
            bins.append({'conf_min': b_start, 'conf_max': b_end, 'count': int(len(sub)), 'p_hit': float(p_hit)})
    return {'bins': bins, 'overall_p_hit': float(overall), 'updated': datetime.now().isoformat(), 'note': 'bin_by_confidence'}

def estimate_p_hit(conf: float, mapping: dict) -> float:
    """Возвращает оценку p_hit по калибровочной карте."""
    bins = mapping.get('bins', [])
    if not bins:
        return mapping.get('overall_p_hit', 0.5)
    for b in bins:
        if conf >= b['conf_min'] and conf < b['conf_max']:
            return b['p_hit']
    # Если выше последнего бина
    last = bins[-1]
    if conf >= last['conf_min']:
        return last['p_hit']
    return mapping.get('overall_p_hit', 0.5)

def should_trade(df_row, pred, confidence, min_confidence=0.58):
    """Сбалансированные фильтры для оптимального количества качественных сделок"""
    
    # Умеренно-строгая уверенность
    if confidence < min_confidence:  
        return False
    
    # Фильтр по экстремальным RSI (чуть более мягкий)
    if 'rsi' in df_row.index:
        if df_row['rsi'] > 78 or df_row['rsi'] < 22:  # Смягчить границы
            return False
    
    # Более гибкий трендовый фильтр
    if 'ema_fast' in df_row.index and 'ema_slow' in df_row.index:
        ema_diff_pct = (df_row['ema_fast'] - df_row['ema_slow']) / df_row['ema_slow'] * 100
        
        # Разрешить торговлю при слабых трендах, запретить при сильных противоположных
        if pred == 1 and ema_diff_pct < -1.0:  # LONG только если нисходящий тренд не сильный
            return False
        if pred == 2 and ema_diff_pct > 1.0:   # SHORT только если восходящий тренд не сильный
            return False
    
    return True
    
    return True

# =====================
# Основной поток
# =====================
if __name__ == "__main__":
    # Парсим аргументы командной строки
    args = parse_arguments()
    
    # Обновляем конфигурацию с параметрами командной строки
    cfg.symbol = args.symbol
    cfg.tf_train = args.timeframe  
    cfg.lookback_days = args.days
    cfg.tp_pct = args.tp_pct
    cfg.sl_pct = args.sl_pct
    cfg.use_versioning = not args.no_versioning  # Инвертируем флаг

    # Переопределение horizon если передан
    if getattr(args, 'horizon', None) is not None:
        if args.horizon <= 0:
            print(f"⚠️  Игнорируем некорректный horizon={args.horizon}; оставляем {cfg.horizon_bars}")
        else:
            if args.horizon > 40:
                print(f"⚠️  Предупреждение: большой horizon={args.horizon} может увеличить время расчёта и риск переобучения")
            cfg.horizon_bars = int(args.horizon)
            print(f"🔁 Horizon override: horizon_bars={cfg.horizon_bars}")

    # Возможное отключение сохранения всех трейдов
    if getattr(args, 'no_debug_trades', False):
        cfg.debug_save_all_thresholds = False

    # Переопределение geometry диапазонов если переданы
    if getattr(args, 'geom_tp', None):
        try:
            s,e,st = [float(x) for x in args.geom_tp.split(',')]
            cfg.geometry_tp_range = (s,e,st)
            print(f"[CLI] geometry_tp_range => {cfg.geometry_tp_range}")
        except Exception as ex:
            print(f"⚠️ Не удалось распарсить --geom-tp: {ex}")

    if getattr(args, 'no_geom_auto_expand', False):
        cfg.geometry_auto_expand = False
        print("🔒 Auto-expansion of TP range disabled (--no-geom-auto-expand)")

    if getattr(args, 'no_anti_saturation', False):
        cfg.geometry_anti_saturation_enable = False
        print("🚫 Anti-saturation penalty disabled (--no-anti-saturation)")

    if getattr(args, 'apply_dynamic_tp', False):
        cfg.dynamic_tp_apply = True
        if getattr(args, 'dynamic_tp_percentile', None) is not None:
            cfg.dynamic_tp_percentile = float(args.dynamic_tp_percentile)
        print(f"🧪 Dynamic TP apply enabled at percentile {cfg.dynamic_tp_percentile}")
        if getattr(args, 'baseline_no_cap', False):
            cfg.baseline_no_cap = True
    else:
        if getattr(args, 'dynamic_tp_percentile', None) is not None:
            cfg.dynamic_tp_percentile = float(args.dynamic_tp_percentile)
            print(f"(Suggest-only) dynamic TP percentile set to {cfg.dynamic_tp_percentile}")
    if getattr(args, 'geom_sl', None):
        try:
            s,e,st = [float(x) for x in args.geom_sl.split(',')]
            cfg.geometry_sl_range = (s,e,st)
            print(f"[CLI] geometry_sl_range => {cfg.geometry_sl_range}")
        except Exception as ex:
            print(f"⚠️ Не удалось распарсить --geom-sl: {ex}")

    # Если запрошен только график efficient frontier из существующего meta
    if getattr(args, 'frontier_from_meta', None):
        try:
            plot_efficient_frontier(args.frontier_from_meta, show_plot=(args.frontier_show or not args.silent))
        except Exception as ex:
            print(f"❌ Ошибка построения efficient frontier: {ex}")
        sys.exit(0)
    
    print(f"Версионирование: {'Включено' if cfg.use_versioning else 'Отключено'}")
    
    # Определяем тип разделения данных
    if args.backtest_days is not None:
        # Фиксированное количество дней для бэктеста
        print(f"Разделение данных: Последние {args.backtest_days} дней для бэктеста")
        cfg.backtest_mode = 'fixed_days'
        cfg.backtest_days = args.backtest_days
    else:
        # Процентное разделение
        cfg.test_size_frac = 1.0 - args.train_split  # Конвертируем из доли обучения в долю теста
        cfg.backtest_mode = 'percentage'
        if args.train_split == 1.0:
            print(f"Разделение данных: 100% обучение (без бэктеста)")
        else:
            print(f"Разделение данных: {args.train_split*100:.0f}% обучение / {cfg.test_size_frac*100:.0f}% тест")
    
    since_ts = pd.Timestamp.utcnow() - pd.Timedelta(days=cfg.lookback_days)  # Already UTC aware
    print(f"🔍 Получение данных {cfg.symbol} {cfg.tf_train} с {since_ts}")
    print(f"📊 Требуемый период: {cfg.lookback_days} дней")
    
    # Передаем требуемое количество дней для проверки доступности данных
    df = fetch_ohlcv_ccxt(cfg.symbol, cfg.tf_train, int(since_ts.timestamp()*1000), cfg.limit_per_call, cfg.lookback_days)
    print("Raw bars:", len(df))
    
    # Показываем временные границы загруженных данных
    if len(df) > 0:
        first_date = pd.to_datetime(df['time'].iloc[0], utc=True)
        last_date = pd.to_datetime(df['time'].iloc[-1], utc=True)
        print(f"📅 Загружены данные с {first_date} по {last_date}")
        print(f"   └─ Период: {(last_date - first_date).days + 1} дней, {len(df)} свечей")
        print(f"   └─ Запрошено: {cfg.lookback_days} дней")

    df = add_features(df, cfg)
    print("With features:", len(df))
    
    # Определяем рыночный режим для адаптивных параметров
    market_regime = detect_market_regime(df)
    print(f"🔍 Market regime value: {market_regime}")
    
    adaptive_params = get_adaptive_parameters(market_regime, cfg.symbol)
    
    print(f"🎯 Обнаружен режим рынка: {market_regime}")
    print(f"📊 Адаптивные параметры: TP={adaptive_params['tp_mult']:.1f}x, SL={adaptive_params['sl_mult']:.1f}x, Horizon={adaptive_params['horizon_bars']}, Threshold={adaptive_params['confidence_threshold']:.3f}")
    
    # === Variant B: Asymmetric grid relabeling replaces simple triple-barrier ===
    # Load risk config if available to extract min_net_rr & cost for feasibility
    try:
        from risk_metrics import load_risk_config
        _rc_variantB = load_risk_config()
        overrides_vb = (_rc_variantB.get('symbols') or {}).get(cfg.symbol.replace('/', ''), {})
        vb_min_net_rr = overrides_vb.get('min_net_rr', _rc_variantB.get('min_net_rr'))
        vb_cost = overrides_vb.get('round_trip_cost_pct', _rc_variantB.get('default_round_trip_cost_pct', 0.002))
    except Exception:
        vb_min_net_rr = None
        vb_cost = 0.002

    # Use dynamic geometry configuration (with optional auto-expansion)
    grid_labels, grid_stats = make_asymmetric_grid_labels(
        df,
        atr_window=cfg.atr_len,
        horizon_bars=adaptive_params['horizon_bars'],
        tp_mult_range=cfg.geometry_tp_range,
        sl_mult_range=cfg.geometry_sl_range,
        min_net_rr=vb_min_net_rr,
        round_trip_cost_pct=vb_cost,
        selection='first_hit_best_rr',
        prefer='rr_then_speed',
        auto_expand=cfg.geometry_auto_expand,
        saturation_threshold=cfg.geometry_saturation_threshold,
        expansion_step=cfg.geometry_expansion_step,
        max_tp_cap=cfg.geometry_max_tp_mult,
    )
    # --- Adaptive SL pass (если включено и слишком много выборов на нижней границе SL) ---
    # Теперь адаптация возможна только после первой разметки grid_labels (target)
    # Если вы видите это сообщение — адаптация SL не применялась, так как 'target' ещё не создан. Это не ошибка.
    # Адаптация будет применена ниже, после создания df['target'].
    df['target'] = grid_labels
    # --- Adaptive SL pass (если включено и слишком много выборов на нижней границе SL) ---
    if cfg.geometry_sl_adaptive:
        try:
            sl_choices = df.get('grid_sl_mult_chosen')
            if sl_choices is not None:
                sl_choices_arr = pd.Series(sl_choices)
                lower_bound = cfg.geometry_sl_range[0]
                if 'target' in df.columns:
                    directional_mask = df['target'].isin([1,2])
                    if directional_mask.any():
                        pct_at_min_sl = float(((sl_choices_arr[directional_mask] == lower_bound).sum()) / directional_mask.sum())
                        if pct_at_min_sl >= cfg.geometry_sl_saturation_threshold and lower_bound < cfg.geometry_sl_enforced_min:
                            # Повторяем только если реально есть место для повышения
                            new_sl_range = (cfg.geometry_sl_enforced_min, cfg.geometry_sl_range[1], cfg.geometry_sl_range[2])
                            print(f"🔁 Adaptive SL: {pct_at_min_sl:.1%} направленных выборов на SL={lower_bound}. Повторная разметка с новым SL диапазоном {new_sl_range}.")
                            grid_labels2, grid_stats2 = make_asymmetric_grid_labels(
                                df,
                                atr_window=cfg.atr_len,
                                horizon_bars=adaptive_params['horizon_bars'],
                                tp_mult_range=grid_stats.get('grid_tp_range', cfg.geometry_tp_range),
                                sl_mult_range=new_sl_range,
                                min_net_rr=vb_min_net_rr,
                                round_trip_cost_pct=vb_cost,
                                selection='first_hit_best_rr',
                                prefer='rr_then_speed',
                                auto_expand=False,  # TP уже расширен при первом прогоне
                            )
                            if grid_labels2 is not None and isinstance(grid_labels2, (list, pd.Series)) and len(grid_labels2) == len(df):
                                df['target'] = [ {'HOLD':0,'LONG':1,'SHORT':2}.get(x,0) for x in grid_labels2 ]
                            else:
                                print("⚠️ Adaptive SL: grid_labels2 некорректен или не совпадает по длине с df")
                            # Объединим статистику
                            grid_stats = {
                                **grid_stats,
                                'adaptive_sl_applied': True,
                                'adaptive_sl_prev_min': lower_bound,
                                'adaptive_sl_new_min': cfg.geometry_sl_enforced_min,
                                'adaptive_sl_pct_at_prev_min': pct_at_min_sl,
                                'adaptive_sl_stats_new': grid_stats2,
                            }
                else:
                    print("⚠️ Adaptive SL: В DataFrame отсутствует колонка 'target', пропускаем адаптацию.")
        except Exception as _adp_err:
            print(f"⚠️ Adaptive SL error: {_adp_err}")

    # Convert string labels to numeric
    label_map = {"HOLD": 0, "LONG": 1, "SHORT": 2}
    df["target"] = df["target"].map(label_map)

    # Define feature columns (расширенный список)
    feature_cols = [
        "close","volume","rsi","ema_fast","ema_slow","atr",
        "macd","macd_signal","macd_hist","bb_width",
        "ret_1","ret_3","ret_6","ret_12","rvol_20",
        # Продвинутые фичи:
        "momentum_5", "momentum_10", "momentum_20", "vol_breakout", "vol_breakout_strong",
        "near_resistance", "near_support", "near_major_resistance", "near_major_support",
        "vpt_sma", "stoch_k", "stoch_d", "williams_r", "cci",
        # Market structure:
        "higher_highs", "lower_lows", "higher_lows", "lower_highs",
        "inside_bar", "outside_bar", "doji", "hammer",
        # Volatility regimes:
        "vol_percentile", "high_vol_regime", "low_vol_regime", "vol_expanding",
        # Time features:
        "is_asia_session", "is_london_session", "is_ny_session", "is_liquid_hours",
        # Trend and momentum:
        "adx", "price_velocity", "price_acceleration", "rsi_divergence"
    ]

    if cfg.higher_tf:
        feature_cols += ["rsi_h","ema_h"]

    # Фильтруем фичи, которые действительно существуют в DataFrame
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"⚠️  Отсутствующие фичи: {missing_features}")
    
    print(f"✅ Используем {len(available_features)} фичей из {len(feature_cols)} запланированных")
    feature_cols = available_features

    # THEN do the Train/Test split after adding all features
    print(f"\n🎯 РАЗДЕЛЕНИЕ ДАННЫХ:")
    
    if cfg.backtest_mode == 'fixed_days':
        # Фиксированное количество дней для бэктеста
        # Вычисляем количество свечей в N днях для данного таймфрейма
        if cfg.tf_train.endswith('m'):
            minutes_per_candle = int(cfg.tf_train[:-1])
        elif cfg.tf_train.endswith('h'):
            minutes_per_candle = int(cfg.tf_train[:-1]) * 60
        elif cfg.tf_train.endswith('d'):
            minutes_per_candle = int(cfg.tf_train[:-1]) * 24 * 60
        else:
            minutes_per_candle = 30  # default fallback
        
        candles_per_day = 24 * 60 // minutes_per_candle
        backtest_candles = cfg.backtest_days * candles_per_day
        
        # Убеждаемся, что у нас достаточно данных
        if backtest_candles >= len(df):
            print(f"❌ ОШИБКА: Запрошено {cfg.backtest_days} дней ({backtest_candles} свечей) для бэктеста,")
            print(f"   но доступно только {len(df)} свечей. Уменьшите --backtest-days или увеличьте --days")
            exit(1)
        
        split_idx = len(df) - backtest_candles
        actual_test_days = backtest_candles / candles_per_day
        test_size_frac = backtest_candles / len(df)
        
        print(f"📅 Режим: Фиксированные {cfg.backtest_days} дней для бэктеста")
        print(f"🔢 Свечей в день: {candles_per_day}, Бэктест: {backtest_candles} свечей")
        
    else:
        # Процентное разделение
        split_idx = int((1 - cfg.test_size_frac) * len(df))
        test_size_frac = cfg.test_size_frac
        backtest_candles = len(df) - split_idx
        
        if cfg.tf_train.endswith('m'):
            minutes_per_candle = int(cfg.tf_train[:-1])
        elif cfg.tf_train.endswith('h'):
            minutes_per_candle = int(cfg.tf_train[:-1]) * 60
        elif cfg.tf_train.endswith('d'):
            minutes_per_candle = int(cfg.tf_train[:-1]) * 24 * 60
        else:
            minutes_per_candle = 30
        
        candles_per_day = 24 * 60 // minutes_per_candle
        actual_test_days = backtest_candles / candles_per_day
        
        if test_size_frac == 0:
            print(f"📅 Режим: 100% обучение (полное использование данных)")
        else:
            print(f"📅 Режим: Процентное разделение {(1-test_size_frac)*100:.0f}%/{test_size_frac*100:.0f}%")
            print(f"🔢 Свечей в день: {candles_per_day}, Бэктест: {backtest_candles} свечей ({actual_test_days:.0f} дней)")
    
    # Создаем строку с информацией о бэктесте для имени файла
    if cfg.backtest_mode == 'fixed_days':
        backtest_info = f"bt{cfg.backtest_days}d"
    else:
        # Для процентного разделения добавляем реальное количество дней
        if test_size_frac == 0:
            # 100% обучения, без бэктеста
            backtest_info = "full100pct"
        else:
            # Вычисляем реальное количество дней для бэктеста
            actual_test_days = int(backtest_candles / candles_per_day)
            backtest_info = f"bt{int(test_size_frac*100)}pct_{actual_test_days}d"
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Показываем временные границы обучения и бэктеста
    train_start = train_df['time'].iloc[0]
    train_end = train_df['time'].iloc[-1]
    
    print(f"\n🎯 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"📈 Обучение: {train_start.strftime('%Y-%m-%d %H:%M')} → {train_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"   └─ {(train_end - train_start).days + 1} дней, {len(train_df):,} свечей")
    
    if test_size_frac == 0:
        # 100% обучения - нет бэктеста
        print(f"📊 Бэктест: Отсутствует (100% данных для обучения)")
        print(f"💯 Соотношение: 100% обучение / 0% тест")
    else:
        # Есть бэктест
        test_start = test_df['time'].iloc[0]
        test_end = test_df['time'].iloc[-1]
        print(f"📊 Бэктест: {test_start.strftime('%Y-%m-%d %H:%M')} → {test_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"   └─ {(test_end - test_start).days + 1} дней, {len(test_df):,} свечей")
        print(f"💯 Соотношение: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% обучение / {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% тест")

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    
    if test_size_frac == 0:
        # 100% обучения - нет тестового набора
        X_test = np.array([]).reshape(0, len(feature_cols))
        y_test = np.array([])
    else:
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values

    # Временная кросс-валидация (оценка стабильности)
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits_cv)
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        # Веса классов
        cw = compute_class_weights(y_tr)
        w_tr = np.array([cw[c] for c in y_tr])
        model_cv = XGBClassifier(**cfg.xgb_params)
        model_cv.fit(X_tr, y_tr,
                     sample_weight=w_tr,
                     eval_set=[(X_val, y_val)],
                     verbose=False)
        pred_val = model_cv.predict(X_val)
        f1 = f1_score(y_val, pred_val, average="macro")
        acc = accuracy_score(y_val, pred_val)
        cv_scores.append((f1, acc))
        print(f"Fold {fold+1}: F1={f1:.3f}, Acc={acc:.3f}")

    # Финальная модель на всем train с early stopping по куску теста
    cw = compute_class_weights(y_train)
    # w_train = np.array([cw[c] for c in y_train])
    
    if test_size_frac == 0:
        # Для 100% обучения убираем early stopping
        xgb_params_full = cfg.xgb_params.copy()
        xgb_params_full.pop('early_stopping_rounds', None)
        model = XGBClassifier(**xgb_params_full)
    else:
        model = XGBClassifier(**cfg.xgb_params)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    if test_size_frac == 0:
        # 100% обучения - без eval_set
        model.fit(X_train, y_train,
                  sample_weight=sample_weights,
                  verbose=True)
        print("\n📊 100% обучение: Holdout-оценка недоступна (нет тестового набора)")
    else:
        # Обычное обучение с eval_set
        model.fit(X_train, y_train,
                  sample_weight=sample_weights,
                  eval_set=[(X_test, y_test)],
                  verbose=True)
        
        # Holdout-оценка
        y_pred = model.predict(X_test)
        print("\nHoldout classification report:\n", classification_report(y_test, y_pred, digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # === Статистика классов ===
    print("\n=== Class distribution ===")
    print(df["target"].value_counts())
    print(df["target"].value_counts(normalize=True))

    # === Визуализация ===
    if not args.silent:
        plt.figure(figsize=(8, 6))
        
        # Подсчет классов
        class_counts = df["target"].value_counts().sort_index()
        
        # Правильные метки для классов (соответствуют mapping: HOLD=0, LONG=1, SHORT=2)
        class_labels = ['HOLD (0)', 'LONG (1)', 'SHORT (2)']
        # Цвета в том же порядке индексов: 0->серый, 1->зелёный (LONG), 2->красный (SHORT)
        class_colors = ['gray', 'green', 'red']
        
        # Создание pie chart с понятными подписями
        plt.pie(class_counts.values, 
               labels=class_labels, 
               autopct='%1.1f%%', 
               startangle=90, 
               colors=class_colors,
               textprops={'fontsize': 12})
        
        plt.title("Распределение классов\n(HOLD = Не торговать, SHORT = Продавать, LONG = Покупать)", 
                  fontsize=14, pad=20)
        plt.axis('equal')  # Круглый график
        plt.show()

        # Важность признаков
        plt.figure(figsize=(10, 6))
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)
        plt.barh(np.array(feature_cols)[sorted_idx], importance[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("🔇 Графики отключены (режим --silent)")

    # Сохранение артефактов
    base, timestamp = create_versioned_basename(
        cfg.symbol, cfg.tf_train, cfg.lookback_days, backtest_info,
        cfg.use_versioning, cfg.version_format
    )
    
    print(f"💾 Сохранение модели с базовым именем: {base}")
    
    model_path = os.path.join(cfg.out_dir, f"xgb_{base}.json")
    model.save_model(model_path)

    meta = {
        "config": asdict(cfg),
        "feature_cols": feature_cols,
        "cv_scores": cv_scores,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "labeling": {
            "method": "asymmetric_grid",
            "grid_stats": grid_stats if 'grid_stats' in locals() else None,
        },
        "version_info": {
            "created_at": datetime.now().isoformat(),
            "timestamp": timestamp if cfg.use_versioning else None,
            "versioning_enabled": cfg.use_versioning,
            "base_name": base
        }
    }
    with open(os.path.join(cfg.out_dir, f"meta_{base}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {model_path}")

    # === Оптимизация порога уверенности (СНАЧАЛА!) ===
    print("\n=== Threshold Optimization ===")
    
    if test_size_frac == 0:
        print("⚠️ Оптимизация порогов недоступна при 100% обучении")
        print("💡 Используйте адаптивный порог из настроек для данного режима рынка")
        optimal_threshold = adaptive_params['confidence_threshold']
        bt_metrics = {
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'total_return_pct': 0,
            'max_drawdown_pct': 0
        }
        # Инициализируем пустые результаты для режима 100% обучения
        results = []
        threshold_summary_rows = []
        p_hit_map = None
        selection_reason = "100% обучение без бэктеста"
    else:
        print("\n� КАЛИБРОВКА P_HIT:")
        print("📊 Делим бэктест пополам: calibration (50%) + validation (50%)")
        half = len(test_df) // 2
        calib_df = test_df.iloc[:half].copy()
        valid_df = test_df.iloc[half:].copy()
        
        calib_start = calib_df['time'].iloc[0]
        calib_end = calib_df['time'].iloc[-1]
        valid_start = valid_df['time'].iloc[0]
        valid_end = valid_df['time'].iloc[-1]
        
        print(f"   🔧 Calibration: {calib_start.strftime('%Y-%m-%d %H:%M')} → {calib_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"      └─ {(calib_end - calib_start).days + 1} дней, {len(calib_df):,} свечей")
        print(f"   ✅ Validation: {valid_start.strftime('%Y-%m-%d %H:%M')} → {valid_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"      └─ {(valid_end - valid_start).days + 1} дней, {len(valid_df):,} свечей")

        # Калибровка p_hit
        p_hit_map = calibrate_p_hit(calib_df, model, feature_cols, adaptive_params, cfg)
        print(f"p_hit overall={p_hit_map.get('overall_p_hit'):.3f}, bins={len(p_hit_map.get('bins', []))}")

        # Предсказания на validation части
        X_valid = valid_df[feature_cols].values
        y_valid_pred = model.predict(X_valid)
        y_valid_proba = model.predict_proba(X_valid)

        # Для ускоренного threshold scan сохраняем проекции
        conf_list = []
        pred_dir_list = []  # 1=LONG,2=SHORT
        for row_proba in y_valid_proba:
            p_hold, p_long, p_short = row_proba
            if max(p_long, p_short) > p_hold:
                conf_list.append(max(p_long, p_short))
                pred_dir_list.append(1 if p_long > p_short else 2)
            else:
                conf_list.append(p_hold)  # не будет торговаться
                pred_dir_list.append(0)
        valid_df['pred_dir'] = pred_dir_list
        valid_df['confidence'] = conf_list

        # Предварительно рассчитываем tp/sl уровни с учетом ПЕРСИГНАЛЬНЫХ grid-множителей (если есть)
        horizon = adaptive_params.get('horizon_bars', cfg.horizon_bars)
        atr_vals = valid_df['atr'].values if 'atr' in valid_df.columns else np.zeros(len(valid_df))
        closes_vals = valid_df['close'].values
        tp_levels = []
        sl_levels = []
        for i in range(len(valid_df)):
            atr_v = atr_vals[i]
            price = closes_vals[i]
            if atr_v <= 0 or price <= 0:
                tp_levels.append(None)
                sl_levels.append(None)
                continue
            # Используем grid_tp_mult_chosen/sl если присутствуют, иначе fallback к адаптивным
            row_tp_mult = valid_df.iloc[i].get('grid_tp_mult_chosen') if 'grid_tp_mult_chosen' in valid_df.columns else None
            row_sl_mult = valid_df.iloc[i].get('grid_sl_mult_chosen') if 'grid_sl_mult_chosen' in valid_df.columns else None
            if row_tp_mult is None or not np.isfinite(row_tp_mult):
                row_tp_mult = adaptive_params.get('tp_mult', cfg.tp_mult)
            if row_sl_mult is None or not np.isfinite(row_sl_mult):
                row_sl_mult = adaptive_params.get('sl_mult', cfg.sl_mult)
            if valid_df['pred_dir'].iloc[i] == 1:  # LONG
                tp_levels.append(price + row_tp_mult * atr_v)
                sl_levels.append(price - row_sl_mult * atr_v)
            elif valid_df['pred_dir'].iloc[i] == 2:
                tp_levels.append(price - row_tp_mult * atr_v)
                sl_levels.append(price + row_sl_mult * atr_v)
            else:
                tp_levels.append(None)
                sl_levels.append(None)
        valid_df['tp_level'] = tp_levels
        valid_df['sl_level'] = sl_levels

        # === Dynamic TP Option B: Two-pass (pre-scan) application ===
        dynamic_tp_info = None
        original_tp_mult_vector = None  # will store unclipped tp multipliers for baseline comparison
        if getattr(cfg, 'dynamic_tp_apply', False) and getattr(cfg, 'dynamic_tp_percentile', None) is not None:
            try:
                dyn_pct = float(cfg.dynamic_tp_percentile)
                min_obs = int(getattr(cfg, 'dynamic_tp_min_observations', 100))
                clip_factor = 1.10  # k=1.10 per design
                # Preliminary backtest at lowest threshold (0.50) to gather MFE samples BEFORE clipping
                prelim_threshold = 0.50
                prelim_bt = detailed_backtest(
                    valid_df,
                    model.predict(valid_df[feature_cols].values),
                    model.predict_proba(valid_df[feature_cols].values),
                    cfg,
                    confidence_threshold=prelim_threshold,
                    p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
                )
                prelim_trades = prelim_bt.get('trades') if isinstance(prelim_bt, dict) else None
                mfe_samples = []
                tp_trade_returns = []  # to approximate target_pct distribution
                tp_mult_used_list = []
                if prelim_trades is not None and len(prelim_trades):
                    for _, tr in prelim_trades.iterrows():
                        mfe = tr.get('mfe_pct')
                        if isinstance(mfe, (int, float)) and mfe > 0:
                            mfe_samples.append(mfe)
                        ex_type = tr.get('exit_reason') or tr.get('exit_type')
                        if ex_type in ('take_profit', 'TP'):
                            ret_pct = tr.get('return_pct')
                            if isinstance(ret_pct, (int, float)) and ret_pct > 0:
                                tp_trade_returns.append(ret_pct)
                            tm = tr.get('tp_mult_used')
                            if isinstance(tm, (int, float)) and tm > 0:
                                tp_mult_used_list.append(tm)
                if len(mfe_samples) >= min_obs:
                    import numpy as _np
                    mfe_arr = _np.array(mfe_samples)
                    suggested_mfe_pct = float(_np.quantile(mfe_arr, dyn_pct))
                    median_tp_mult_used = float(_np.median(tp_mult_used_list)) if tp_mult_used_list else None
                    median_target_pct = float(_np.median(tp_trade_returns)) if tp_trade_returns else None
                    atr_price_ratio_med = None
                    if median_tp_mult_used and median_tp_mult_used > 0 and median_target_pct and median_target_pct > 0:
                        # target_pct ≈ tp_mult_used * (ATR/Price) * 100 => (ATR/Price) ≈ target_pct / tp_mult_used
                        atr_price_ratio_med = (median_target_pct) / median_tp_mult_used
                    tp_mult_cap = None
                    if atr_price_ratio_med and atr_price_ratio_med > 0:
                        raw_cap = (suggested_mfe_pct * clip_factor) / atr_price_ratio_med
                        # Guard rails
                        grid_stats_local = meta.get('labeling', {}).get('grid_stats', {}) if 'meta' in locals() else {}
                        grid_tp_range = grid_stats_local.get('grid_tp_range') or getattr(cfg, 'geometry_tp_range', (1.0,4.0,0.25))
                        grid_tp_max = grid_tp_range[1] if isinstance(grid_tp_range, (list, tuple)) and len(grid_tp_range) >= 2 else 4.0
                        tp_mult_cap = float(min(max(raw_cap, 1.0), grid_tp_max))
                    if tp_mult_cap:
                        # Apply clipping to chosen tp mults (directional only)
                        if 'grid_tp_mult_chosen' in valid_df.columns:
                            original_tp_mult = valid_df['grid_tp_mult_chosen'].values.copy()
                            original_tp_mult_vector = original_tp_mult.copy()  # preserve for baseline
                            dir_mask = valid_df['pred_dir'].isin([1,2]) if 'pred_dir' in valid_df.columns else _np.ones(len(valid_df), dtype=bool)
                            clip_mask = (valid_df['grid_tp_mult_chosen'] > tp_mult_cap) & dir_mask
                            pct_clipped = float(_np.sum(clip_mask) / max(1, _np.sum(dir_mask)))
                            valid_df.loc[clip_mask, 'grid_tp_mult_chosen'] = tp_mult_cap
                            # Recompute tp/sl levels after clipping
                            new_tp_levels = []
                            new_sl_levels = []
                            for i, row in valid_df.iterrows():
                                atr_v = row.get('atr')
                                price = row.get('close')
                                if not isinstance(atr_v, (int, float)) or not isinstance(price, (int, float)) or atr_v <= 0 or price <= 0:
                                    new_tp_levels.append(row.get('tp_level'))
                                    new_sl_levels.append(row.get('sl_level'))
                                    continue
                                tp_m = row.get('grid_tp_mult_chosen')
                                sl_m = row.get('grid_sl_mult_chosen')
                                if row.get('pred_dir') == 1:  # LONG
                                    new_tp_levels.append(price + tp_m * atr_v)
                                    new_sl_levels.append(price - sl_m * atr_v)
                                elif row.get('pred_dir') == 2:  # SHORT
                                    new_tp_levels.append(price - tp_m * atr_v)
                                    new_sl_levels.append(price + sl_m * atr_v)
                                else:
                                    new_tp_levels.append(None); new_sl_levels.append(None)
                            valid_df['tp_level'] = new_tp_levels
                            valid_df['sl_level'] = new_sl_levels
                            # Post-clip distribution
                            after_mean = float(_np.nanmean(valid_df['grid_tp_mult_chosen'])) if len(valid_df) else None
                            after_median = float(_np.nanmedian(valid_df['grid_tp_mult_chosen'])) if len(valid_df) else None
                            before_mean = float(_np.nanmean(original_tp_mult)) if len(original_tp_mult) else None
                            before_median = float(_np.nanmedian(original_tp_mult)) if len(original_tp_mult) else None
                            dynamic_tp_info = {
                                'mode': 'applied',
                                'requested_percentile': dyn_pct,
                                'samples': len(mfe_samples),
                                'suggested_mfe_pct': suggested_mfe_pct,
                                'approx_median_tp_mult_prelim': median_tp_mult_used,
                                'approx_median_target_pct_prelim': median_target_pct,
                                'approx_atr_price_ratio_median': atr_price_ratio_med,
                                'tp_mult_cap': tp_mult_cap,
                                'clip_factor': clip_factor,
                                'pct_clipped': pct_clipped,
                                'tp_mult_mean_before': before_mean,
                                'tp_mult_median_before': before_median,
                                'tp_mult_mean_after': after_mean,
                                'tp_mult_median_after': after_median,
                                'samples_source_threshold': prelim_threshold,
                                'note': 'dynamic TP cap applied before threshold scan'
                            }
                        else:
                            dynamic_tp_info = {
                                'mode': 'suggest_only',
                                'requested_percentile': dyn_pct,
                                'samples': len(mfe_samples),
                                'reason': 'grid_tp_mult_chosen_missing_cannot_clip'
                            }
                    else:
                        dynamic_tp_info = {
                            'mode': 'suggest_only',
                            'requested_percentile': dyn_pct,
                            'samples': len(mfe_samples),
                            'suggested_mfe_pct': suggested_mfe_pct,
                            'reason': 'cannot_compute_tp_mult_cap'
                        }
                else:
                    dynamic_tp_info = {
                        'mode': 'suggest_only',
                        'requested_percentile': dyn_pct,
                        'samples': len(mfe_samples),
                        'reason': f'not_enough_samples_min_{min_obs}'
                    }
            except Exception as e:
                dynamic_tp_info = {
                    'mode': 'error',
                    'error': str(e)
                }

        # === ПОЛНОЕ СКАНИРОВАНИЕ ПОРОГА: detailed_backtest для каждого threshold ===
        X_valid_full = valid_df[feature_cols].values
        y_valid_pred_full = model.predict(X_valid_full)
        y_valid_proba_full = model.predict_proba(X_valid_full)
        # Расширяем верхнюю границу до 0.90 для поиска лучшего PF в более высоких confidence зонах
        # Список threshold'ов используемый для основной оптимизации
        thresholds_list = list(np.round(np.arange(0.50, 0.901, 0.01), 2))

        # --- (OPTIONAL) BASELINE NO-CAP SCAN (до основной оптимизации, чтобы не портить состояние) ---
        baseline_scan_results = None
        baseline_optimal = None
        if getattr(args, 'baseline_no_cap', False):
            if original_tp_mult_vector is None or not (getattr(cfg, 'dynamic_tp_apply', False) and dynamic_tp_info and dynamic_tp_info.get('mode') == 'applied'):
                print("⚠️ baseline_no_cap: пропуск (нет применённого dynamic TP или отсутствуют исходные множители)")
            else:
                try:
                    print("\n🔁 BASELINE (no-cap) threshold scan — сравнение без обрезки TP множителей")
                    # Создаем копию validation фрейма с исходными TP мультипликаторами
                    baseline_df = valid_df.copy()
                    baseline_df['grid_tp_mult_chosen'] = original_tp_mult_vector
                    # Пересчёт TP/SL уровней под baseline
                    new_tp_levels = []
                    new_sl_levels = []
                    for i, row in baseline_df.iterrows():
                        atr_v = row.get('atr'); price = row.get('close')
                        if not isinstance(atr_v, (int, float)) or not isinstance(price, (int, float)) or atr_v <= 0 or price <= 0:
                            new_tp_levels.append(None); new_sl_levels.append(None); continue
                        tp_m = row.get('grid_tp_mult_chosen'); sl_m = row.get('grid_sl_mult_chosen')
                        if row.get('pred_dir') == 1:
                            new_tp_levels.append(price + tp_m * atr_v)
                            new_sl_levels.append(price - sl_m * atr_v)
                        elif row.get('pred_dir') == 2:
                            new_tp_levels.append(price - tp_m * atr_v)
                            new_sl_levels.append(price + sl_m * atr_v)
                        else:
                            new_tp_levels.append(None); new_sl_levels.append(None)
                    baseline_df['tp_level'] = new_tp_levels
                    baseline_df['sl_level'] = new_sl_levels
                    # Предсказания уже рассчитаны выше (confidence/pred_dir) — переиспользуем baseline_df['confidence'] и 'pred_dir'
                    baseline_results = []
                    # Быстрый цикл по тем же thresholds_list
                    for thr in thresholds_list:
                        bt_bl = detailed_backtest(
                            baseline_df,
                            y_valid_pred_full,
                            y_valid_proba_full,
                            cfg,
                            confidence_threshold=thr,
                            p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
                        )
                        m_bl = bt_bl['metrics'] if isinstance(bt_bl, dict) else {}
                        baseline_results.append({
                            'threshold': thr,
                            'total_trades': m_bl.get('total_trades', 0),
                            'win_rate': m_bl.get('win_rate', 0.0),
                            'profit_factor': m_bl.get('profit_factor', 0.0),
                            'tp_count': m_bl.get('trades_by_exit_reason', {}).get('take_profit', 0) if m_bl else 0,
                            'sl_count': m_bl.get('trades_by_exit_reason', {}).get('stop_loss', 0) if m_bl else 0,
                            'time_exit_count': m_bl.get('trades_by_exit_reason', {}).get('time_exit', 0) if m_bl else 0
                        })
                    # Отбор baseline optimal по тем же критериям
                    valid_bl = [r for r in baseline_results if r['total_trades'] > 0]
                    if valid_bl:
                        max_pf_bl = max(r['profit_factor'] for r in valid_bl)
                        if max_pf_bl < 1.0:
                            baseline_optimal = min(valid_bl, key=lambda x: x['threshold'])
                        else:
                            profitable_bl = [r for r in valid_bl if r['profit_factor'] >= 1.5 and r['total_trades'] >= 30]
                            if profitable_bl:
                                baseline_optimal = max(profitable_bl, key=lambda x: x['profit_factor'])
                            else:
                                moderate_bl = [r for r in valid_bl if r['profit_factor'] >= 1.2]
                                if moderate_bl:
                                    baseline_optimal = max(moderate_bl, key=lambda x: x['profit_factor'])
                                else:
                                    substantial_bl = [r for r in valid_bl if r['total_trades'] >= 30]
                                    if substantial_bl:
                                        baseline_optimal = max(substantial_bl, key=lambda x: x['win_rate'])
                                    else:
                                        baseline_optimal = max(valid_bl, key=lambda x: x['win_rate'])
                    baseline_scan_results = baseline_results
                except Exception as bl_err:
                    print(f"⚠️ baseline_no_cap error: {bl_err}")
                    baseline_scan_results = {'error': str(bl_err)}

        # --- (OPTIONAL) SENSITIVITY PASS FOR ALTERNATIVE DYNAMIC TP PERCENTILES ---
        sensitivity_results = None
        if getattr(args, 'dynamic_tp_sensitivity', ''):
            raw_list = [s.strip() for s in str(args.dynamic_tp_sensitivity).split(',') if s.strip()]
            alt_pcts = []
            for s in raw_list:
                try:
                    v = float(s)
                    if 0 < v < 1:
                        alt_pcts.append(v)
                except ValueError:
                    pass
            # Удалим основной percentile если он в списке (чтобы не дублировать)
            base_pct = getattr(cfg, 'dynamic_tp_percentile', None)
            alt_pcts = [p for p in alt_pcts if p != base_pct]
            if alt_pcts:
                print(f"\n🔍 SENSITIVITY: динамические перцентили {alt_pcts}")
                sensitivity_results = []
                # Подмножество порогов для ускоренного сравнения (можно переопределить CLI)
                if getattr(args, 'sensitivity_thresholds', ''):
                    try:
                        _thr_list = [float(x) for x in args.sensitivity_thresholds.split(',') if x.strip()]
                        sensitivity_thresholds = sorted(set(_thr_list))
                    except Exception:
                        sensitivity_thresholds = sorted(set([0.73,0.75,0.78,0.79,0.80,0.81,0.82,0.84,0.86,0.88,0.90]))
                else:
                    sensitivity_thresholds = sorted(set([0.73,0.75,0.78,0.79,0.80,0.81,0.82,0.84,0.86,0.88,0.90]))
                min_trades_req = int(getattr(args, 'sensitivity_min_trades', 30))
                # Общие предсказания / proba используем повторно
                X_valid_full = valid_df[feature_cols].values
                y_valid_pred_full = model.predict(X_valid_full)
                y_valid_proba_full = model.predict_proba(X_valid_full)
                import numpy as _np
                percentile_summary = []
                for alt_pct in alt_pcts:
                    try:
                        if original_tp_mult_vector is None:
                            # Если не было клиппинга — используем текущее как "оригинал"
                            original_vec = valid_df['grid_tp_mult_chosen'].values.copy() if 'grid_tp_mult_chosen' in valid_df.columns else None
                        else:
                            original_vec = original_tp_mult_vector.copy()
                        if original_vec is None:
                            print(f"⚠️ Sensitivity {alt_pct}: нет исходных tp_mult, пропуск")
                            continue
                        temp_df = valid_df.copy()
                        temp_df['grid_tp_mult_chosen'] = original_vec  # восстановление
                        # 1) Preliminary backtest для сбора MFE (как в основном блоке)
                        prelim_bt_alt = detailed_backtest(
                            temp_df,
                            y_valid_pred_full,
                            y_valid_proba_full,
                            cfg,
                            confidence_threshold=0.50,
                            p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
                        )
                        trades_alt = prelim_bt_alt.get('trades') if isinstance(prelim_bt_alt, dict) else None
                        mfe_samples_alt = []
                        tp_trade_returns_alt = []
                        tp_mult_used_alt = []
                        if trades_alt is not None and len(trades_alt):
                            for _, tr in trades_alt.iterrows():
                                mfev = tr.get('mfe_pct')
                                if isinstance(mfev, (int,float)) and mfev>0:
                                    mfe_samples_alt.append(mfev)
                                ex_type = tr.get('exit_reason') or tr.get('exit_type')
                                if ex_type in ('take_profit','TP'):
                                    rp = tr.get('return_pct')
                                    if isinstance(rp,(int,float)) and rp>0:
                                        tp_trade_returns_alt.append(rp)
                                    tmv = tr.get('tp_mult_used')
                                    if isinstance(tmv,(int,float)) and tmv>0:
                                        tp_mult_used_alt.append(tmv)
                        if len(mfe_samples_alt) < int(getattr(cfg,'dynamic_tp_min_observations',100)):
                            sensitivity_results.append({
                                'percentile': alt_pct,
                                'status': 'insufficient_samples',
                                'samples': len(mfe_samples_alt)
                            })
                            continue
                        mfe_arr_alt = _np.array(mfe_samples_alt)
                        suggested_mfe_pct_alt = float(_np.quantile(mfe_arr_alt, alt_pct))
                        median_tp_mult_used_alt = float(_np.median(tp_mult_used_alt)) if tp_mult_used_alt else None
                        median_target_pct_alt = float(_np.median(tp_trade_returns_alt)) if tp_trade_returns_alt else None
                        atr_price_ratio_med_alt = None
                        if median_tp_mult_used_alt and median_tp_mult_used_alt>0 and median_target_pct_alt and median_target_pct_alt>0:
                            atr_price_ratio_med_alt = median_target_pct_alt/median_tp_mult_used_alt
                        tp_mult_cap_alt = None
                        clip_factor = dynamic_tp_info.get('clip_factor',1.10) if dynamic_tp_info else 1.10
                        if atr_price_ratio_med_alt and atr_price_ratio_med_alt>0:
                            raw_cap_alt = (suggested_mfe_pct_alt * clip_factor)/atr_price_ratio_med_alt
                            grid_tp_range = getattr(cfg,'geometry_tp_range',(1.0,4.0,0.25))
                            grid_tp_max = grid_tp_range[1]
                            tp_mult_cap_alt = float(min(max(raw_cap_alt,1.0), grid_tp_max))
                        # Применяем альтернативный cap на копии
                        pct_clipped_alt = None
                        if tp_mult_cap_alt and 'grid_tp_mult_chosen' in temp_df.columns:
                            dir_mask_alt = temp_df['pred_dir'].isin([1,2]) if 'pred_dir' in temp_df.columns else _np.ones(len(temp_df),dtype=bool)
                            before_vec = temp_df['grid_tp_mult_chosen'].values.copy()
                            clip_mask_alt = (temp_df['grid_tp_mult_chosen']>tp_mult_cap_alt) & dir_mask_alt
                            temp_df.loc[clip_mask_alt,'grid_tp_mult_chosen']=tp_mult_cap_alt
                            pct_clipped_alt = float(_np.sum(clip_mask_alt)/max(1,_np.sum(dir_mask_alt)))
                            # Пересчёт tp/sl
                            new_tp_levels_alt=[]; new_sl_levels_alt=[]
                            for irow,row in temp_df.iterrows():
                                atr_v=row.get('atr'); price=row.get('close')
                                if not isinstance(atr_v,(int,float)) or not isinstance(price,(int,float)) or atr_v<=0 or price<=0:
                                    new_tp_levels_alt.append(row.get('tp_level')); new_sl_levels_alt.append(row.get('sl_level')); continue
                                tp_m=row.get('grid_tp_mult_chosen'); sl_m=row.get('grid_sl_mult_chosen')
                                if row.get('pred_dir')==1:
                                    new_tp_levels_alt.append(price + tp_m*atr_v)
                                    new_sl_levels_alt.append(price - sl_m*atr_v)
                                elif row.get('pred_dir')==2:
                                    new_tp_levels_alt.append(price - tp_m*atr_v)
                                    new_sl_levels_alt.append(price + sl_m*atr_v)
                                else:
                                    new_tp_levels_alt.append(None); new_sl_levels_alt.append(None)
                            temp_df['tp_level']=new_tp_levels_alt
                            temp_df['sl_level']=new_sl_levels_alt
                        # Threshold scan (ограниченный набор)
                        alt_results=[]
                        for thr in sensitivity_thresholds:
                            bt_alt_thr = detailed_backtest(
                                temp_df,
                                y_valid_pred_full,
                                y_valid_proba_full,
                                cfg,
                                confidence_threshold=thr,
                                p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
                            )
                            m_alt = bt_alt_thr.get('metrics',{})
                            alt_results.append({
                                'threshold': thr,
                                'total_trades': m_alt.get('total_trades',0),
                                'profit_factor': m_alt.get('profit_factor',0.0),
                                'win_rate': m_alt.get('win_rate',0.0)
                            })
                        # Выбор лучшего: сначала фильтр по минимальному числу сделок
                        valid_alt = [r for r in alt_results if r['total_trades']>0]
                        best_alt_entry=None
                        if valid_alt:
                            cand = [r for r in valid_alt if r['total_trades']>=min_trades_req]
                            if cand:
                                # Жёсткий фильтр PF>=1.5 среди eligible
                                pf_cand = [r for r in cand if r['profit_factor']>=1.5]
                                best_alt_entry = max((pf_cand or cand), key=lambda x:x['profit_factor'])
                            else:
                                # Недостаточно сделок: берём max PF, но помечаем как ineligible
                                best_alt_entry = max(valid_alt,key=lambda x:x['profit_factor'])
                                best_alt_entry = dict(best_alt_entry)
                                best_alt_entry['ineligible_due_to_trades'] = True
                        # Опциональный быстрый bootstrap PF для best_alt_entry
                        boot_ci = None
                        if best_alt_entry and getattr(args,'sensitivity_bootstrap_pf',0)>0:
                            try:
                                thr_sel = best_alt_entry['threshold']
                                bt_for_boot = detailed_backtest(
                                    temp_df,
                                    y_valid_pred_full,
                                    y_valid_proba_full,
                                    cfg,
                                    confidence_threshold=thr_sel,
                                    p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
                                )
                                trades_df_bt = bt_for_boot.get('trades') if isinstance(bt_for_boot, dict) else None
                                if trades_df_bt is not None and len(trades_df_bt)>1:
                                    import random as _rnd
                                    pnl_col = 'pnl_pct' if 'pnl_pct' in trades_df_bt.columns else ('return_pct' if 'return_pct' in trades_df_bt.columns else None)
                                    if pnl_col:
                                        pnl_vals = trades_df_bt[pnl_col].astype(float).values
                                        n_boot_sens = int(getattr(args,'sensitivity_bootstrap_pf'))
                                        def _pf(arr):
                                            wins = arr[arr > 0]
                                            losses = -arr[arr < 0]
                                            gp = wins.sum(); gl = losses.sum()
                                            if gl <= 0:
                                                return float('inf') if gp > 0 else 0.0
                                            return float(gp / gl)
                                        m = len(pnl_vals)
                                        pfs = []
                                        for _ in range(n_boot_sens):
                                            idx = _rnd.choices(range(m), k=m)
                                            sample = pnl_vals[idx]
                                            pf_v = _pf(sample)
                                            if pf_v == float('inf'):
                                                pf_v = 50.0
                                            pfs.append(pf_v)
                                        pfs_sorted = sorted(pfs)
                                        def _pct(q):
                                            if not pfs_sorted:
                                                return None
                                            pos = int(q * (len(pfs_sorted)-1))
                                            return float(pfs_sorted[pos])
                                        boot_ci = {
                                            'n': n_boot_sens,
                                            'median': _pct(0.5),
                                            'p05': _pct(0.05),
                                            'p95': _pct(0.95),
                                            'pct_lt_1': float(sum(1 for x in pfs if x < 1.0) / len(pfs))
                                        }
                            except Exception as _boot_sens_err:
                                boot_ci = {'error': str(_boot_sens_err)[:200]}
                        pf_at_main = None
                        main_thr = None
                        if 'results' in locals() and results:
                            # текущий optimal threshold будет определён позже, но мы можем взять 0.79 или base_pct's chosen
                            pass
                        # Сохраняем
                        sensitivity_results.append({
                            'percentile': alt_pct,
                            'tp_mult_cap': tp_mult_cap_alt,
                            'pct_clipped': pct_clipped_alt,
                            'suggested_mfe_pct': suggested_mfe_pct_alt,
                            'samples': len(mfe_samples_alt),
                            'atr_price_ratio_median': atr_price_ratio_med_alt,
                            'best_entry': best_alt_entry,
                            'best_entry_bootstrap_pf': boot_ci,
                            'thresholds_tested': alt_results
                        })
                        # Заполняем сводку по перцентилю
                        eligible_flag = bool(best_alt_entry and not best_alt_entry.get('ineligible_due_to_trades'))
                        percentile_summary.append({
                            'percentile': alt_pct,
                            'tp_mult_cap': tp_mult_cap_alt,
                            'pct_clipped': pct_clipped_alt,
                            'eligible': eligible_flag,
                            'best_pf': (best_alt_entry or {}).get('profit_factor'),
                            'best_threshold': (best_alt_entry or {}).get('threshold'),
                            'best_trades': (best_alt_entry or {}).get('total_trades'),
                            'boot_pf_median': (boot_ci or {}).get('median') if isinstance(boot_ci, dict) else None
                        })
                    except Exception as alt_err:
                        sensitivity_results.append({
                            'percentile': alt_pct,
                            'status': 'error',
                            'error': str(alt_err)[:200]
                        })
                # Прилепим сводку
                if 'dynamic_tp_info' in locals() and isinstance(dynamic_tp_info, dict):
                    dynamic_tp_info['percentile_summary'] = percentile_summary
            else:
                print("ℹ️ SENSITIVITY: список альтернатив пуст или невалиден")
        # Прикрепляем результаты чувствительности к dynamic_tp_info чтобы они попали в meta
        if sensitivity_results and 'dynamic_tp_info' in locals() and isinstance(dynamic_tp_info, dict):
            dynamic_tp_info['sensitivity'] = sensitivity_results
            # Auto-select recommendation/apply
            if getattr(args,'dynamic_tpp_auto_select',None):
                pass
            mode_auto = getattr(args,'dynamic_tp_auto_select','')
            if mode_auto:
                # Алгоритм выбора: среди eligible берём максимальный boot_pf_median если есть, иначе best_pf; при равенстве — большее число сделок
                elig = [s for s in percentile_summary if s.get('eligible')]
                def _score(item):
                    boot = item.get('boot_pf_median')
                    pf = item.get('best_pf')
                    trades = item.get('best_trades') or 0
                    base = boot if isinstance(boot,(int,float)) else (pf if isinstance(pf,(int,float)) else -1)
                    return (base, trades)
                chosen = max(elig, key=_score) if elig else (max(percentile_summary, key=_score) if percentile_summary else None)
                dynamic_tp_info['auto_selection'] = {
                    'mode': mode_auto,
                    'chosen_percentile': (chosen or {}).get('percentile'),
                    'reason': 'highest_bootstrap_median_then_pf_then_trades_among_eligible',
                    'eligible_count': len(elig),
                    'note': 'apply mode not mutating current run; set needs_rerun: true to re-run with chosen percentile'
                }
                if mode_auto == 'apply' and dynamic_tp_info['auto_selection']['chosen_percentile'] is not None:
                    dynamic_tp_info['auto_selection']['needs_rerun'] = True
        thresholds = [round(x, 2) for x in np.arange(0.50, 0.901, 0.01)]
        results = []
        threshold_summary_rows = []
        print("\n🔁 Threshold scan (full detailed_backtest alignment)...")
        for threshold in thresholds:
            bt = detailed_backtest(
                valid_df,
                y_valid_pred_full,
                y_valid_proba_full,
                cfg,
                confidence_threshold=threshold,
                p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
            )
            m = bt['metrics']
            trades_df_thr = bt.get('trades')
            if cfg.debug_save_all_thresholds:
                trades_path_dir = getattr(cfg, 'debug_trades_prefix', None) or cfg.out_dir
                os.makedirs(trades_path_dir, exist_ok=True)
                base_sym = cfg.symbol.replace('/','')
                thr_str = f"{threshold:.2f}".replace('.','p')
                fname = f"trades_{base_sym}_{cfg.tf_train}_thr{thr_str}.csv"
                try:
                    trades_df_thr.to_csv(os.path.join(trades_path_dir, fname), index=False)
                except Exception as e:
                    print(f"⚠️ Не удалось сохранить trades для threshold {threshold}: {e}")
            trades_by_exit = m.get('trades_by_exit_reason', {})
            tp_c = trades_by_exit.get('take_profit', 0)
            sl_c = trades_by_exit.get('stop_loss', 0)
            te_c = trades_by_exit.get('time_exit', 0)
            total_trades_bt = m.get('total_trades', 0)
            prefilter = m.get('total_signals_generated', total_trades_bt)
            accept_rate = m.get('acceptance_rate', 0)
            win_rate_bt = m.get('win_rate', 0)
            profit_factor_bt = m.get('profit_factor', 0)
            ev_sum = m.get('ev_calibrated_sum', 0)
            ev_mean = ev_sum / total_trades_bt if total_trades_bt > 0 else 0.0
            p_hit_avg = win_rate_bt  # proxy; можно позже считать средний p_hit из трейдов
            score_bt = win_rate_bt * min(profit_factor_bt if np.isfinite(profit_factor_bt) else 10.0, 10.0) * total_trades_bt / 1000 if total_trades_bt > 0 else 0.0
            # Build trade list with selected excursion metrics for dynamic TP suggestion
            trades_list_compact = []
            if trades_df_thr is not None and len(trades_df_thr):
                # Source column names in trades CSV
                # entry_time, exit_time, side, return_pct, exit_reason, realized_rr, mfe_pct, mae_pct
                for _, row in trades_df_thr.iterrows():
                    rec = {
                        'entry_time': row.get('entry_time'),
                        'exit_time': row.get('exit_time'),
                        'direction': row.get('side'),
                        'pnl_pct': row.get('return_pct'),
                        'exit_type': row.get('exit_reason'),
                        'rr': row.get('realized_rr', row.get('net_rr_chosen')),
                        'mfe_pct': row.get('mfe_pct'),
                        'mae_pct': row.get('mae_pct')
                    }
                    trades_list_compact.append(rec)
            entry = {
                'threshold': threshold,
                'total_trades': total_trades_bt,
                'prefilter_trades': prefilter,
                'accepted_trades': total_trades_bt,
                'accept_rate': accept_rate,
                'win_rate': win_rate_bt,
                'profit_factor': profit_factor_bt,
                'score': score_bt,
                'tp_count': tp_c,
                'sl_count': sl_c,
                'time_exit_count': te_c,
                'ev_sum': ev_sum,
                'ev_mean': ev_mean,
                'p_hit_avg': p_hit_avg,
                'max_drawdown_pct': m.get('max_drawdown_pct', 0),
                'total_return_pct': m.get('total_return_pct', 0),
                'filter_reason_counts': m.get('filtered_reason_counts', {}),
                'trades': trades_list_compact
            }
            results.append(entry)
            threshold_summary_rows.append({
                'threshold': threshold,
                'trades': total_trades_bt,
                'prefilter': prefilter,
                'accepted': total_trades_bt,
                'win_rate': win_rate_bt,
                'p_hit_avg': p_hit_avg,
                'ev_sum': ev_sum,
                'ev_mean': ev_mean,
                'accept_rate': accept_rate
            })
            print(f"Threshold {threshold:.2f} | Trades:{total_trades_bt:4d} | Prefilter:{prefilter:4d} | AR:{accept_rate:.2%} | WR:{win_rate_bt:.1%} | PF:{profit_factor_bt:.2f} | TE:{te_c} | EV_mean:{ev_mean:.5f}")

        # threshold_summary_rows позже попадут в meta['threshold_table']
    
    # Найти оптимальный порог с улучшенной логикой
    # Проверяем что results определён (не в режиме 100% обучения)
    if test_size_frac == 0:
        # В режиме 100% обучения пропускаем анализ результатов
        print(f"\n💡 Используется дефолтный порог: {optimal_threshold:.2f}")
        best_result = None
    else:
        valid_results = [r for r in results if r['total_trades'] > 0]
        if valid_results:
            max_pf_observed = max(r.get('profit_factor', 0) for r in valid_results)
            # EARLY GUARD: если ВСЕ протестированные пороги убыточны (PF < 1.0) — помечаем как unprofitable и берём минимальный threshold
            if max_pf_observed < 1.0:
                best_result = min(valid_results, key=lambda x: x['threshold'])  # самый низкий порог даёт максимум сделок для доп.анализа
                optimal_threshold = best_result['threshold']
                selection_reason = "все протестированные пороги убыточны (PF<1.0) — fallback к минимальному threshold"
                print("\n⚠️ МОДЕЛЬ УБЫТОЧНА НА ВСЕХ ПОРОГАХ (PF < 1.0). Рекомендуется НЕ ИСПОЛЬЗОВАТЬ её в торговле до улучшений.")
                print(f"📉 Максимальный PF в скане: {max_pf_observed:.3f}")
            else:
                # Приоритет 1: Результаты с PF >= 1.5 и достаточным количеством сделок (синхронизировано с period selection)
                profitable_results = [r for r in valid_results if r['profit_factor'] >= 1.5 and r['total_trades'] >= 30]
                if profitable_results:
                    best_result = max(profitable_results, key=lambda x: x['profit_factor'])
                    selection_reason = f"выбран среди прибыльных (PF≥1.5, ≥30 сделок) по максимальному PF"
                else:
                    # Приоритет 2: Любые результаты с PF >= 1.2
                    moderately_profitable = [r for r in valid_results if r['profit_factor'] >= 1.2]
                    if moderately_profitable:
                        best_result = max(moderately_profitable, key=lambda x: x['profit_factor'])
                        selection_reason = f"выбран среди умеренно прибыльных (PF≥1.2) по максимальному PF"
                    else:
                        # Новый Приоритет 2.5: Если есть пороги с PF > 1.0, выбираем лучший PF с достаточным числом сделок
                        try:
                            min_trades_pf_gt1 = int(getattr(args, 'plateau_min_trades', 30))
                        except Exception:
                            min_trades_pf_gt1 = 30
                        pf_gt1 = [r for r in valid_results if r.get('profit_factor', 0) > 1.0 and r.get('total_trades', 0) >= min_trades_pf_gt1]
                        if pf_gt1:
                            # Сначала максимальный PF, при равенстве — большее число сделок
                            best_result = max(pf_gt1, key=lambda x: (x['profit_factor'], x['total_trades']))
                            selection_reason = f"выбран среди порогов с PF>1.0 и ≥{min_trades_pf_gt1} сделок по максимальному PF"
                        else:
                            # Приоритет 3: Результаты с достаточным количеством сделок
                            substantial_results = [r for r in valid_results if r['total_trades'] >= 30]
                            if substantial_results:
                                best_result = max(substantial_results, key=lambda x: x['score'])
                                selection_reason = "выбран из результатов с ≥30 сделок по максимальному Score"
                            else:
                                # Последний приоритет: лучший по Score
                                best_result = max(valid_results, key=lambda x: x['score'])
                                selection_reason = "выбран из всех результатов по максимальному Score (мало сделок)"
            optimal_threshold = best_result['threshold']
            # Debug: сверим PF из results по этому порогу против поиска в списке
            debug_pf_list = {r['threshold']: r['profit_factor'] for r in valid_results}
            debug_wr_list = {r['threshold']: r['win_rate'] for r in valid_results}
            sel_pf = debug_pf_list.get(optimal_threshold)
            sel_wr = debug_wr_list.get(optimal_threshold)
            print(f"\n🎯 ОПТИМАЛЬНЫЙ ПОРОГ: {optimal_threshold:.2f} ({selection_reason})")
            print(f"🔍 DEBUG CHECK: PF(sel)={sel_pf:.4f} WR(sel)={sel_wr:.4f}; Top3 PF thresholds: " + 
                ', '.join([f"{thr:.2f}:{pf:.2f}" for thr,pf in sorted(debug_pf_list.items(), key=lambda kv: kv[1], reverse=True)[:3]]))
            time_exits = best_result.get('time_exit_count', 0)
            print(f"Сделок: {best_result['total_trades']}")
            print(f"Win Rate: {best_result['win_rate']:.1%}")
            print(f"Profit Factor: {best_result['profit_factor']:.2f}")
            print(f"Score: {best_result['score']:.3f}")
            print(f"TP:SL:TE: {best_result['tp_count']}:{best_result['sl_count']}:{time_exits}")
            
            # Проверяем согласованность
            total_exits = best_result['tp_count'] + best_result['sl_count'] + time_exits
            if total_exits != best_result['total_trades']:
                print(f"⚠️ Несогласованность: {total_exits} выходов vs {best_result['total_trades']} сделок")
        else:
            # Fallback к адаптивному порогу если оптимизация не дала результатов
            optimal_threshold = adaptive_params['confidence_threshold']
            print(f"\n⚠️ Оптимизация не дала результатов, используем адаптивный порог: {optimal_threshold:.2f}")
    
    # === Детальный бэктест с оптимальным порогом ===
    # Теперь threshold_optimization уже основан на detailed_backtest (full logic per threshold)
    if test_size_frac == 0:
        print(f"\n📊 Детальный бэктест недоступен при 100% обучении")
        print(f"💡 Используйте модель в реальной торговле с порогом: {optimal_threshold:.2f}")
    else:
        print(f"\n📊 Используем результаты детального бэктеста для оптимального порога: {optimal_threshold:.2f}")
        # Выполним полный detailed_backtest только для выбранного optimal_threshold на validation части
        X_valid_full = valid_df[feature_cols].values
        y_valid_pred_full = model.predict(X_valid_full)
        y_valid_proba_full = model.predict_proba(X_valid_full)
        backtest_results = detailed_backtest(
            valid_df,
            y_valid_pred_full,
            y_valid_proba_full,
            cfg,
            confidence_threshold=optimal_threshold,
            p_hit_map=p_hit_map if 'p_hit_map' in locals() else None
        )
        bt_metrics = backtest_results['metrics']
    print(f"Total Trades: {bt_metrics['total_trades']}")
    
    if test_size_frac > 0 and backtest_results['metrics']['total_trades'] > 0:
        bt_metrics = backtest_results['metrics']
        print(f"Total Trades: {bt_metrics['total_trades']}")
        print(f"Win Rate: {bt_metrics['win_rate']*100:.2f}%")
        print(f"Total Return: {bt_metrics['total_return_pct']:.2f}%")
        print(f"Profit Factor: {bt_metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {bt_metrics['max_drawdown_pct']:.2f}%")
        print(f"Avg Win: {bt_metrics['avg_win_pct']:.2f}%")
        print(f"Avg Loss: {bt_metrics['avg_loss_pct']:.2f}%")
        print(f"Exit Reasons: {bt_metrics['trades_by_exit_reason']}")
        
        # Дополнительный анализ
        total_signals = bt_metrics.get('total_signals_generated', 0)
        filtered_signals = bt_metrics.get('signals_after_filtering', 0)
        if total_signals > 0:
            filter_efficiency = filtered_signals / total_signals * 100
            print(f"Filter Efficiency: {filter_efficiency:.1f}% ({filtered_signals}/{total_signals})")
        
        # Анализ соотношения TP/SL
        tp_count = bt_metrics['trades_by_exit_reason'].get('take_profit', 0)
        sl_count = bt_metrics['trades_by_exit_reason'].get('stop_loss', 0)
        if sl_count > 0:
            tp_sl_ratio = tp_count / sl_count
            print(f"TP:SL Ratio: {tp_sl_ratio:.2f}:1")
        
        # Сохранение детальных результатов
        backtest_results['trades'].to_csv(os.path.join(cfg.out_dir, f"trades_{base}.csv"), index=False)
        
        # График equity curve с датами
        if not args.silent:
            plt.figure(figsize=(15, 8))
            
            # Создаем временную ось для графика
            equity_dates = pd.date_range(start=test_start, periods=len(backtest_results['equity_curve']), freq='30min')
            
            plt.plot(equity_dates, backtest_results['equity_curve'], 'b-', linewidth=2)
            # Форматируем заголовок в зависимости от периода
            period_days = (test_end - test_start).days
            if period_days > 200:
                title_format = f'Детальный Бэктест - Кривая Эквити\nПериод: {test_start.strftime("%b %Y")} - {test_end.strftime("%b %Y")} ({period_days} дней)'
            else:
                title_format = f'Детальный Бэктест - Кривая Эквити\nПериод: {test_start.strftime("%Y-%m-%d")} - {test_end.strftime("%Y-%m-%d")} ({period_days} дней)'
            
            plt.title(title_format, fontsize=14, pad=20)
            plt.ylabel('Баланс ($)', fontsize=12)
            plt.xlabel('Дата', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Форматирование оси времени
            from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
            
            # Определяем интервал меток в зависимости от периода
            period_days = (test_end - test_start).days
            
            if period_days > 200:  # Для больших периодов - месяц и год
                plt.gca().xaxis.set_major_locator(MonthLocator())
                plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))
            elif period_days > 60:  # Для средних периодов - каждые 7 дней
                plt.gca().xaxis.set_major_locator(DayLocator(interval=7))
                plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))
            else:  # Для коротких периодов - каждые 2-3 дня
                plt.gca().xaxis.set_major_locator(DayLocator(interval=2))
                plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))
            
            plt.gca().tick_params(axis='x', rotation=45)
            
            # Добавляем статистику на график
            final_balance = backtest_results['equity_curve'][-1]
            total_return = (final_balance - 10000) / 10000 * 100
            plt.text(0.02, 0.98, f'Общий доход: {total_return:.2f}%\nСделок: {bt_metrics["total_trades"]}\nWin Rate: {bt_metrics["win_rate"]:.1%}',
                    transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        else:
            print("🔇 График equity curve отключен (режим --silent)")
    
    # --- Bootstrap PF CI (optional) ---
    if getattr(args, 'bootstrap_pf', 0) and test_size_frac > 0 and backtest_results['metrics']['total_trades'] > 0:
        n_boot = int(args.bootstrap_pf)
        if n_boot > 0:
            try:
                trades_df_bt = backtest_results.get('trades')
                if trades_df_bt is not None and len(trades_df_bt) > 1:
                    import random as _rnd
                    # Prefer 'pnl_pct' else 'return_pct'
                    pnl_col = 'pnl_pct' if 'pnl_pct' in trades_df_bt.columns else ('return_pct' if 'return_pct' in trades_df_bt.columns else None)
                    if pnl_col:
                        pnl_vals = trades_df_bt[pnl_col].astype(float).values
                        point_pf = bt_metrics.get('profit_factor')
                        def _pf(arr):
                            wins = arr[arr > 0]
                            losses = -arr[arr < 0]
                            gp = wins.sum()
                            gl = losses.sum()
                            if gl <= 0:
                                return float('inf') if gp > 0 else 0.0
                            return float(gp / gl)
                        # Pre-sample indexes for speed
                        m = len(pnl_vals)
                        pfs = []
                        for _ in range(n_boot):
                            idx = _rnd.choices(range(m), k=m)
                            sample = pnl_vals[idx]
                            pf_v = _pf(sample)
                            if pf_v == float('inf'):
                                pf_v = 50.0  # cap extreme
                            pfs.append(pf_v)
                        pfs_sorted = sorted(pfs)
                        def _pct(q):
                            if not pfs_sorted:
                                return None
                            pos = int(q * (len(pfs_sorted)-1))
                            return float(pfs_sorted[pos])
                        pf_median = _pct(0.5)
                        pf_p05 = _pct(0.05)
                        pf_p95 = _pct(0.95)
                        pct_lt_1 = float(sum(1 for x in pfs if x < 1.0) / len(pfs))
                        half_point = point_pf * 0.5 if (point_pf and point_pf not in (0, float('inf'))) else None
                        pct_lt_half_point = float(sum(1 for x in pfs if half_point is not None and x < half_point) / len(pfs)) if half_point is not None else None
                        bt_metrics['bootstrap_pf'] = {
                            'n': n_boot,
                            'point_pf': point_pf,
                            'median': pf_median,
                            'p05': pf_p05,
                            'p95': pf_p95,
                            'pct_lt_1': pct_lt_1,
                            'pct_lt_point_half': pct_lt_half_point
                        }
                        print(f"📊 Bootstrap PF ({n_boot}): median={pf_median:.2f} p05={pf_p05:.2f} p95={pf_p95:.2f} pct<1={pct_lt_1:.1%}")
                    else:
                        print("⚠️ Bootstrap PF: колонка с pnl не найдена (pnl_pct/return_pct)")
                else:
                    print("⚠️ Bootstrap PF: недостаточно сделок")
            except Exception as boot_err:
                print(f"⚠️ Bootstrap PF error: {boot_err}")

    # === ВСЕГДА сохраняем результаты бэктеста в метаданные ===
    # Обновляем метаданные с результатами бэктеста (даже если сделок нет)
    meta_file_path = os.path.join(cfg.out_dir, f"meta_{base}.json")
    with open(meta_file_path, "r") as f:
        meta = json.load(f)
    
    # Добавляем результаты бэктеста в метаданные
    # ПРИМЕЧАНИЕ: threshold_optimization теперь также использует full detailed_backtest для каждого порога
    
    # Извлекаем детальную информацию о сделках
    trades_by_exit = bt_metrics.get('trades_by_exit_reason', {})
    tp_count = trades_by_exit.get('take_profit', 0)
    sl_count = trades_by_exit.get('stop_loss', 0)
    time_exit_count = trades_by_exit.get('time_exit', 0)
    
    meta['backtest_results'] = {
        'win_rate': bt_metrics.get('win_rate', 0),
        'profit_factor': bt_metrics.get('profit_factor', 0),
        'total_trades': bt_metrics.get('total_trades', 0),
        'total_return_pct': bt_metrics.get('total_return_pct', 0),
        # Удобные поля для быстрой интеграции
        'total_pnl_usd': bt_metrics.get('total_pnl_usd'),
        'starting_equity': bt_metrics.get('starting_equity'),
        'max_drawdown_pct': bt_metrics.get('max_drawdown_pct', 0),
        'confidence_threshold': optimal_threshold,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'time_exit_count': time_exit_count,
        'note': 'detailed_backtest with full trading logic'
    }
    # Добавляем расширенный breakdown по исходам если есть
    if 'exit_pnl_breakdown' in bt_metrics:
        meta['backtest_results']['exit_pnl_breakdown'] = bt_metrics['exit_pnl_breakdown']
    if 'exit_pnl_breakdown_error' in bt_metrics:
        meta['backtest_results']['exit_pnl_breakdown_error'] = bt_metrics['exit_pnl_breakdown_error']
    # Добавляем счётчики фильтров из детального бэктеста, если есть
    if 'filtered_reason_counts' in bt_metrics:
        meta['backtest_results']['filter_reason_counts'] = bt_metrics.get('filtered_reason_counts')
    if 'acceptance_rate' in bt_metrics:
        meta['backtest_results']['acceptance_rate'] = bt_metrics.get('acceptance_rate')
    if 'ev_calibrated_sum' in bt_metrics:
        meta['backtest_results']['ev_calibrated_sum'] = bt_metrics.get('ev_calibrated_sum')
    if 'ev_naive_sum' in bt_metrics:
        meta['backtest_results']['ev_naive_sum'] = bt_metrics.get('ev_naive_sum')
    # MFE/MAE агрегаты
    for k in [
        'time_exit_mfe_mean_pct','time_exit_mae_mean_pct','time_exit_mfe_median_pct','time_exit_mae_median_pct',
        'time_exit_avg_unrealized_left_pct','overall_mfe_mean_pct','overall_mae_mean_pct',
        'overall_avg_unrealized_left_pct',
        'overall_efficiency_ratio_mean','overall_efficiency_ratio_median',
        'overall_efficiency_ratio_mean_clipped','overall_efficiency_ratio_median_clipped',
        'time_exit_efficiency_ratio_mean','time_exit_efficiency_ratio_median',
        'time_exit_efficiency_ratio_mean_clipped','time_exit_efficiency_ratio_median_clipped'
    ]:
        if k in bt_metrics:
            meta['backtest_results'][k] = bt_metrics[k]
    if 'p_hit_map' not in meta and test_size_frac > 0 and 'p_hit_map' in locals():
        meta['p_hit_map'] = p_hit_map

    # mfe_by_exit: агрегируем по типу выхода в оптимальном threshold (после выбора optimal_threshold)
    if 'results' in locals() and results:
        opt_thr_res_loc = next((r for r in results if r.get('threshold') == optimal_threshold), None)
        if opt_thr_res_loc and 'trades' in opt_thr_res_loc:
            trades_opt = opt_thr_res_loc['trades']
            # Normalize exit types from raw trades to canonical keys
            # Raw exit_type values are typically: 'take_profit', 'stop_loss', 'time_exit'
            normalize_map = {
                'take_profit': 'TP',
                'TP': 'TP',
                'stop_loss': 'SL',
                'SL': 'SL',
                'time_exit': 'time_exit'
            }
            grouped = {'TP': [], 'SL': [], 'time_exit': []}
            for tr in trades_opt:
                ex_raw = tr.get('exit_type')
                key = normalize_map.get(ex_raw)
                if key and isinstance(tr.get('mfe_pct'), (int, float)):
                    grouped[key].append(tr.get('mfe_pct'))
            mfe_by_exit = {}
            for key, lst in grouped.items():
                if lst:
                    arr = np.array(lst)
                    mfe_by_exit[key] = {
                        'count': len(lst),
                        'mean': float(np.mean(arr)),
                        'median': float(np.median(arr))
                    }
                else:
                    mfe_by_exit[key] = {'count': 0, 'mean': None, 'median': None}
            meta['backtest_results']['mfe_by_exit'] = mfe_by_exit

    # === Dynamic TP Meta Registration ===
    if 'dynamic_tp_info' in locals() and dynamic_tp_info is not None:
        meta['dynamic_tp'] = dynamic_tp_info
        # Встраиваем baseline сравнение (если выполнялось)
        if 'baseline_scan_results' in locals() and baseline_scan_results is not None and isinstance(baseline_scan_results, list):
            # Найти метрику baseline для выбранного clipped порога
            clipped_thr = meta['backtest_results']['confidence_threshold'] if 'backtest_results' in meta else None
            baseline_for_clipped = next((r for r in baseline_scan_results if abs(r['threshold'] - clipped_thr) < 1e-9), None)
            pf_clipped = meta['backtest_results'].get('profit_factor') if 'backtest_results' in meta else None
            pf_baseline_same_thr = baseline_for_clipped.get('profit_factor') if baseline_for_clipped else None
            pf_delta = None
            if pf_clipped is not None and pf_baseline_same_thr is not None:
                try:
                    pf_delta = pf_clipped - pf_baseline_same_thr
                except Exception:
                    pf_delta = None
            baseline_opt_payload = None
            if 'baseline_optimal' in locals() and baseline_optimal:
                baseline_opt_payload = {
                    'threshold': baseline_optimal.get('threshold'),
                    'profit_factor': baseline_optimal.get('profit_factor'),
                    'win_rate': baseline_optimal.get('win_rate'),
                    'total_trades': baseline_optimal.get('total_trades')
                }
            # Компактный список результатов baseline (не тащим все поля чтобы не раздувать meta)
            baseline_compact = [
                {k: r[k] for k in ('threshold','profit_factor','win_rate','total_trades')}
                for r in baseline_scan_results
            ] if isinstance(baseline_scan_results, list) else baseline_scan_results
            meta['dynamic_tp']['baseline_no_cap'] = {
                'available': True,
                'baseline_optimal': baseline_opt_payload,
                'pf_at_clipped_threshold': pf_baseline_same_thr,
                'pf_with_cap_at_threshold': pf_clipped,
                'pf_delta_cap_minus_baseline': pf_delta,
                'results_compact': baseline_compact[:50]  # ограничим размер
            }
        elif getattr(args, 'baseline_no_cap', False):
            # Флаг был запрошен, но baseline не выполнен
            if 'baseline_no_cap' not in meta['dynamic_tp']:
                meta['dynamic_tp']['baseline_no_cap'] = {
                    'available': False,
                    'reason': 'requested_but_not_applicable'
                }
    elif getattr(cfg, 'dynamic_tp_percentile', None) is not None and 'dynamic_tp' not in meta:
        # Fallback: keep legacy suggest-only block if earlier logic failed to populate
        meta['dynamic_tp'] = {
            'mode': 'suggest_only',
            'requested_percentile': getattr(cfg, 'dynamic_tp_percentile'),
            'samples': 0,
            'reason': 'dynamic_tp_info_missing'
        }
    
    # Добавляем полные результаты оптимизации порогов + мини-таблица
    if 'results' in locals() and results:
        # Найдём запись оптимального порога и добавим ev_mean если есть
        opt_entry = next((r for r in results if r.get('threshold') == meta['backtest_results']['confidence_threshold']), None)
        ev_mean_opt = opt_entry.get('ev_mean') if opt_entry else None
        # Будущий блок threshold_optimization соберем в отдельный dict чтобы избежать KeyError
        # Дополнительные агрегаты для меты
        _max_pf_global = max((r.get('profit_factor', 0) for r in results), default=0)
        _min_pf_global = min((r.get('profit_factor', 0) for r in results), default=0)
        _avg_pf_global = float(np.mean([r.get('profit_factor', 0) for r in results])) if results else 0.0
        thr_opt_dict = {
            'tested_thresholds': [r['threshold'] for r in results],
            'all_results': results,
            'optimal_threshold': optimal_threshold,
            'selection_reason': selection_reason if 'selection_reason' in locals() else "Адаптивный порог",
            'optimization_date': datetime.now().isoformat(),
            'note': 'full detailed_backtest scan (aligned, v1)',
            'scan_mode': 'full_bt_v1_aligned',
            'ev_mean_optimal': ev_mean_opt,
            'max_profit_factor_tested': _max_pf_global,
            'min_profit_factor_tested': _min_pf_global,
            'avg_profit_factor_tested': _avg_pf_global,
            'unprofitable_flag': bool(_max_pf_global < 1.0)
        }
        # Суммарные причины фильтрации по оптимальному порогу
        if opt_entry and opt_entry.get('filter_reason_counts'):
            thr_opt_dict['optimal_filter_reason_counts'] = opt_entry['filter_reason_counts']
        # Построим распределения метрик для оптимального порога (prefilter subset)
        try:
            if test_size_frac > 0 and 'valid_df' in locals() and opt_entry:
                thr = meta['backtest_results']['confidence_threshold']
                pre_mask = (valid_df['pred_dir'].isin([1,2])) & (valid_df['confidence'] >= thr)
                dist_df = valid_df[pre_mask].copy()
                distributions = {}
                if not dist_df.empty:
                    # Получим стоимость сделки повторно (round_trip_cost_pct)
                    try:
                        from risk_metrics import load_risk_config
                        _rc_tmp = load_risk_config()
                        _ovr_tmp = (_rc_tmp.get('symbols') or {}).get(cfg.symbol, {})
                        default_cost_pct = _ovr_tmp.get('round_trip_cost_pct', _rc_tmp.get('default_round_trip_cost_pct', 0.002))
                    except Exception:
                        default_cost_pct = 0.002
                    rows = []
                    for _, r in dist_df.iterrows():
                        price = r['close']
                        tp_level = r.get('tp_level')
                        sl_level = r.get('sl_level')
                        if tp_level is None or sl_level is None or price <= 0:
                            continue
                        if r['pred_dir'] == 1:
                            target_pct = (tp_level - price)/price
                            stop_pct = (price - sl_level)/price
                        else:
                            target_pct = (price - tp_level)/price
                            stop_pct = (sl_level - price)/price
                        net_target = target_pct - default_cost_pct
                        net_stop = stop_pct + default_cost_pct
                        if net_stop <= 0:
                            continue
                        net_rr = net_target / net_stop if net_stop > 0 else 0
                        p_be_tmp = 1.0 if net_target <= 0 else net_stop / (net_stop + net_target)
                        conf_edge = r['confidence'] - p_be_tmp
                        atr_val = r.get('atr', np.nan)
                        atr_pct_local = (atr_val / price) if (isinstance(atr_val, (int,float)) and atr_val>0) else np.nan
                        rows.append((net_rr, conf_edge, atr_pct_local))
                    if rows:
                        arr = np.array(rows)
                        net_rr_vals = arr[:,0]
                        conf_edge_vals = arr[:,1]
                        atr_pct_vals = arr[:,2]
                        def hist_counts(values, bins):
                            h,_ = np.histogram(values[~np.isnan(values)], bins=bins)
                            return {'bins': list(map(float, bins)), 'counts': list(map(int, h))}
                        distributions['net_rr'] = hist_counts(net_rr_vals, np.array([0,0.5,0.8,1.0,1.2,1.5,2,3,5]))
                        distributions['conf_edge'] = hist_counts(conf_edge_vals, np.array([-1,-0.05,-0.02,0,0.0025,0.005,0.01,0.015,0.02,0.03,0.05,0.1]))
                        valid_atr = atr_pct_vals[~np.isnan(atr_pct_vals)]
                        if valid_atr.size > 5:
                            q_bins = np.unique(np.quantile(valid_atr, [0,0.1,0.25,0.5,0.75,0.9,1]))
                            if len(q_bins) >= 3:
                                distributions['atr_pct'] = hist_counts(valid_atr, q_bins)
                        distributions['sample_size'] = int(len(rows))
                if distributions:
                    thr_opt_dict['distributions'] = distributions
        except Exception as dist_err:
            thr_opt_dict['distribution_error'] = str(dist_err)[:200]
        meta['threshold_optimization'] = thr_opt_dict
        # === Построение Pareto фронта (минимизируем max_drawdown_pct, максимизируем profit_factor) ===
        try:
            df_res = pd.DataFrame(results)
            if {'threshold','profit_factor','max_drawdown_pct','total_trades'}.issubset(df_res.columns):
                df_res = df_res.sort_values(['max_drawdown_pct','profit_factor'], ascending=[True, False])
                pareto = []
                best_pf = -1.0
                for _, row in df_res.iterrows():
                    pf = row['profit_factor']
                    dd = row['max_drawdown_pct']
                    if pf >= best_pf:  # не доминируется предыдущими с меньшим DD и большим PF
                        pareto.append({
                            'threshold': float(row['threshold']),
                            'profit_factor': float(pf),
                            'max_drawdown_pct': float(dd),
                            'total_trades': int(row.get('total_trades', 0))
                        })
                        if pf > best_pf:
                            best_pf = pf
                meta['threshold_optimization']['pareto_front'] = pareto
                # Plateau detection: thresholds with PF within X% of max PF and >= min trades
                try:
                    plateau_margin = float(getattr(args,'plateau_pf_margin',0.05))
                    plateau_min_trades = int(getattr(args,'plateau_min_trades',30))
                    max_pf = float(df_res['profit_factor'].max()) if not df_res.empty else 0.0
                    if max_pf and max_pf>0:
                        cutoff = max_pf * (1.0 - plateau_margin)
                        plateau_rows = df_res[(df_res['profit_factor']>=cutoff) & (df_res['total_trades']>=plateau_min_trades)]
                        meta['threshold_optimization']['pf_plateau'] = {
                            'margin': plateau_margin,
                            'min_trades': plateau_min_trades,
                            'max_pf': float(max_pf),
                            'cutoff': float(cutoff),
                            'thresholds': [
                                {
                                    'threshold': float(r['threshold']),
                                    'profit_factor': float(r['profit_factor']),
                                    'total_trades': int(r['total_trades'])
                                } for _, r in plateau_rows.iterrows()
                            ]
                        }
                except Exception as _pl_err:
                    meta['threshold_optimization']['pf_plateau_error'] = str(_pl_err)[:200]
        except Exception as pf_err:
            meta['threshold_optimization']['pareto_front_error'] = str(pf_err)[:200]
        # Сохраняем мини-таблицу (если была сформирована)
        if 'threshold_summary_rows' in locals():
            meta['threshold_table'] = threshold_summary_rows
        print(f"💾 Сохранены результаты оптимизации {len(results)} порогов (EV + PF)")

    # Если уже есть sensitivity (альтернативные перцентили), добавим сравнение по основному клипу на том же подмножестве порогов
    try:
        if 'dynamic_tp' in meta and isinstance(meta['dynamic_tp'], dict) and 'sensitivity' in meta['dynamic_tp'] and 'threshold_optimization' in meta:
            subset_thr = [0.73,0.75,0.78,0.79,0.80,0.81,0.82]
            main_subset = []
            all_res_map = {r['threshold']: r for r in meta['threshold_optimization'].get('all_results', []) if isinstance(r, dict)}
            for thr in subset_thr:
                r = all_res_map.get(thr)
                if r:
                    main_subset.append({
                        'threshold': thr,
                        'total_trades': r.get('total_trades'),
                        'profit_factor': ('inf' if isinstance(r.get('profit_factor'), (int,float)) and np.isinf(r.get('profit_factor')) else r.get('profit_factor')),
                        'win_rate': r.get('win_rate')
                    })
            if main_subset:
                meta['dynamic_tp']['sensitivity_main_percentile_subset'] = main_subset
    except Exception as _sens_main_err:
        meta.setdefault('dynamic_tp', {}).setdefault('sensitivity_meta_error', str(_sens_main_err)[:200])
    
    # Сохраняем обновленные метаданные
    def _convert(o):
        import pandas as _pd
        import math as _math
        if isinstance(o, (_pd.Timestamp, )):
            return o.isoformat()
        if isinstance(o, (np.integer, )):
            return int(o)
        if isinstance(o, (np.floating, )):
            if _math.isinf(float(o)):
                return 'inf'
            if _math.isnan(float(o)):
                return None
            return float(o)
        if isinstance(o, (np.ndarray, )):
            return o.tolist()
        return o
    # Deep convert trades timestamps if present
    if 'threshold_optimization' in meta and 'all_results' in meta['threshold_optimization']:
        for res in meta['threshold_optimization']['all_results']:
            if isinstance(res, dict) and 'trades' in res and isinstance(res['trades'], list):
                for tr in res['trades']:
                    if isinstance(tr, dict):
                        for k,v in list(tr.items()):
                            try:
                                tr[k] = _convert(v)
                            except Exception:
                                tr[k] = str(v)
    with open(meta_file_path, "w") as f:
        json.dump(meta, f, indent=2, default=_convert)
    
    print(f"✅ Результаты бэктеста сохранены в метаданные: {bt_metrics['total_trades']} сделок")
    print(f"💾 Оптимальный порог {optimal_threshold} сохранен в мета-файле модели")

    # Автосохранение efficient frontier (если запрошено флагом)
    if getattr(args, 'save_frontier', None):
        try:
            frontier_name = args.save_frontier
            if frontier_name in (None, '', 'auto'):
                frontier_name = f"frontier_{base}.png"
            save_path = os.path.join(cfg.out_dir, frontier_name)
            plot_efficient_frontier(meta_file_path, show_plot=not args.silent, save_path=save_path)
            print(f"📈 Frontier saved automatically: {save_path}")
        except Exception as sf_err:
            print(f"⚠️ Frontier auto-save failed: {sf_err}")

    # === ДОПОЛНИТЕЛЬНАЯ ДИАГНОСТИКА (1-6) ===
    # 1. Feasibility net_rr
    try:
        from risk_metrics import load_risk_config as _diag_load_risk
        _rconf = _diag_load_risk()
        # Определение комиссий / стоимости (symbol override учитываем)
        _sym = cfg.symbol.replace('/', '')
        _override = (_rconf.get('symbols') or {}).get(_sym, {})
        _cost = _override.get('round_trip_cost_pct', _rconf.get('default_round_trip_cost_pct', 0.002))
        # Используем фактические tp_pct/sl_pct из конфигурации модели (до мультипликаторов обучения)
        _tp_raw = cfg.tp_pct
        _sl_raw = cfg.sl_pct
        theoretical_net_rr = None
        feasible = None
        min_net_rr_conf = _override.get('min_net_rr', _rconf.get('min_net_rr'))
        if _tp_raw and _sl_raw and _tp_raw > 0 and _sl_raw > 0:
            theoretical_net_rr = (_tp_raw - _cost) / (_sl_raw + _cost) if (_sl_raw + _cost) > 0 else None
            feasible = (theoretical_net_rr is not None and min_net_rr_conf is not None and theoretical_net_rr >= min_net_rr_conf)
        recommended_tp_for_min = None
        if min_net_rr_conf is not None and _sl_raw is not None:
            # tp >= cost + min_rr * (sl + cost)
            recommended_tp_for_min = _cost + min_net_rr_conf * (_sl_raw + _cost)
        recommended_sl_for_min = None
        if min_net_rr_conf is not None and _tp_raw is not None and min_net_rr_conf > 0:
            # sl <= ((tp - cost)/min_rr) - cost
            recommended_sl_for_min = ((_tp_raw - _cost) / min_net_rr_conf) - _cost if (_tp_raw - _cost) > 0 else None
    except Exception as _fe:
        theoretical_net_rr = None
        feasible = None
        recommended_tp_for_min = None
        recommended_sl_for_min = None

    # 2. Calibration metrics (Brier, ECE, MCE) на validation (если есть)
    calibration_block = {}
    try:
        if test_size_frac > 0 and 'valid_df' in locals():
            # Используем только классы LONG/SHORT vs HOLD? Для простоты возьмем макс вероятности предсказанного класса.
            probs = y_valid_proba_full if 'y_valid_proba_full' in locals() else None
            if probs is not None and len(probs) == len(valid_df):
                # Confidence уже есть в valid_df['confidence']
                conf_vals = valid_df['confidence'].values
                # Определим бинарный успех: совпадение направления (pred_dir==actual LONG/SHORT когда не HOLD)
                if 'pred_dir' in valid_df.columns:
                    hits = (valid_df['pred_dir'] == valid_df['target']).astype(int).values
                    # Brier
                    brier = float(np.mean((conf_vals - hits) ** 2))
                    # Бины для ECE
                    bins = np.linspace(0.5, 1.0, 11)
                    bin_ids = np.digitize(conf_vals, bins, right=True)
                    ece = 0.0
                    mce = 0.0
                    total = len(conf_vals)
                    for b in range(1, len(bins)+1):
                        mask = bin_ids == b
                        if not np.any(mask):
                            continue
                        avg_conf = conf_vals[mask].mean()
                        avg_acc = hits[mask].mean()
                        gap = abs(avg_conf - avg_acc)
                        ece += (mask.sum()/total) * gap
                        mce = max(mce, gap)
                    sparse_bins = []
                    for b in range(1, len(bins)+1):
                        if np.sum(bin_ids == b) < 5:
                            if b <= len(bins):
                                sparse_bins.append(float(bins[b-1]))
                    calibration_block = {
                        'brier_score': brier,
                        'ece': float(ece),
                        'mce': float(mce),
                        'sparse_bins': sparse_bins
                    }
    except Exception as _ce:
        calibration_block['error'] = str(_ce)[:120]

    # 3. Class distribution train/test
    class_dist_block = {}
    try:
        if 'train_df' in locals():
            train_counts = train_df['target'].value_counts().to_dict()
            train_total = sum(train_counts.values()) or 1
            train_pct = {str(k): v/train_total for k,v in train_counts.items()}
        else:
            train_counts, train_pct = {}, {}
        if 'test_df' in locals():
            test_counts = test_df['target'].value_counts().to_dict()
            test_total = sum(test_counts.values()) or 1
            test_pct = {str(k): v/test_total for k,v in test_counts.items()}
        else:
            test_counts, test_pct = {}, {}
        class_dist_block = {
            'train': {'counts': train_counts, 'pct': train_pct},
            'test': {'counts': test_counts, 'pct': test_pct}
        }
    except Exception as _de:
        class_dist_block['error'] = str(_de)[:120]

    # 4. CV summary (train vs val logloss already stored in cv?)
    cv_summary_block = {}
    try:
        if 'cv_scores' in meta:  # already saved earlier
            # meta['cv_scores'] = [[train_loss, val_loss], ...]
            arr = np.array(meta['cv_scores'])
            if arr.ndim == 2 and arr.shape[1] == 2:
                train_losses = arr[:,0]
                val_losses = arr[:,1]
                cv_summary_block = {
                    'folds': int(arr.shape[0]),
                    'train_logloss_mean': float(np.mean(train_losses)),
                    'train_logloss_std': float(np.std(train_losses)),
                    'val_logloss_mean': float(np.mean(val_losses)),
                    'val_logloss_std': float(np.std(val_losses)),
                    'overfit_flag': bool(np.mean(val_losses) - np.mean(train_losses) > 0.03)  # простой критерий
                }
    except Exception as _cv:
        cv_summary_block['error'] = str(_cv)[:120]

    # 5. Filter funnel for optimal threshold
    funnel_block = {}
    try:
        if test_size_frac > 0 and 'valid_df' in locals() and 'threshold_optimization' in meta:
            thr = meta['threshold_optimization'].get('optimal_threshold', meta['backtest_results']['confidence_threshold'])
            vd = valid_df.copy()
            # Предполагаем, что valid_df содержит pred_dir, confidence, tp_level, sl_level
            candidates_mask = (vd['pred_dir'].isin([1,2])) & (vd['confidence'] >= thr)
            candidates_df = vd[candidates_mask]
            candidates = len(candidates_df)
            edge_pass_df = []
            prob_pass_df = []
            net_rr_pass_df = []
            conf_edge_pass_df = []
            cal_ev_pass_df = []
            near_miss_net_rr = 0
            near_miss_conf_edge = 0
            if candidates > 0:
                # Reconstruct minimal metrics for funnel (reuse cost/tp/sl)
                for _, r in candidates_df.iterrows():
                    price = r['close']
                    tp_level = r.get('tp_level')
                    sl_level = r.get('sl_level')
                    if tp_level is None or sl_level is None or price <=0:
                        continue
                    if r['pred_dir'] == 1:
                        t_pct = (tp_level - price)/price
                        s_pct = (price - sl_level)/price
                    else:
                        t_pct = (price - tp_level)/price
                        s_pct = (sl_level - price)/price
                    net_t = t_pct - _cost
                    net_s = s_pct + _cost
                    if net_s <= 0:
                        continue
                    p_be_loc = 1.0 if net_t <= 0 else net_s/(net_s + net_t)
                    conf = r['confidence']
                    conf_edge_val = conf - p_be_loc
                    net_rr_val = net_t / net_s if net_s>0 else 0
                    edge_ok = net_t >= _override.get('min_edge_pct', _rconf.get('min_edge_pct',0.005))
                    prob_ok = conf >= p_be_loc
                    min_net_rr_req = _override.get('min_net_rr', _rconf.get('min_net_rr'))
                    net_rr_ok = True if min_net_rr_req is None else net_rr_val >= min_net_rr_req
                    min_conf_edge_bp_req = _override.get('min_conf_edge_bp', _rconf.get('min_conf_edge_bp'))
                    conf_edge_ok = True if min_conf_edge_bp_req is None else conf_edge_val >= min_conf_edge_bp_req
                    # Calibrated EV check skipped here (need p_hit_map mapping) -> assume pass if not configured
                    cal_ev_ok = True
                    if edge_ok:
                        edge_pass_df.append((net_rr_val, conf_edge_val))
                        if prob_ok:
                            prob_pass_df.append((net_rr_val, conf_edge_val))
                            if net_rr_ok:
                                net_rr_pass_df.append((net_rr_val, conf_edge_val))
                                if conf_edge_ok:
                                    conf_edge_pass_df.append((net_rr_val, conf_edge_val))
                                    if cal_ev_ok:
                                        cal_ev_pass_df.append((net_rr_val, conf_edge_val))
                            else:
                                # near-miss net_rr (within 10% of requirement)
                                if min_net_rr_req and (min_net_rr_req*0.9) <= net_rr_val < min_net_rr_req:
                                    near_miss_net_rr +=1
                        else:
                            # near miss conf_edge? We'll evaluate later after net_rr check
                            pass
                    # near miss conf_edge (within 20% of required margin)
                    if conf_edge_val < 0 and min_conf_edge_bp_req and (conf_edge_val + min_conf_edge_bp_req*0.2) >= 0:
                        near_miss_conf_edge +=1
            funnel_block = {
                'threshold': float(thr),
                'candidates': candidates,
                'edge_pass': len(edge_pass_df),
                'prob_pass': len(prob_pass_df),
                'net_rr_pass': len(net_rr_pass_df),
                'conf_edge_pass': len(conf_edge_pass_df),
                'cal_ev_pass': len(cal_ev_pass_df),
                'accepted': len(cal_ev_pass_df),
                'near_miss_net_rr': near_miss_net_rr,
                'near_miss_conf_edge': near_miss_conf_edge
            }
    except Exception as _fu:
        funnel_block['error'] = str(_fu)[:120]

    # 6. Feature importance (top 15)
    feat_imp_block = {}
    try:
        booster = getattr(model, 'get_booster', lambda: None)()
        if booster is not None:
            score_dict = booster.get_score(importance_type='gain')
            if score_dict:
                # map f0 -> feature names
                feat_map = {}
                for idx, col in enumerate(feature_cols):
                    feat_map[f'f{idx}'] = col
                items = []
                for k,v in score_dict.items():
                    name = feat_map.get(k, k)
                    items.append({'feature': name, 'gain': v})
                items.sort(key=lambda x: x['gain'], reverse=True)
                feat_imp_block = {'top': items[:15], 'count': len(items)}
    except Exception as _fi:
        feat_imp_block['error'] = str(_fi)[:120]

    # Aggregate diagnostics
    diagnostics = {
        'feasibility': {
            'tp_pct': cfg.tp_pct,
            'sl_pct': cfg.sl_pct,
            'cost_pct': _cost,
            'min_net_rr': min_net_rr_conf,
            'theoretical_net_rr': theoretical_net_rr,
            'feasible': feasible,
            'recommended_tp_for_min_net_rr': recommended_tp_for_min,
            'recommended_sl_for_min_net_rr': recommended_sl_for_min
        },
        'calibration': calibration_block,
        'class_distribution': class_dist_block,
        'cv_summary': cv_summary_block,
        'filter_funnel_optimal': funnel_block,
        'feature_importance': feat_imp_block,
        'warnings': []
    }
    # Populate warnings
    if feasible is not None and min_net_rr_conf is not None and theoretical_net_rr is not None and theoretical_net_rr < min_net_rr_conf:
        diagnostics['warnings'].append('unreachable_min_net_rr')
    if funnel_block.get('accepted', 0) == 0:
        diagnostics['warnings'].append('no_accepted_trades_optimal')
    if calibration_block.get('ece') and calibration_block.get('ece') > 0.05:
        diagnostics['warnings'].append('poor_calibration_ece')
    if calibration_block.get('sparse_bins') and len(calibration_block['sparse_bins']) > 0:
        diagnostics['warnings'].append('sparse_calibration_bins')
    if cv_summary_block.get('overfit_flag'):
        diagnostics['warnings'].append('overfit_risk')
    # Save diagnostics into meta file (append)
    try:
        with open(meta_file_path, 'r') as _mf:
            _mdata = json.load(_mf)
        _mdata['diagnostics'] = diagnostics
        with open(meta_file_path, 'w') as _mf:
            json.dump(_mdata, _mf, indent=2)
        print('🩺 Diagnostics block appended to meta file.')
    except Exception as _dsave:
        print(f'⚠️ Failed to append diagnostics: {_dsave}')

    # === Создание графика оптимизации ===
    print("\n📊 Создание графика оптимизации...")
    
    if test_size_frac > 0:
        # Создаем график только если был бэктест
        if not args.silent:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Добавляем общий заголовок с информацией о периоде бэктеста
            test_start = test_df['time'].iloc[0]
            test_end = test_df['time'].iloc[-1]
            period_days = (test_end - test_start).days
            if period_days > 200:
                period_text = f'{test_start.strftime("%b %Y")} - {test_end.strftime("%b %Y")} ({period_days} дней)'
            else:
                period_text = f'{test_start.strftime("%Y-%m-%d")} - {test_end.strftime("%Y-%m-%d")} ({period_days} дней)'
        
            fig.suptitle(f'Оптимизация Порога Уверенности - {cfg.symbol} {cfg.tf_train}\n'
                         f'Период бэктеста: {period_text}', fontsize=16, y=0.95)
            
            thresholds_plot = [r['threshold'] for r in results if r['total_trades'] > 0]
            trades_plot = [r['total_trades'] for r in results if r['total_trades'] > 0]
            wr_plot = [r['win_rate']*100 for r in results if r['total_trades'] > 0]
            pf_plot = [min(r['profit_factor'], 10) for r in results if r['total_trades'] > 0]
            scores_plot = [min(r['score'], 10) for r in results if r['total_trades'] > 0]
            
            if thresholds_plot:  # Проверяем, есть ли данные для графика
                # График 1: Количество сделок
                ax1.plot(thresholds_plot, trades_plot, 'b-o', linewidth=2, markersize=6)
                ax1.set_xlabel('Threshold')
                ax1.set_ylabel('Total Trades')
                ax1.set_title('Trades vs Threshold')
                ax1.grid(True, alpha=0.3)
                if 'best_result' in locals():
                    ax1.axvline(best_result['threshold'], color='red', linestyle='--', alpha=0.7, label=f'Best: {best_result["threshold"]}')
                    ax1.legend()
                
                # График 2: Win Rate
                ax2.plot(thresholds_plot, wr_plot, 'g-o', linewidth=2, markersize=6)
                ax2.set_xlabel('Threshold')
                ax2.set_ylabel('Win Rate %')
                ax2.set_title('Win Rate vs Threshold')
                ax2.grid(True, alpha=0.3)
                if 'best_result' in locals():
                    ax2.axvline(best_result['threshold'], color='red', linestyle='--', alpha=0.7, label=f'Best: {best_result["threshold"]}')
                    ax2.legend()
                
                # График 3: Profit Factor
                ax3.plot(thresholds_plot, pf_plot, 'orange', marker='o', linewidth=2, markersize=6)
                ax3.set_xlabel('Threshold')
                ax3.set_ylabel('Profit Factor (capped at 10)')
                ax3.set_title('Profit Factor vs Threshold')
                ax3.grid(True, alpha=0.3)
                if 'best_result' in locals():
                    ax3.axvline(best_result['threshold'], color='red', linestyle='--', alpha=0.7, label=f'Best: {best_result["threshold"]}')
                    ax3.legend()
                
                # График 4: Score (основной критерий)
                ax4.plot(thresholds_plot, scores_plot, 'purple', marker='o', linewidth=2, markersize=6)
                ax4.set_xlabel('Threshold')
                ax4.set_ylabel('Score (capped at 10)')
                ax4.set_title('Optimization Score vs Threshold')
                ax4.grid(True, alpha=0.3)
                if 'best_result' in locals():
                    ax4.axvline(best_result['threshold'], color='red', linestyle='--', alpha=0.7, label=f'Best: {best_result["threshold"]}')
                    ax4.legend()
            
                plt.tight_layout()
                plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("📊 График недоступен - нет данных для отображения")
        else:
            print("🔇 График оптимизации порогов отключен (режим --silent)")
    else:
        print("📊 График недоступен при 100% обучении (нет данных бэктеста)")
    
    # === Рекомендации по использованию ===
    if test_size_frac > 0 and 'best_result' in locals():
        print(f"\n📝 РЕКОМЕНДАЦИЯ:")
        threshold_key = f"{cfg.symbol.replace('/','')}_{cfg.tf_train}"
        print(f"Обновите model_thresholds.json: \"{threshold_key}\": {best_result['threshold']:.2f}")
        print(f"(Predict-2.3.py автоматически загрузит оптимальный порог из файла)")
        
        # Дополнительный анализ
        print(f"\n📊 ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
        valid_results = [r for r in results if r['total_trades'] > 0]
        reasonable_options = [r for r in valid_results if r['total_trades'] >= 40 and r['win_rate'] >= 0.80]
        if reasonable_options:
            print("Альтернативные хорошие пороги (≥40 сделок, ≥80% WR):")
            for opt in sorted(reasonable_options, key=lambda x: x['score'], reverse=True)[:3]:
                print(f"  Threshold {opt['threshold']:.2f}: {opt['total_trades']} сделок, {opt['win_rate']:.1%} WR, PF {opt['profit_factor']:.2f}, Score {opt['score']:.3f}")
        
        # Показать топ-3 по score для сравнения
        if valid_results:
            print(f"\nТОП-3 ПО SCORE:")
            sorted_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)
            for i, opt in enumerate(sorted_results[:3], 1):
                print(f"  {i}. Threshold {opt['threshold']:.2f}: {opt['total_trades']} сделок, {opt['win_rate']:.1%} WR, PF {opt['profit_factor']:.2f}, Score {opt['score']:.3f}")

