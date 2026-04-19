# Predict-Advanced.py - версия с 51 фичей из CreateModel-2.py
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import ccxt
import ta
try:
    import winsound
except ImportError:
    winsound = None
from xgboost import XGBClassifier
from datetime import datetime
from telegram_sender import send_trading_signal
from ModelLaunchNotifier import notify_model_launch, notify_model_error
from typing import Tuple, Optional
from SMCFilter import SMCFilter


# === Helpers: probability remapping and feature health diagnostics ===
def _remap_proba_to_hls(proba_row: np.ndarray, model: XGBClassifier) -> np.ndarray:
    """Ensure probabilities are ordered as [HOLD, LONG, SHORT].

    Uses model.classes_ if available. Supports numeric labels (0,1,2) and
    string labels ('HOLD','LONG','SHORT'). Falls back to original order on error.
    """
    try:
        if proba_row is None:
            return proba_row
        # Ensure 1D array
        p = np.asarray(proba_row).ravel()
        classes = getattr(model, 'classes_', None)
        if classes is None:
            return p
        cls_arr = np.array(classes)
        # Numeric classes (expected: 0=HOLD,1=LONG,2=SHORT)
        if np.issubdtype(cls_arr.dtype, np.number):
            targ = np.array([0, 1, 2])
            idx = []
            for t in targ:
                found = np.where(cls_arr == t)[0]
                if len(found) == 0:
                    return p  # fallback
                idx.append(int(found[0]))
            return p[idx]
        # String classes
        names = [str(x).strip().lower() for x in cls_arr.tolist()]
        target_names = ['hold', 'long', 'short']
        idx = []
        for tn in target_names:
            if tn in names:
                idx.append(names.index(tn))
            else:
                return p  # fallback
        return p[idx]
    except Exception:
        return np.asarray(proba_row).ravel()


def _feature_health_debug(df: pd.DataFrame, feature_cols: list, window: int = 200) -> dict:
    """Quick feature sanity check on the last window rows.

    Returns a dict with lists of constant, NaN-containing, and Inf-containing features.
    """
    out = {"const_features": [], "nan_cols": [], "inf_cols": []}
    try:
        sub = df[feature_cols].tail(window).copy()
        for col in feature_cols:
            s = sub[col]
            # Track NaNs / Infs across the window
            if s.isna().any():
                out["nan_cols"].append(col)
            if np.isinf(s.values).any():
                out["inf_cols"].append(col)
            # Constant (or effectively constant) feature detection
            if s.nunique(dropna=True) <= 1:
                out["const_features"].append(col)
    except Exception:
        pass
    return out


# === Variant A: Dynamic feasibility-driven TP/SL adaptation ===
def adapt_tp_sl_for_min_net_rr(price: float,
                               stop_loss: float,
                               take_profit: float,
                               side: str,
                               cost_pct: float,
                               min_net_rr: Optional[float],
                               max_tp_mult: float = 5.0,
                               min_sl_mult: float = 0.4,
                               atr: Optional[float] = None,
                               base_sl_mult: Optional[float] = None,
                               base_tp_mult: Optional[float] = None,
                               atr_price: Optional[float] = None,
                               grow_tp_step: float = 0.25,
                               shrink_sl_step: float = 0.1) -> Tuple[float, float, dict]:
    """Dynamically adjust TP/SL geometry to attempt reaching configured min_net_rr.

    Strategy:
      1. Compute current net_rr = ( (TP-P)/P - cost ) / ( (P-SL)/P + cost ) for LONG (analogous for SHORT).
      2. If already >= min_net_rr or min_net_rr not set -> return unchanged.
      3. If ATR context & original multipliers provided, first try increasing TP multiplier up to max_tp_mult.
      4. If still infeasible, try shrinking SL multiplier down to min_sl_mult (recompute SL accordingly).
      5. If both exhausted and still < min_net_rr, mark infeasible and return last geometry.

    Returns: (new_stop_loss, new_take_profit, info_dict)
    info_dict: { 'attempts': int, 'final_net_rr': float, 'adjusted': bool, 'infeasible': bool,
                 'tp_mult_final': float|None, 'sl_mult_final': float|None }
    """
    info = {
        'attempts': 0,
        'adjusted': False,
        'infeasible': False,
        'tp_mult_final': None,
        'sl_mult_final': None,
        'final_net_rr': None,
    }
    if side not in ('LONG', 'SHORT'):
        return stop_loss, take_profit, info
    if min_net_rr is None:
        return stop_loss, take_profit, info
    try:
        price = float(price)
        stop_loss = float(stop_loss)
        take_profit = float(take_profit)
    except Exception:
        return stop_loss, take_profit, info
    if price <= 0:
        return stop_loss, take_profit, info

    def distances(p, slv, tpv, sd):
        if sd == 'LONG':
            t = (tpv - p) / p
            s = (p - slv) / p
        else:
            t = (p - tpv) / p
            s = (slv - p) / p
        return t, s

    target_pct, stop_pct = distances(price, stop_loss, take_profit, side)
    if target_pct <= 0 or stop_pct <= 0:
        return stop_loss, take_profit, info
    net_target = target_pct - cost_pct
    net_stop = stop_pct + cost_pct
    if net_target <= 0:
        return stop_loss, take_profit, info
    net_rr = net_target / net_stop if net_stop > 0 else 0
    if min_net_rr is None or net_rr >= min_net_rr:
        info['final_net_rr'] = net_rr
        return stop_loss, take_profit, info

    # If we cannot use ATR context, we at least try to extend TP proportionally (no precise multipliers known)
    tp_mult = None
    sl_mult = None
    if atr and atr_price is not None and atr > 0:
        # Derive current multipliers if possible
        if base_tp_mult is not None and base_sl_mult is not None:
            # Provided by caller -> trust
            tp_mult = base_tp_mult
            sl_mult = base_sl_mult
        else:
            # Approximate from current distances
            tp_mult = target_pct * price / atr
            sl_mult = stop_pct * price / atr
        # Grow TP first
        while tp_mult < max_tp_mult:
            tp_mult += grow_tp_step
            info['attempts'] += 1
            if side == 'LONG':
                take_profit = price + tp_mult * atr
            else:
                take_profit = price - tp_mult * atr
            target_pct, stop_pct = distances(price, stop_loss, take_profit, side)
            net_target = target_pct - cost_pct
            net_stop = stop_pct + cost_pct
            if net_target > 0 and net_stop > 0:
                net_rr = net_target / net_stop
            else:
                net_rr = 0
            if net_rr >= min_net_rr:
                info.update({'adjusted': True, 'tp_mult_final': tp_mult, 'sl_mult_final': sl_mult, 'final_net_rr': net_rr})
                return stop_loss, take_profit, info
        # Shrink SL if still not enough
        while sl_mult > min_sl_mult:
            sl_mult = max(min_sl_mult, sl_mult - shrink_sl_step)
            info['attempts'] += 1
            if side == 'LONG':
                stop_loss = price - sl_mult * atr
            else:
                stop_loss = price + sl_mult * atr
            target_pct, stop_pct = distances(price, stop_loss, take_profit, side)
            net_target = target_pct - cost_pct
            net_stop = stop_pct + cost_pct
            if net_target > 0 and net_stop > 0:
                net_rr = net_target / net_stop
            else:
                net_rr = 0
            if net_rr >= min_net_rr:
                info.update({'adjusted': True, 'tp_mult_final': tp_mult, 'sl_mult_final': sl_mult, 'final_net_rr': net_rr})
                return stop_loss, take_profit, info
        # Infeasible with ATR adjustments
        info.update({'infeasible': True, 'tp_mult_final': tp_mult, 'sl_mult_final': sl_mult, 'final_net_rr': net_rr})
        return stop_loss, take_profit, info
    else:
        # Fallback: scale TP linearly up to 2x distance, then SL down 50%
        orig_tp = take_profit
        orig_sl = stop_loss
        for scale in [1.25, 1.5, 1.75, 2.0]:
            info['attempts'] += 1
            if side == 'LONG':
                take_profit = price + (orig_tp - price) * scale
            else:
                take_profit = price - (price - orig_tp) * scale
            target_pct, stop_pct = distances(price, stop_loss, take_profit, side)
            net_target = target_pct - cost_pct
            net_stop = stop_pct + cost_pct
            net_rr = net_target / net_stop if (net_target > 0 and net_stop > 0) else 0
            if net_rr >= min_net_rr:
                info.update({'adjusted': True, 'final_net_rr': net_rr})
                return stop_loss, take_profit, info
        # shrink SL rough
        if side == 'LONG':
            stop_loss = price - (price - orig_sl) * 0.7
        else:
            stop_loss = price + (orig_sl - price) * 0.7
        target_pct, stop_pct = distances(price, stop_loss, take_profit, side)
        net_target = target_pct - cost_pct
        net_stop = stop_pct + cost_pct
        net_rr = net_target / net_stop if (net_target > 0 and net_stop > 0) else 0
        if net_rr >= min_net_rr:
            info.update({'adjusted': True, 'final_net_rr': net_rr})
            return stop_loss, take_profit, info
        info.update({'infeasible': True, 'final_net_rr': net_rr})
        return stop_loss, take_profit, info


def parse_args():
    p = argparse.ArgumentParser(description="Predict signal with advanced XGBoost model (51 features)")
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--tf", type=str, default="30m")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--min_conf", type=float, default=0.66)
    p.add_argument("--atr_sl_mul", type=float, default=1.5, help="ATR multiplier for stop loss")
    p.add_argument("--atr_tp_mul", type=float, default=2.5, help="ATR multiplier for take profit")
    p.add_argument("--interval", type=int, default=60, help="Prediction interval in seconds (default: 60)")
    p.add_argument("--enable_smc_filter", action="store_true", help="Enable SMC (Smart Money Concepts) filter for signal validation")
    p.add_argument("--smc_min_confluence", type=int, default=3, help="Minimum confluence score for SMC filter (3-5, default: 3)")
    return p.parse_args()


def load_optimal_threshold(symbol, timeframe, meta_file_path=None):
    """Загружает оптимальный порог для конкретной монеты и таймфрейма
    
    Приоритет:
    1. Из model_thresholds.json (ручные настройки имеют приоритет)
    2. Из конкретного мета-файла (переданного через meta_file_path)
    3. Если ничего не найдено - возвращает (None, None) (модель будет пропущена для безопасности)
    
    Returns:
        tuple: (threshold, source) где source = "MANUAL" или "AUTO"
    """
    
    # 1. Сначала проверяем ручные настройки в model_thresholds.json
    try:
        with open("model_thresholds.json", "r") as f:
            thresholds = json.load(f)
        
        key = f"{symbol.replace('/', '')}_{timeframe}"
        
        if key in thresholds["optimal_thresholds"]:
            threshold = thresholds["optimal_thresholds"][key]
            print(f"[MANUAL] Используем РУЧНОЙ порог из model_thresholds.json: {threshold}")
            return threshold, "MANUAL"
        
    except Exception as e:
        print(f"Ошибка загрузки из model_thresholds.json: {e}")
    
    # 2. Загружаем из конкретного мета-файла (приоритет - файл, выбранный AutoLauncher)
    if meta_file_path and os.path.exists(meta_file_path):
        try:
            print(f"[AUTO] Используем АВТОМАТИЧЕСКИЙ порог из указанного мета-файла: {meta_file_path}")
            
            with open(meta_file_path, "r") as f:
                meta = json.load(f)
            
            if "backtest_results" in meta and "confidence_threshold" in meta["backtest_results"]:
                threshold = meta["backtest_results"]["confidence_threshold"]
                print(f"Найден автоматический порог: {threshold}")
                return threshold, "AUTO"
            else:
                print(f"Предупреждение: В мета-файле {meta_file_path} нет confidence_threshold")
                
        except Exception as e:
            print(f"Ошибка загрузки из указанного мета-файла: {e}")
    
    # 3. Если ничего не найдено - возвращаем None
    print(f"[ERROR] ПОРОГ НЕ НАЙДЕН для {symbol} {timeframe}")
    print("   Нет ни ручной настройки в model_thresholds.json, ни мета-файла модели")
    print("   Модель будет пропущена для безопасности")
    return None, None


# ===== НОВЫЕ ФИЧИ (скопированы из CreateModel-2.py) =====

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет моментум фичи"""
    # Momentum oscillator
    df['momentum_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
    df['momentum_20'] = ta.momentum.ROCIndicator(df['close'], window=20).roc()
    
    # Commodity Channel Index
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    # Ultimate Oscillator
    df['ultimate_osc'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
    
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет объёмные фичи"""
    # On Balance Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['obv_ma'] = df['obv'].rolling(window=10).mean()
    df['obv_signal'] = (df['obv'] > df['obv_ma']).astype(int)
    
    # Chaikin Money Flow
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
    
    # Volume Price Trend
    df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
    
    # Volume indicators
    volume_ma = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / volume_ma
    df['vol_breakout'] = (df['vol_ratio'] > 2.0).astype(int)
    df['vol_breakout_strong'] = (df['vol_ratio'] > 3.0).astype(int)
    
    return df

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет паттерновые фичи"""
    # Поддержка/сопротивление (упрощенная версия)
    window = 20
    df['resistance'] = df['high'].rolling(window=window).max()
    df['support'] = df['low'].rolling(window=window).min()
    
    # Расстояние до уровней
    df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
    df['dist_to_support'] = (df['close'] - df['support']) / df['close']
    
    # Близость к уровням
    df['near_resistance'] = (df['dist_to_resistance'] < 0.02).astype(int)
    df['near_support'] = (df['dist_to_support'] < 0.02).astype(int)
    
    # Расширенные уровни
    df['major_resistance'] = df['high'].rolling(window=50).max()
    df['major_support'] = df['low'].rolling(window=50).min()
    df['near_major_resistance'] = ((df['major_resistance'] - df['close']) / df['close'] < 0.03).astype(int)
    df['near_major_support'] = ((df['close'] - df['major_support']) / df['close'] < 0.03).astype(int)
    
    return df

def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет структурные фичи"""
    # Структура тренда
    lookback = 10
    df['higher_highs'] = (df['high'] > df['high'].shift(lookback)).astype(int)
    df['lower_lows'] = (df['low'] < df['low'].shift(lookback)).astype(int)
    df['higher_lows'] = (df['low'] > df['low'].shift(lookback)).astype(int)
    df['lower_highs'] = (df['high'] < df['high'].shift(lookback)).astype(int)
    
    # Тренд
    df['uptrend'] = (df['higher_highs'] & df['higher_lows']).astype(int)
    df['downtrend'] = (df['lower_highs'] & df['lower_lows']).astype(int)
    
    return df

def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет паттерны свечей"""
    # Размеры тела и теней
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # Нормализованные размеры
    df['body_ratio'] = df['body_size'] / df['total_range']
    df['upper_shadow_ratio'] = df['upper_shadow'] / df['total_range']
    df['lower_shadow_ratio'] = df['lower_shadow'] / df['total_range']
    
    # Паттерны
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
    
    # Простые свечные паттерны
    df['doji'] = (df['body_ratio'] < 0.1).astype(int)
    df['hammer'] = ((df['lower_shadow_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(int)
    
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет фичи волатильности"""
    # Percentile волатильности
    vol_window = 50
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['vol_percentile'] = atr.rolling(window=vol_window).rank(pct=True)
    
    # Режимы волатильности
    df['high_vol_regime'] = (df['vol_percentile'] > 0.8).astype(int)
    df['low_vol_regime'] = (df['vol_percentile'] < 0.2).astype(int)
    
    # Expanding/contracting volatility
    vol_ma_short = atr.rolling(window=10).mean()
    vol_ma_long = atr.rolling(window=30).mean()
    df['vol_expanding'] = (vol_ma_short > vol_ma_long).astype(int)
    
    return df

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет торговые сессии"""
    # Конвертируем в UTC время
    df['hour_utc'] = pd.to_datetime(df.index).hour
    
    # Торговые сессии (UTC)
    df['is_asia_session'] = ((df['hour_utc'] >= 0) & (df['hour_utc'] < 9)).astype(int)
    df['is_london_session'] = ((df['hour_utc'] >= 8) & (df['hour_utc'] < 16)).astype(int)
    df['is_ny_session'] = ((df['hour_utc'] >= 13) & (df['hour_utc'] < 22)).astype(int)
    
    # Ликвидные часы (overlap London + NY)
    df['is_liquid_hours'] = ((df['hour_utc'] >= 13) & (df['hour_utc'] < 16)).astype(int)
    
    return df

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет price action фичи"""
    # Скорость и ускорение цены
    df['price_velocity'] = df['close'].pct_change(5)
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # Дивергенция с RSI (упрощенная)
    rsi = ta.momentum.RSIIndicator(df['close']).rsi()
    price_trend = (df['close'] > df['close'].shift(10)).astype(int)
    rsi_trend = (rsi > rsi.shift(10)).astype(int)
    df['rsi_divergence'] = (price_trend != rsi_trend).astype(int)
    
    return df

def add_all_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет все 51 фичу"""
    df = add_momentum_features(df)
    df = add_volume_features(df)
    df = add_pattern_features(df)
    df = add_structure_features(df)
    df = add_candlestick_features(df)
    df = add_volatility_features(df)
    df = add_session_features(df)
    df = add_price_action_features(df)
    
    return df


def choose_tp_sl_via_grid(
    price: float,
    atr: float,
    side: str,
    p_hit_cal: Optional[float],
    min_net_rr: Optional[float],
    cost_pct: float,
    tp_range: Tuple[float, float, float],
    sl_range: Tuple[float, float, float],
    tp_mult_cap: Optional[float] = None,
) -> Tuple[float, float, float, float, dict]:
    """Перебирает малую ATR-сетку (tp_mult, sl_mult) и выбирает пару с максимальным ожидаемым EV.

    Критерии отбора:
      - Рассчитываем target_pct и stop_pct исходя из (tp_mult, sl_mult)
      - Учитываем стоимость сделки (cost_pct) => net_target, net_stop
      - Считаем net_rr = net_target / net_stop, требуем net_rr >= min_net_rr (если задано)
      - Считаем ожидаемую доходность EV = p_hit_cal * net_target - (1 - p_hit_cal) * net_stop
      - Выбираем максимум EV среди допустимых кандидатов

    Параметры:
      - tp_range/sl_range: (start, end, step)
      - tp_mult_cap: если задан, ограничивает верхнюю границу множителя TP

    Возвращает: (stop_loss, take_profit, sl_mult, tp_mult, info)
    """
    info = {
        'used_grid': True,
        'best_ev': None,
        'best_net_rr': None,
        'candidates': 0,
        'min_net_rr': min_net_rr,
        'cost_pct': cost_pct,
        'tp_mult_cap': tp_mult_cap,
    }
    try:
        price = float(price)
        atr = float(atr)
        cost_pct = float(cost_pct)
    except Exception:
        return (price, price, 0.0, 0.0, {**info, 'error': 'invalid price/atr/cost'})
    if not np.isfinite(price) or not np.isfinite(atr) or price <= 0 or atr <= 0:
        return (price, price, 0.0, 0.0, {**info, 'error': 'non-finite price/atr'})
    if side not in ('LONG', 'SHORT'):
        return (price, price, 0.0, 0.0, {**info, 'error': 'invalid side'})
    if p_hit_cal is None or not isinstance(p_hit_cal, (int, float)):
        return (price, price, 0.0, 0.0, {**info, 'error': 'invalid p_hit_cal'})

    tp_start, tp_end, tp_step = tp_range
    sl_start, sl_end, sl_step = sl_range
    if tp_mult_cap is not None:
        tp_end = min(tp_end, float(tp_mult_cap))

    best = None  # (ev, sl_mult, tp_mult, net_rr, stop_loss, take_profit)
    tp_m = tp_start
    # Итерация по сетке
    while tp_m <= tp_end + 1e-9:
        sl_m = sl_start
        while sl_m <= sl_end + 1e-9:
            # Геометрия
            if side == 'LONG':
                tp_v = price + tp_m * atr
                sl_v = price - sl_m * atr
                target_pct = (tp_v - price) / price
                stop_pct = (price - sl_v) / price
            else:
                tp_v = price - tp_m * atr
                sl_v = price + sl_m * atr
                target_pct = (price - tp_v) / price
                stop_pct = (sl_v - price) / price

            net_target = target_pct - cost_pct
            net_stop = stop_pct + cost_pct
            if net_target <= 0 or net_stop <= 0:
                info['candidates'] += 1
                sl_m += sl_step
                continue
            net_rr = net_target / net_stop
            if min_net_rr is not None and net_rr < min_net_rr:
                info['candidates'] += 1
                sl_m += sl_step
                continue
            # Ожидаемая доходность (калиброванная)
            ev = p_hit_cal * net_target - (1 - p_hit_cal) * net_stop

            info['candidates'] += 1
            if (best is None) or (ev > best[0]):
                best = (ev, sl_m, tp_m, net_rr, sl_v, tp_v)
            sl_m += sl_step
        tp_m += tp_step

    if best is None:
        # Ничего не прошло фильтры — вернём исходную цену (нет сделки)
        return (price, price, 0.0, 0.0, {**info, 'error': 'no_feasible_candidates'})

    ev, sl_m, tp_m, net_rr, sl_v, tp_v = best
    info['best_ev'] = ev
    info['best_net_rr'] = net_rr
    return (sl_v, tp_v, sl_m, tp_m, info)


def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    print("Fetching OHLCV data via ccxt...")
    exchanges = [
        ('binance', ccxt.binance()),
        ('kucoin', ccxt.kucoin()),
        ('gate', ccxt.gate()),
        ('mexc', ccxt.mexc())
    ]
    
    # Попробуем разные биржи
    for exchange_name, exchange in exchanges:
        try:
            if hasattr(exchange, 'load_markets'):
                exchange.load_markets()
                
            if symbol in exchange.markets:
                print("Fetching data from", exchange_name)
                data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                print(f"Fetched {len(data)} bars")
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                print(f"Используем {symbol} на {exchange_name}")
                return df
        except Exception as e:
            print(f"❌ {exchange_name}: {e}")
            continue
    
    raise ValueError(f"Не удалось получить данные для {symbol}")


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт всех фичей для предикции"""
    
    # Базовые технические индикаторы
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_ma"] = df["rsi"].rolling(window=3).mean()
    df["rsi_signal"] = (df["rsi"] < 30).astype(int) * 2 + (df["rsi"] > 70).astype(int)

    # EMA
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_55"] = ta.trend.EMAIndicator(df["close"], window=55).ema_indicator()
    df["ema_fast"] = df["ema_21"]  # Alias для совместимости
    df["ema_slow"] = df["ema_55"]  # Alias для совместимости
    df["ema_signal"] = (df["close"] > df["ema_21"]).astype(int)
    df["ema_cross"] = (df["ema_21"] > df["ema_55"]).astype(int)

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    # Volume
    df["vol_ma"] = df["volume"].rolling(window=10).mean()
    df["vol_signal"] = (df["volume"] > df["vol_ma"]).astype(int)

    # MACD
    macd_line, macd_signal, macd_histogram = ta.trend.MACD(df["close"]).macd(), \
                                            ta.trend.MACD(df["close"]).macd_signal(), \
                                            ta.trend.MACD(df["close"]).macd_diff()
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_histogram
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]  # Добавлено
    df["bb_signal"] = (df["close"] > df["bb_upper"]).astype(int) * 2 + (df["close"] < df["bb_lower"]).astype(int)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_signal"] = (df["stoch_k"] < 20).astype(int) * 2 + (df["stoch_k"] > 80).astype(int)

    # Дополнительные базовые фичи
    df["close_sma_ratio"] = df["close"] / df["close"].rolling(window=20).mean()
    df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    
    # Rolling volatility of returns (align with training)
    df["rvol_20"] = df["ret_1"].rolling(20).std()
    
    # Momentum 5
    df["momentum_5"] = ta.momentum.ROCIndicator(df["close"], window=5).roc()
    
    # VPT SMA
    vpt = ta.volume.VolumePriceTrendIndicator(df["close"], df["volume"]).volume_price_trend()
    df["vpt_sma"] = vpt.rolling(window=10).mean()
    
    # ADX
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()

    # Мульти-таймфрейм (H1)
    df_h1 = df.resample('1h').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_h1["rsi_h1"] = ta.momentum.RSIIndicator(df_h1["close"], window=14).rsi()
    df_h1["ema_h1"] = ta.trend.EMAIndicator(df_h1["close"], window=50).ema_indicator()
    
    # Ресэмплинг обратно к 30м
    df["rsi_h1"] = df_h1["rsi_h1"].reindex(df.index, method='ffill')
    df["ema_h1"] = df_h1["ema_h1"].reindex(df.index, method='ffill')
    df["rsi_h"] = df["rsi_h1"]  # Alias для совместимости
    df["ema_h"] = df["ema_h1"]  # Alias для совместимости
    df["h1_signal"] = (df["close"] > df["ema_h1"]).astype(int)

    # Добавляем все 51 продвинутую фичу
    df = add_all_advanced_features(df)

    return df


# Добавить перед calculate_sl_tp
def detect_market_regime(df):
    """Определяет текущий рыночный режим и возвращает (regime, details)

    Режимы:
      - trending_bull: EMA50 > EMA200, цена выше EMA50, ADX сильный
      - trending_bear: EMA50 < EMA200, цена ниже EMA50, ADX сильный
      - mean_reversion: волатильность сжата (BB width contraction) или цена статистически перепродана/перекуплена при слабом тренде
      - neutral: ничего явно не выражено

    details включает используемые метрики и причины выбора.
    """
    details = {}
    try:
        if len(df) < 210:  # Нужно >= 200 баров для EMA200 плюс запас
            return "neutral", {"reason": "insufficient_data", "length": len(df)}

        close = df['close']
        high = df['high']
        low = df['low']

        ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()

        last_close = float(close.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])
        last_ema200 = float(ema200.iloc[-1])

        # Условия тренда по отношению и наклону (смотрим последние 5 баров для сглаживания)
        ema50_slope = float(ema50.iloc[-1] - ema50.iloc[-5]) / last_ema50 if last_ema50 != 0 else 0
        ema200_slope = float(ema200.iloc[-1] - ema200.iloc[-5]) / last_ema200 if last_ema200 != 0 else 0

        uptrend_condition = last_close > last_ema50 > last_ema200 and ema50_slope > 0 and ema200_slope >= 0
        downtrend_condition = last_close < last_ema50 < last_ema200 and ema50_slope < 0 and ema200_slope <= 0

        # ADX и компоненты DI для проверки силы тренда
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        adx_series = adx_ind.adx()
        adx_val = float(adx_series.iloc[-1])
        di_plus = float(adx_ind.adx_pos().iloc[-1])
        di_minus = float(adx_ind.adx_neg().iloc[-1])
        strong_trend = adx_val > 22  # чуть мягче порога 25

        # Bollinger Bands метрики
        bb = ta.volatility.BollingerBands(close)
        bb_h = bb.bollinger_hband()
        bb_l = bb.bollinger_lband()
        bb_m = bb.bollinger_mavg()
        bb_width_series = (bb_h - bb_l) / bb_m
        current_bb_width = float(bb_width_series.iloc[-1])
        bb_width_mean20 = float(bb_width_series.iloc[-20:].mean())
        contraction = current_bb_width < bb_width_mean20 * 0.8  # волатильность сжалась

        # Оценка статистического отклонения цены (z-score относительно среднего Bollinger, std берём из ширины)
        # width = (upper-lower)/middle = 2*k*std/middle => std ≈ (upper-lower)/(2*k)
        # k (num_std) по умолчанию = 2
        bb_std = (bb_h - bb_l) / 4.0
        last_std = float(bb_std.iloc[-1]) if float(bb_std.iloc[-1]) != 0 else 1e-9
        last_bb_mid = float(bb_m.iloc[-1])
        zscore = (last_close - last_bb_mid) / last_std

        # Слабость тренда: ADX ниже 20 или DI близки
        weak_trend = adx_val < 18 or abs(di_plus - di_minus) < 5

        # Mean reversion условия: либо сжатие + слабый тренд, либо экстремум z-score при не сильном тренде
        extreme_extension = abs(zscore) >= 1.0  # цена вышла за ~1 std относительно середины
        mean_reversion_condition = (contraction and weak_trend) or (extreme_extension and not strong_trend)

        # Итоговый выбор режима (приоритет трендов если они сильные)
        if uptrend_condition and strong_trend and di_plus > di_minus:
            regime = "trending_bull"
        elif downtrend_condition and strong_trend and di_minus > di_plus:
            regime = "trending_bear"
        elif mean_reversion_condition:
            regime = "mean_reversion"
        else:
            regime = "neutral"

        # Направление ожидаемого отката при mean_reversion
        mean_rev_direction = None
        if regime == "mean_reversion":
            if zscore > 0.75:
                mean_rev_direction = "down"  # ожидаем откат вниз после перекупленности
            elif zscore < -0.75:
                mean_rev_direction = "up"    # ожидаем откат вверх после перепроданности
            else:
                mean_rev_direction = "sideways"  # просто диапазон

        details.update({
            "close": last_close,
            "ema50": last_ema50,
            "ema200": last_ema200,
            "ema50_slope_pct": ema50_slope * 100,
            "ema200_slope_pct": ema200_slope * 100,
            "adx": adx_val,
            "di_plus": di_plus,
            "di_minus": di_minus,
            "strong_trend": strong_trend,
            "weak_trend": weak_trend,
            "bb_width": current_bb_width,
            "bb_width_mean20": bb_width_mean20,
            "vol_contraction": contraction,
            "zscore": zscore,
            "extreme_extension": extreme_extension,
            "uptrend_condition": uptrend_condition,
            "downtrend_condition": downtrend_condition,
            "mean_reversion_condition": mean_reversion_condition,
            "mean_reversion_direction": mean_rev_direction,
        })

        # Причины в человеко-читаемом виде
        reasons = []
        if regime == "trending_bull":
            reasons.append("EMA50 > EMA200, положительный наклон, ADX сильный, DI+ > DI-")
        if regime == "trending_bear":
            reasons.append("EMA50 < EMA200, отрицательный наклон, ADX сильный, DI- > DI+")
        if regime == "mean_reversion":
            if contraction and weak_trend:
                reasons.append("Сжатие волатильности (BB width contraction) + слабый тренд (низкий ADX / схожие DI)")
            if extreme_extension and not strong_trend:
                reasons.append(f"Цена статистически отклонена (z-score {zscore:.2f}) при отсутствии сильного тренда")
            if mean_rev_direction == "down":
                reasons.append("Цена выше среднего диапазона → возможен откат вниз")
            elif mean_rev_direction == "up":
                reasons.append("Цена ниже среднего диапазона → возможен откат вверх")
        if regime == "neutral":
            reasons.append("Нет достаточных признаков тренда или отклонения для классификации")

        details['reasons'] = reasons
        return regime, details
    except Exception as e:
        return "neutral", {"error": str(e)}


def format_regime_explanation(regime: str, details: dict) -> str:
    """Строит краткое текстовое объяснение для печати"""
    if 'error' in details:
        return f"Не удалось определить режим (ошибка: {details['error']})"
    parts = []
    if regime == 'mean_reversion' and details.get('mean_reversion_direction') in ('up', 'down'):
        direction_txt = 'ожидается откат ВВЕРХ' if details['mean_reversion_direction'] == 'up' else 'ожидается откат ВНИЗ'
        parts.append(direction_txt)
    if details.get('reasons'):
        parts.append("; ".join(details['reasons']))
    # Добавим ключевые числа (ограничим)
    metrics = []
    for k in ["adx", "zscore", "bb_width", "bb_width_mean20"]:
        if k in details:
            metrics.append(f"{k}={details[k]:.2f}")
    if metrics:
        parts.append(" | ".join(metrics))
    return " | ".join(parts)

# Обновить функцию calculate_sl_tp
def calculate_sl_tp(price, atr, signal, market_regime, sl_mul=None, tp_mul=None):
    """Рассчитывает stop-loss и take-profit на основе ATR и рыночного режима.
    Скорректированные, более широкие стопы для снижения ложных срабатываний.
    """
    # Используем адаптивные значения из бэктестов
    if sl_mul is None or tp_mul is None:
        if market_regime == "trending_bull":
            # Более широкий стоп для следования за трендом
            sl_mul, tp_mul = 2.0, 2.5
        elif market_regime == "trending_bear":
            # Значительно увеличен стоп с 0.8 до 2.0
            sl_mul, tp_mul = 2.0, 2.5
        elif market_regime == "mean_reversion":
            # Для отката к среднему R:R может быть ближе к 1, но стоп все равно нужен адекватный
            sl_mul, tp_mul = 1.5, 1.5
        else:  # neutral или fallback
            # Более консервативные значения для неопределенности
            sl_mul, tp_mul = 2.5, 3.0
    
    if signal == "LONG":
        stop_loss = price - (atr * sl_mul)
        take_profit = price + (atr * tp_mul)
    elif signal == "SHORT":
        stop_loss = price + (atr * sl_mul)
        take_profit = price - (atr * tp_mul)
    else:  # HOLD
        stop_loss = price
        take_profit = price
    
    return stop_loss, take_profit, sl_mul, tp_mul


def main():
    args = parse_args()
    
    # Информация об интервале
    interval_text = f"{args.interval} сек" if args.interval < 60 else f"{args.interval//60} мин"
    print(f"⏱️ Интервал проверок: {interval_text}")
    
    # Загружаем оптимальный порог (используем конкретный мета-файл, выбранный AutoLauncher)
    optimal_threshold, threshold_source = load_optimal_threshold(args.symbol, args.tf, args.meta)
    
    if optimal_threshold is None:
        print(f"\n[CRITICAL] КРИТИЧЕСКАЯ ОШИБКА: Невозможно запустить торговлю для {args.symbol} {args.tf}")
        print("[SOLUTION] РЕШЕНИЕ: Добавьте порог в model_thresholds.json или обучите модель заново")
        print("[EXAMPLE] Пример: \"ETHUSDT_30m\": 0.65")
        sys.exit(1)
    
    print(f"Используется оптимальный порог для {args.symbol}: {optimal_threshold} ({threshold_source})")
    
    # Загружаем метаданные и модель
    try:
        with open(args.meta, "r") as f:
            meta = json.load(f)
        p_hit_map = meta.get('p_hit_map')  # может быть None для старых моделей
        
        model = XGBClassifier()
        model.load_model(args.model)
        
        feature_cols = meta["feature_cols"]
        
        # Извлекаем данные для телеграм сообщений
        backtest_results = meta.get('backtest_results', {})
        horizon_bars = meta.get('config', {}).get('horizon_bars', 6)
        model_filename = os.path.basename(args.model)
        
        # 🚀 УВЕДОМЛЕНИЕ О ЗАПУСКЕ МОДЕЛИ
        notify_model_launch(args.symbol, args.tf, args.model, args.meta, optimal_threshold, threshold_source)
        
        # 🛡️ ИНИЦИАЛИЗАЦИЯ SMC ФИЛЬТРА
        smc_filter = None
        if args.enable_smc_filter:
            smc_filter = SMCFilter(min_confluence_score=args.smc_min_confluence)
            print(f"🛡️ SMC Filter АКТИВИРОВАН (минимальный confluence: {args.smc_min_confluence}/6)")
            print(f"   Все ML сигналы будут проверяться через Smart Money Concepts")
        else:
            print(f"⚠️ SMC Filter ОТКЛЮЧЕН (используется только ML модель)")
        
        print(f"Система запущена. Нажмите Ctrl+C для остановки...")
        
    except Exception as e:
        error_msg = f"Ошибка загрузки модели: {e}"
        notify_model_error(args.symbol, args.tf, error_msg)
        sys.exit(1)
    
    # Подготовим функцию оценки p_hit из p_hit_map
    def estimate_p_hit_live(conf: float) -> float:
        # Поддержка только новой схемы: {'conf_min','conf_max'} (и fallback на min/max)
        if not p_hit_map or 'bins' not in p_hit_map:
            return conf  # fallback: используем confidence как proxy
        bins = p_hit_map.get('bins', [])
        for b in bins:
            l = b.get('conf_min', b.get('min'))
            h = b.get('conf_max', b.get('max'))
            if l is None or h is None:
                continue
            try:
                l = float(l)
                h = float(h)
                if l <= float(conf) < h:
                    return float(b.get('p_hit', conf))
            except Exception:
                continue
        # если не найдено (конфиг может иметь правый край), возьмём последний бин с доступными границами
        if bins:
            last_bin = bins[-1]
            l = last_bin.get('conf_min', last_bin.get('min', 0))
            try:
                if float(conf) >= float(l):
                    return float(last_bin.get('p_hit', conf))
            except Exception:
                pass
        return float(p_hit_map.get('overall_p_hit', conf))

    last_signal = None
    
    while True:
        try:
            print("\nStart predicting...")

            # Получаем данные
            df = fetch_ohlcv_ccxt(args.symbol, args.tf, args.limit)
            
            # Рассчитываем фичи
            df_feat = calculate_features(df)
            
            # Убираем NaN
            df_feat = df_feat.dropna()
            
            if len(df_feat) == 0:
                print("❌ Нет данных после расчёта фичей")
                time.sleep(args.interval)
                continue
                
            # Проверяем наличие всех фичей
            missing_features = [f for f in feature_cols if f not in df_feat.columns]
            if missing_features:
                print(f"❌ Отсутствуют фичи: {missing_features}")
                time.sleep(args.interval)
                continue
            
            # Предсказание
            X = df_feat[feature_cols].values
            # Диагностика NaN/inf в последней строке фичей
            try:
                x_last = X[-1]
                nan_count = int(np.isnan(x_last).sum())
                inf_count = int(np.isinf(x_last).sum())
            except Exception:
                nan_count = inf_count = -1

            y_proba = model.predict_proba(X)
            
            # Последняя свеча: получаем сырые вероятности и ремапим к [HOLD,LONG,SHORT]
            last_proba_raw = y_proba[-1]
            last_proba = _remap_proba_to_hls(last_proba_raw, model)
            last_timestamp = df_feat.index[-1]
            
            # Классы: 0=HOLD, 1=LONG, 2=SHORT
            hold_conf = float(last_proba[0])
            long_conf = float(last_proba[1]) 
            short_conf = float(last_proba[2])

            # Отладка: если HOLD > 0.99 долгое время, логируем сырые вероятности и NaN в фичах
            if hold_conf >= 0.99:
                try:
                    with open("signals_debug.log", "a", encoding="utf-8") as dfdbg:
                        dfdbg.write(f"DEBUG {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
                        # Сырые вероятности до ремапа
                        try:
                            dfdbg.write(
                                "raw_proba: "
                                f"[{float(last_proba_raw[0]):.6f}, {float(last_proba_raw[1]):.6f}, {float(last_proba_raw[2]):.6f}]\n"
                            )
                        except Exception:
                            pass
                        # Вероятности после ремапа
                        dfdbg.write(
                            f"mapped_proba: HOLD={hold_conf:.6f}, LONG={long_conf:.6f}, SHORT={short_conf:.6f}\n"
                        )
                        dfdbg.write(f"nan_count_last_row={nan_count}, inf_count_last_row={inf_count}\n\n")
                except Exception:
                    pass
            
            # Определяем сигнал
            max_conf = max(long_conf, short_conf)
            
            if max_conf >= optimal_threshold:
                if long_conf > short_conf:
                    signal = "LONG"
                    confidence = long_conf
                else:
                    signal = "SHORT"
                    confidence = short_conf
            else:
                signal = "HOLD"
                confidence = hold_conf
            
            # Показываем результат только при изменении сигнала
            current_price = df_feat['close'].iloc[-1]
            
            # 🛡️ SMC ФИЛЬТРАЦИЯ (если включена)
            smc_result = None
            original_signal = signal  # Сохраняем оригинальный ML сигнал для лога
            
            if smc_filter and signal in ["LONG", "SHORT"]:
                try:
                    # Конвертируем символ для SMC (BTCUSDT -> BTC/USDT)
                    smc_symbol = args.symbol if '/' in args.symbol else f"{args.symbol[:-4]}/{args.symbol[-4:]}"
                    
                    smc_result = smc_filter.validate_signal(
                        signal=signal,
                        df=df_feat,
                        ml_confidence=confidence,
                        symbol=smc_symbol
                    )
                    
                    if not smc_result['approved']:
                        print(f"\n🚫 SMC ФИЛЬТР ОТКЛОНИЛ {signal} СИГНАЛ")
                        print(f"   ML Confidence: {confidence:.1%}")
                        print(f"   Confluence Score: {smc_result['confluence_score']}/6 (требуется ≥{args.smc_min_confluence})")
                        print(f"   Причины отклонения:")
                        for reason in smc_result['reasons']:
                            if '❌' in reason or '⚠️' in reason:
                                print(f"     {reason}")
                        
                        # Переводим в HOLD
                        signal = "HOLD"
                        confidence = hold_conf
                    else:
                        # Сигнал одобрен - показываем детали
                        print(f"\n✅ SMC ФИЛЬТР ОДОБРИЛ {signal} СИГНАЛ")
                        print(f"   {smc_result['recommendation']}")
                        print(f"   Подтверждения:")
                        for reason in smc_result['reasons']:
                            if '✅' in reason:
                                print(f"     {reason}")
                
                except Exception as e:
                    print(f"⚠️ Ошибка SMC фильтра: {e}")
                    print(f"   Продолжаем с оригинальным ML сигналом: {signal}")
                    # При ошибке SMC не блокируем сигнал

            # Логируем все сигналы в файл signal.log
            try:
                # --- Новая проверка на p_hit ---
                p_hit_cal = estimate_p_hit_live(confidence)
                MIN_P_HIT_THRESHOLD = 0.55  # Минимальная калиброванная вероятность для входа
                if p_hit_cal < MIN_P_HIT_THRESHOLD:
                    print(f"🚫 Сигнал {signal} ({confidence:.1%}) пропущен: p_hit ({p_hit_cal:.1%}) < {MIN_P_HIT_THRESHOLD:.0%}")
                    time.sleep(args.interval)
                    continue
                # --- Конец проверки ---

                # Определяем рыночный режим
                market_regime, regime_details = detect_market_regime(df)
                regime_explanation = format_regime_explanation(market_regime, regime_details)
                print(f"📊 Рыночный режим: {market_regime.upper()} | {regime_explanation}")

                # Рассчитываем SL/TP с адаптивными значениями
                current_atr = df_feat['atr'].iloc[-1]
                stop_loss, take_profit, used_sl_mul, used_tp_mul = calculate_sl_tp(
                    current_price, current_atr, signal, market_regime
                )

                # === Optional: Live grid-based TP/SL optimizer (Variant B approx) ===
                optimizer_info = None
                try:
                    from risk_metrics import load_risk_config
                    _cfg_live = load_risk_config()
                    overrides_live = (_cfg_live.get('symbols') or {}).get(args.symbol.replace('/', ''), {})
                    use_grid_opt = overrides_live.get('use_live_grid_optimizer', _cfg_live.get('use_live_grid_optimizer', False))
                    min_net_rr_live = overrides_live.get('min_net_rr', _cfg_live.get('min_net_rr'))
                    cost_pct_live_opt = overrides_live.get('round_trip_cost_pct', _cfg_live.get('default_round_trip_cost_pct', 0.002))
                except Exception:
                    use_grid_opt = False
                    min_net_rr_live = None
                    cost_pct_live_opt = 0.002

                # Пытаемся взять сеточные диапазоны из meta; иначе дефолт из обучающей геометрии
                tp_range = (1.0, 4.0, 0.25)
                sl_range = (0.6, 2.0, 0.2)
                tp_mult_cap_meta = None
                try:
                    with open(args.meta, "r") as _mf:
                        _meta = json.load(_mf)
                    # Ищем возможные ключи с диапазонами
                    labeling_stats = _meta.get('labeling') or {}
                    if isinstance(labeling_stats, dict):
                        grid_stats = labeling_stats.get('grid_stats') or labeling_stats
                        # Популярные пути: grid_tp_range/grid_sl_range
                        _tp = grid_stats.get('grid_tp_range')
                        _sl = grid_stats.get('grid_sl_range')
                        if isinstance(_tp, (list, tuple)) and len(_tp) == 3:
                            tp_range = (float(_tp[0]), float(_tp[1]), float(_tp[2]))
                        if isinstance(_sl, (list, tuple)) and len(_sl) == 3:
                            sl_range = (float(_sl[0]), float(_sl[1]), float(_sl[2]))
                    # Ищем cap множителя TP (если бэктест его вычислил)
                    dyn_tp = _meta.get('dynamic_tp') or {}
                    tp_mult_cap_meta = dyn_tp.get('tp_mult_cap')
                except Exception:
                    pass

                if signal in ("LONG", "SHORT") and use_grid_opt:
                    # Калиброванная вероятность попадания
                    p_hit_cal_live = estimate_p_hit_live(confidence)
                    sl_v_opt, tp_v_opt, sl_m_opt, tp_m_opt, optimizer_info = choose_tp_sl_via_grid(
                        price=current_price,
                        atr=current_atr,
                        side=signal,
                        p_hit_cal=p_hit_cal_live if isinstance(p_hit_cal_live, (int, float)) else None,
                        min_net_rr=min_net_rr_live,
                        cost_pct=cost_pct_live_opt,
                        tp_range=tp_range,
                        sl_range=sl_range,
                        tp_mult_cap=tp_mult_cap_meta,
                    )
                    # Если оптимизация нашла допустимого кандидата — применим его
                    if optimizer_info and not optimizer_info.get('error'):
                        stop_loss, take_profit = sl_v_opt, tp_v_opt
                        used_sl_mul, used_tp_mul = sl_m_opt, tp_m_opt

                # === Dynamic adaptation (Variant A) ===
                # Load risk config for min_net_rr & cost
                try:
                    from risk_metrics import load_risk_config
                    _cfg_dyn = load_risk_config()
                    overrides_dyn = (_cfg_dyn.get('symbols') or {}).get(args.symbol.replace('/', ''), {})
                    min_net_rr_cfg = overrides_dyn.get('min_net_rr', _cfg_dyn.get('min_net_rr'))
                    cost_pct_dyn = overrides_dyn.get('round_trip_cost_pct', _cfg_dyn.get('default_round_trip_cost_pct', 0.002))
                except Exception:
                    min_net_rr_cfg = None
                    cost_pct_dyn = 0.002

                adapt_info = None
                if signal in ("LONG", "SHORT") and min_net_rr_cfg is not None:
                    # attempt adaptation only if baseline geometry insufficient
                    stop_loss_adapt, take_profit_adapt, adapt_info = adapt_tp_sl_for_min_net_rr(
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        side=signal,
                        cost_pct=cost_pct_dyn,
                        min_net_rr=min_net_rr_cfg,
                        atr=current_atr,
                        base_sl_mult=used_sl_mul,
                        base_tp_mult=used_tp_mul,
                        atr_price=current_price,
                        max_tp_mult=6.0,
                        min_sl_mult=0.4,
                        grow_tp_step=0.25,
                        shrink_sl_step=0.1,
                    )
                    if adapt_info and adapt_info.get('adjusted'):
                        stop_loss, take_profit = stop_loss_adapt, take_profit_adapt
                    # If infeasible we still continue; metrics filter may block later


                # Добавляем информацию о режиме в лог
                # Рассчитываем калиброванный p_hit (если доступен)
                p_hit_cal = estimate_p_hit_live(confidence)
                # Для расчёта калиброванного EV нужны tp/sl расстояния
                if signal in ["LONG", "SHORT"] and current_price > 0:
                    if signal == "LONG":
                        target_pct_raw = (take_profit - current_price) / current_price
                        stop_pct_raw = (current_price - stop_loss) / current_price
                    else:
                        target_pct_raw = (current_price - take_profit) / current_price
                        stop_pct_raw = (stop_loss - current_price) / current_price
                else:
                    target_pct_raw = 0
                    stop_pct_raw = 0

                # Подтягиваем cost из risk_config (если модуль доступен)
                try:
                    from risk_metrics import load_risk_config
                    _rc = load_risk_config()
                    overrides = (_rc.get('symbols') or {}).get(args.symbol.replace('/', ''), {})
                    cost_pct_live = overrides.get('round_trip_cost_pct', _rc.get('default_round_trip_cost_pct', 0.002))
                except Exception:
                    cost_pct_live = 0.002

                if target_pct_raw > 0 and stop_pct_raw > 0:
                    net_target_live = target_pct_raw - cost_pct_live
                    net_stop_live = stop_pct_raw + cost_pct_live
                    if net_target_live > 0:
                        ev_calibrated_pct = p_hit_cal * net_target_live - (1 - p_hit_cal) * net_stop_live
                    else:
                        ev_calibrated_pct = 0.0
                    p_be_live = (net_stop_live / (net_stop_live + net_target_live)) if (net_target_live > 0) else 1.0
                else:
                    net_target_live = net_stop_live = ev_calibrated_pct = p_be_live = 0.0

                # Быстрая проверка здоровья фичей на последних наблюдениях
                debug_health = {}
                try:
                    debug_health = _feature_health_debug(df_feat, feature_cols)
                except Exception:
                    debug_health = {}

                signal_log = {
                    "time": str(last_timestamp),
                    "features_used": feature_cols,
                    "timeframe": args.tf,
                    "price": float(current_price),
                    "confidence": float(confidence),
                    "timestamp": last_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "details": {
                        "SHORT": float(short_conf),
                        "HOLD": float(hold_conf),
                        "LONG": float(long_conf)
                    },
                    "debug_proba_raw": [
                        float(last_proba_raw[0]) if isinstance(last_proba_raw, (list, tuple, np.ndarray)) and len(last_proba_raw) > 0 else None,
                        float(last_proba_raw[1]) if isinstance(last_proba_raw, (list, tuple, np.ndarray)) and len(last_proba_raw) > 1 else None,
                        float(last_proba_raw[2]) if isinstance(last_proba_raw, (list, tuple, np.ndarray)) and len(last_proba_raw) > 2 else None,
                    ],
                    "signal": signal,
                    "original_ml_signal": original_signal,  # Добавляем оригинальный ML сигнал
                    "symbol": args.symbol.replace("/", ""),
                    "threshold": float(optimal_threshold),
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                    "time_of_event": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "market_regime": market_regime,
                    "market_regime_details": regime_details,
                    "sl_multiplier": float(used_sl_mul),
                    "tp_multiplier": float(used_tp_mul),
                    "grid_optimizer": optimizer_info,
                    "p_hit_cal": float(p_hit_cal) if isinstance(p_hit_cal, (int,float)) else None,
                    "ev_calibrated_pct": ev_calibrated_pct,
                    "net_target_pct": net_target_live,
                    "net_stop_pct": net_stop_live,
                    "p_be": p_be_live,
                    "cost_pct": cost_pct_live,
                    "adaptation": adapt_info,
                    "smc_filter": {  # Добавляем SMC информацию
                        "enabled": args.enable_smc_filter,
                        "result": smc_result if smc_result else None
                    },
                    "debug_feature_health": {
                        "const_features": debug_health.get("const_features"),
                        "nan_cols": debug_health.get("nan_cols"),
                        "inf_cols": debug_health.get("inf_cols"),
                    }
                }
                
                # Добавим отладочную информацию о NaN/Inf
                signal_log["debug_nan_count_last_row"] = nan_count
                signal_log["debug_inf_count_last_row"] = inf_count

                log_data = json.dumps(signal_log, ensure_ascii=False, indent=2)
                with open("signals.log", "a", encoding="utf-8") as f:
                    f.write("Время события: " + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") + "\n" + log_data + "\n\n")
                
            except Exception as log_e:
                print(f"❌ Ошибка записи в лог: {log_e}")
            
            if signal != last_signal:
                print(f"\n📊 НОВЫЙ СИГНАЛ: {signal}")
                print(f"Время: {last_timestamp}")
                print(f"Цена: ${current_price:,.2f}")
                print(f"Уверенность: {confidence:.1%}")
                print(f"Детали: LONG={long_conf:.1%}, SHORT={short_conf:.1%}, HOLD={hold_conf:.1%}")

                # Добавляем SL/TP информацию для торговых сигналов
                if signal in ["LONG", "SHORT"]:
                    sl_pct = abs(stop_loss-current_price)/current_price*100
                    tp_pct = abs(take_profit-current_price)/current_price*100
                    print(f"Stop Loss: ${stop_loss:,.2f} ({sl_pct:.2f}%)")
                    print(f"Take Profit: ${take_profit:,.2f} ({tp_pct:.2f}%)")
                
                # Звуковой сигнал при торговом сигнале
                if signal in ["LONG", "SHORT"]:
                    try:
                        if winsound is not None:
                            winsound.Beep(1000, 500)
                    except Exception:
                        pass
                
                # Отправляем торговый сигнал в Telegram (только LONG и SHORT)
                if signal in ["LONG", "SHORT"]:
                    try:
                        # Создаем результат в формате, ожидаемом telegram_sender
                        result = {
                            "signal": signal,
                            "symbol": args.symbol.replace("/", ""),
                            "timeframe": args.tf,
                            "price": current_price,
                            "confidence": confidence,
                            "timestamp": last_timestamp,
                            "details": {
                                "LONG": long_conf,
                                "SHORT": short_conf,
                                "HOLD": hold_conf
                            },
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            # Добавим калиброванную вероятность и EV чтобы telegram_sender их отобразил
                            "p_hit_cal": signal_log.get('p_hit_cal'),
                            "ev_calibrated_pct": signal_log.get('ev_calibrated_pct'),
                            "adaptation": signal_log.get('adaptation'),
                            # SMC информация
                            "smc_filter": {
                                "enabled": args.enable_smc_filter,
                                "result": smc_result if smc_result else None
                            },
                            # Добавляем информацию из мета-файла
                            "model_filename": model_filename,
                            "horizon_bars": horizon_bars,
                            "backtest_stats": {
                                "total_trades": backtest_results.get('total_trades', 0),
                                "tp_count": backtest_results.get('tp_count', 0),
                                "sl_count": backtest_results.get('sl_count', 0),
                                "time_exit_count": backtest_results.get('time_exit_count', 0),
                                "win_rate": backtest_results.get('win_rate', 0),
                                "profit_factor": backtest_results.get('profit_factor', 0),
                                "starting_equity": backtest_results.get('starting_equity', 0),
                                "total_pnl_usd": backtest_results.get('total_pnl_usd', 0),
                                "exit_pnl_breakdown": backtest_results.get('exit_pnl_breakdown', {})
                            }
                        }
                        # Применяем фильтры качества перед отправкой
                        skip_reason = None
                        try:
                            from risk_metrics import compute_signal_metrics, load_risk_config
                            _cfg_rm = load_risk_config()
                            metrics_live = compute_signal_metrics(result, _cfg_rm)
                            if metrics_live.get('net_rr_ok') is False:
                                skip_reason = f"net_rr {metrics_live.get('net_rr'):.2f} < min {metrics_live.get('min_net_rr')}"
                            elif metrics_live.get('conf_edge_ok') is False:
                                skip_reason = f"confidence edge {metrics_live.get('conf_edge'):.4f} < min {metrics_live.get('min_conf_edge_bp')}"
                            elif metrics_live.get('cal_ev_ok') is False:
                                skip_reason = "calibrated EV negative"
                        except Exception as _ferr:
                            print(f"⚠️ Ошибка фильтра метрик: {_ferr}")

                        if skip_reason:
                            print(f"🚫 Сигнал НЕ отправлен: {skip_reason}")
                            try:
                                filtered_entry = {
                                    "time": last_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                    "symbol": result['symbol'],
                                    "signal": signal,
                                    "confidence": float(confidence),
                                    "threshold": float(optimal_threshold),
                                    "skip_reason": skip_reason,
                                    "price": float(current_price),
                                    "stop_loss": float(stop_loss),
                                    "take_profit": float(take_profit),
                                    "p_hit_cal": result.get('p_hit_cal'),
                                    "ev_calibrated_pct": result.get('ev_calibrated_pct'),
                                    "adaptation": signal_log.get('adaptation')
                                }
                                with open('filtered_signals.log', 'a', encoding='utf-8') as ff:
                                    ff.write(json.dumps(filtered_entry, ensure_ascii=False) + "\n")
                            except Exception as _logf:
                                print(f"⚠️ Ошибка логирования filtered_signals: {_logf}")
                        else:
                            send_trading_signal(result)
                            print(f"Сигнал отправлен в Telegram")
                            
                    except Exception as e:
                        print(f"❌ Ошибка отправки в Telegram: {e}")
                
                last_signal = signal
            
            # Краткий статус
            interval_text = f"{args.interval} сек" if args.interval < 60 else f"{args.interval//60} мин"
            
            # Форматируем вероятности для всех классов
            proba_text = f"L:{long_conf:.1%} S:{short_conf:.1%} H:{hold_conf:.1%}"
            
            print(f"{datetime.now().strftime('%H:%M:%S')} | {args.symbol} | {signal} | ${current_price:,.2f} | {proba_text}")
            print("Prediction complete.")

            # Пауза между проверками
            time.sleep(args.interval)
            
        except KeyboardInterrupt:
            print(f"\n🛑 Система остановлена пользователем")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            interval_text = f"{args.interval} сек" if args.interval < 60 else f"{args.interval//60} мин"
            print(f"🔄 Повторная попытка через {interval_text}...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
