# SMCFilter.py
# Второй слой фильтрации: Smart Money Concepts
# Валидирует ML сигналы через confluence проверки
import pandas as pd
import numpy as np
import ta
from typing import Dict, Optional
import requests
import time

class SMCFilter:
    """
    Фильтр в стиле Smart Money Concepts для валидации ML сигналов
    
    Проверяет 6 confluence условий:
    1. VWAP alignment (цена относительно справедливой стоимости)
    2. Order Blocks (зоны институциональных входов)
    3. Liquidity Sweeps (сбор ликвидности маркет-мейкерами)
    4. Break of Structure (подтверждение смены тренда)
    5. Volume Confirmation (подтверждение объёмом)
    6. BTC Dominance Context (макро-контекст для альткоинов)
    
    Сигнал одобряется если набрано минимум min_confluence_score подтверждений
    """
    
    def __init__(self, min_confluence_score: int = 3):
        """
        Args:
            min_confluence_score: Минимальное количество подтверждений (из 6 возможных)
                                 3 - сбалансированный (рекомендуется)
                                 4 - строгий (меньше сигналов, выше качество)
                                 2 - мягкий (больше сигналов, ниже качество)
        """
        self.min_confluence_score = min_confluence_score
        self.btc_dominance_cache = None
        self.btc_dominance_timestamp = None
    
    def validate_signal(self, 
                       signal: str,
                       df: pd.DataFrame, 
                       ml_confidence: float,
                       symbol: str) -> Dict:
        """
        Валидирует ML сигнал через SMC confluence
        
        Args:
            signal: "LONG", "SHORT" или "HOLD"
            df: DataFrame с OHLCV данными
            ml_confidence: Уверенность ML модели (0-1)
            symbol: Торговый символ (например "BTC/USDT")
        
        Returns:
            {
                'approved': bool,
                'confluence_score': int (0-6),
                'confluence_details': dict,
                'reasons': list[str],
                'recommendation': str
            }
        """
        if signal == "HOLD":
            return {
                'approved': False, 
                'confluence_score': 0, 
                'confluence_details': {},
                'reasons': ['No directional signal'],
                'recommendation': 'HOLD - No ML signal'
            }
        
        # Добавляем необходимые индикаторы если их нет
        df = self._ensure_indicators(df)
        
        # Проверяем каждое SMC условие
        confluence_checks = {
            'vwap_aligned': self._check_vwap_alignment(df, signal),
            'order_block': self._check_order_block(df, signal),
            'liquidity_sweep': self._check_liquidity_sweep(df, signal),
            'break_of_structure': self._check_break_of_structure(df, signal),
            'volume_confirmation': self._check_volume_confirmation(df, signal),
            'btc_context': self._check_btc_dominance_context(signal, symbol)
        }
        
        confluence_score = sum(confluence_checks.values())
        
        # Решение
        approved = confluence_score >= self.min_confluence_score
        reasons = self._build_reasons(confluence_checks, signal)
        recommendation = self._get_recommendation(confluence_score, ml_confidence, approved)
        
        return {
            'approved': approved,
            'confluence_score': confluence_score,
            'confluence_details': confluence_checks,
            'reasons': reasons,
            'recommendation': recommendation
        }
    
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет VWAP если его нет"""
        if 'vwap' not in df.columns:
            try:
                vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
                    high=df['high'], 
                    low=df['low'], 
                    close=df['close'], 
                    volume=df['volume']
                )
                df['vwap'] = vwap_indicator.volume_weighted_average_price()
            except:
                # Fallback: простой VWAP расчёт
                df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _check_vwap_alignment(self, df: pd.DataFrame, signal: str) -> bool:
        """
        VWAP Alignment:
        - LONG: цена выше VWAP (покупатели контролируют)
        - SHORT: цена ниже VWAP (продавцы контролируют)
        
        VWAP показывает "справедливую цену" с учётом объёмов
        """
        last_close = df['close'].iloc[-1]
        last_vwap = df['vwap'].iloc[-1]
        
        if pd.isna(last_vwap):
            return False
        
        if signal == "LONG":
            return last_close > last_vwap
        else:  # SHORT
            return last_close < last_vwap
    
    def _check_order_block(self, df: pd.DataFrame, signal: str) -> bool:
        """
        Order Block Detection:
        Ищем зоны где институциональные игроки размещали крупные ордера
        
        Признаки:
        - Сильная импульсная свеча (тело > 1.5x среднего)
        - Повышенный объём (>1.3x среднего)
        - Разворот после этой свечи
        - Текущая цена вернулась к зоне этой свечи
        """
        if len(df) < 20:
            return False
        
        # Ищем в последних 10 свечах
        lookback = min(10, len(df) - 5)
        avg_body = df['close'].iloc[-20:].sub(df['open'].iloc[-20:]).abs().mean()
        avg_volume = df['volume'].iloc[-20:].mean()
        
        for i in range(-lookback, -2):
            candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            body_size = abs(candle['close'] - candle['open'])
            
            # Проверяем импульсность свечи
            if body_size < avg_body * 1.5:
                continue
            
            volume_spike = candle['volume'] > avg_volume * 1.3
            
            if signal == "LONG":
                # Ищем медвежью импульсную свечу, после которой был разворот вверх
                is_bearish_impulse = candle['close'] < candle['open']
                reversal_up = next_candle['close'] > next_candle['open']
                
                if is_bearish_impulse and reversal_up and volume_spike:
                    # Проверяем что текущая цена около этой зоны
                    current_price = df['close'].iloc[-1]
                    ob_low = candle['low']
                    ob_high = candle['high']
                    
                    # Допускаем 2% отклонение
                    if ob_low <= current_price <= ob_high * 1.02:
                        return True
            
            elif signal == "SHORT":
                # Ищем бычью импульсную свечу, после которой был разворот вниз
                is_bullish_impulse = candle['close'] > candle['open']
                reversal_down = next_candle['close'] < next_candle['open']
                
                if is_bullish_impulse and reversal_down and volume_spike:
                    current_price = df['close'].iloc[-1]
                    ob_low = candle['low']
                    ob_high = candle['high']
                    
                    if ob_low * 0.98 <= current_price <= ob_high:
                        return True
        
        return False
    
    def _check_liquidity_sweep(self, df: pd.DataFrame, signal: str) -> bool:
        """
        Liquidity Sweep:
        Маркет-мейкеры "охотятся" за стопами, пробивая уровни и возвращаясь обратно
        
        Бычий sweep: пробили вниз недавний Low (собрали стопы лонгов), но вернулись выше
        Медвежий sweep: пробили вверх недавний High (собрали стопы шортов), но вернулись ниже
        
        Это сильный сигнал разворота
        """
        if len(df) < 30:
            return False
        
        # Определяем недавние экстремумы (исключая последнюю свечу)
        recent_high = df['high'].iloc[-20:-1].max()
        recent_low = df['low'].iloc[-20:-1].min()
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        if signal == "LONG":
            # Ищем медвежий sweep
            # Предыдущая свеча пробила low, текущая закрылась выше этого уровня
            sweep_occurred = prev_candle['low'] < recent_low
            rejection = last_candle['close'] > recent_low
            
            # Дополнительная проверка: текущая свеча бычья
            current_bullish = last_candle['close'] > last_candle['open']
            
            return sweep_occurred and rejection and current_bullish
        
        elif signal == "SHORT":
            # Ищем бычий sweep
            sweep_occurred = prev_candle['high'] > recent_high
            rejection = last_candle['close'] < recent_high
            
            current_bearish = last_candle['close'] < last_candle['open']
            
            return sweep_occurred and rejection and current_bearish
        
        return False
    
    def _check_break_of_structure(self, df: pd.DataFrame, signal: str) -> bool:
        """
        Break of Structure (BOS):
        Пробой важных структурных уровней подтверждает смену тренда
        
        Определяем resistance/support как среднее из топ-3 экстремумов
        """
        if len(df) < 30:
            return False
        
        # Структурные уровни из последних 20 свечей
        highs = df['high'].iloc[-20:-1]
        lows = df['low'].iloc[-20:-1]
        
        # Resistance = среднее из 3 самых высоких максимумов
        # Support = среднее из 3 самых низких минимумов
        resistance = highs.nlargest(3).mean()
        support = lows.nsmallest(3).mean()
        
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        if signal == "LONG":
            # Пробили resistance вверх
            # Проверяем что пробой произошел недавно (текущая или предыдущая свеча)
            current_breaks = current_close > resistance
            prev_was_below = prev_close <= resistance
            
            return current_breaks and prev_was_below
        
        else:  # SHORT
            # Пробили support вниз
            current_breaks = current_close < support
            prev_was_above = prev_close >= support
            
            return current_breaks and prev_was_above
    
    def _check_volume_confirmation(self, df: pd.DataFrame, signal: str) -> bool:
        """
        Volume Confirmation:
        Объём подтверждает силу движения
        
        Требования:
        - Текущий объём минимум на 20% выше среднего за 20 свечей
        - Для сильных движений объём должен расти
        """
        if len(df) < 20:
            return False
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:].mean()
        
        # Базовая проверка: объём выше среднего
        volume_above_avg = current_volume > avg_volume * 1.2
        
        if not volume_above_avg:
            return False
        
        # Дополнительная проверка: объём растёт последние 2-3 свечи
        recent_volumes = df['volume'].iloc[-3:].values
        volume_increasing = recent_volumes[-1] > recent_volumes[0]
        
        return volume_above_avg and volume_increasing
    
    def _check_btc_dominance_context(self, signal: str, symbol: str) -> bool:
        """
        BTC Dominance Context:
        Макро-контекст для альткоинов
        
        Логика:
        - Для BTC: всегда True (не зависит от доминации)
        - Для альтов: учитываем динамику доминации
          * Доминация 50-60%: нейтрально (разрешаем)
          * Доминация >60%: плохо для лонгов альтов
          * Доминация <50%: хорошо для лонгов альтов (alt season)
        
        API: CoinGecko (бесплатный, кеш 5 минут)
        """
        # Для BTC всегда True
        if 'BTC' in symbol.upper() and 'USDT' in symbol.upper():
            return True
        
        try:
            # Кешируем на 5 минут
            current_time = time.time()
            
            if (self.btc_dominance_cache is None or 
                self.btc_dominance_timestamp is None or
                current_time - self.btc_dominance_timestamp > 300):
                
                # CoinGecko API
                response = requests.get(
                    'https://api.coingecko.com/api/v3/global',
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    btc_dominance = data['data']['market_cap_percentage']['btc']
                    self.btc_dominance_cache = btc_dominance
                    self.btc_dominance_timestamp = current_time
                else:
                    # При ошибке API возвращаем True (не блокируем сигнал)
                    return True
            
            dominance = self.btc_dominance_cache
            
            # Логика для альткоинов
            if 50 <= dominance <= 60:
                return True  # Нейтральная зона
            elif dominance > 60:
                # Высокая доминация - хуже для лонгов альтов, лучше для шортов
                return signal == "SHORT"
            else:  # dominance < 50
                # Низкая доминация - alt season, лучше для лонгов
                return signal == "LONG"
                
        except Exception as e:
            # При любой ошибке считаем контекст нейтральным
            # Не блокируем сигнал из-за недоступности внешнего API
            return True
    
    def _build_reasons(self, checks: Dict[str, bool], signal: str) -> list:
        """Формирует человеко-читаемые причины для лога"""
        reasons = []
        
        # VWAP
        if checks['vwap_aligned']:
            if signal == "LONG":
                reasons.append("✅ Цена выше VWAP (покупатели контролируют)")
            else:
                reasons.append("✅ Цена ниже VWAP (продавцы контролируют)")
        else:
            reasons.append("❌ VWAP не поддерживает направление")
        
        # Order Block
        if checks['order_block']:
            reasons.append(f"✅ Найден {'бычий' if signal=='LONG' else 'медвежий'} Order Block")
        else:
            reasons.append(f"❌ Нет {'бычьего' if signal=='LONG' else 'медвежьего'} Order Block")
        
        # Liquidity Sweep
        if checks['liquidity_sweep']:
            reasons.append("✅ Ликвидность собрана (sweep pattern)")
        else:
            reasons.append("❌ Нет liquidity sweep")
        
        # Break of Structure
        if checks['break_of_structure']:
            reasons.append("✅ Break of Structure подтверждён")
        else:
            reasons.append("❌ Нет Break of Structure")
        
        # Volume
        if checks['volume_confirmation']:
            reasons.append("✅ Объём подтверждает движение")
        else:
            reasons.append("⚠️ Слабый объём (риск ложного пробоя)")
        
        # BTC Dominance
        if checks['btc_context']:
            reasons.append("✅ BTC dominance context благоприятен")
        else:
            reasons.append("⚠️ BTC dominance context неблагоприятен")
        
        return reasons
    
    def _get_recommendation(self, confluence_score: int, ml_confidence: float, approved: bool) -> str:
        """Формирует итоговую рекомендацию"""
        if not approved:
            return f"❌ СИГНАЛ ОТКЛОНЁН (SMC: {confluence_score}/6, требуется ≥{self.min_confluence_score})"
        
        if confluence_score >= 5:
            return f"🔥 СИЛЬНЫЙ СИГНАЛ (SMC: {confluence_score}/6, ML: {ml_confidence:.1%})"
        elif confluence_score == 4:
            return f"✅ ХОРОШИЙ СИГНАЛ (SMC: {confluence_score}/6, ML: {ml_confidence:.1%})"
        elif confluence_score == 3:
            return f"⚠️ УМЕРЕННЫЙ СИГНАЛ (SMC: {confluence_score}/6, ML: {ml_confidence:.1%})"
        else:
            return f"⚡ СЛАБЫЙ СИГНАЛ (SMC: {confluence_score}/6, ML: {ml_confidence:.1%})"


# ======================
# Тестирование
# ======================
def test_smc_filter():
    """Быстрый тест SMC фильтра на реальных данных"""
    print("🧪 Тестирование SMC Filter...")
    print("="*60)
    
    try:
        import ccxt
        
        # Получаем свежие данные
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '30m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Загружено {len(df)} свечей BTC/USDT (30m)")
        print(f"Последняя цена: ${df['close'].iloc[-1]:,.2f}")
        print()
        
        # Создаём фильтр
        smc = SMCFilter(min_confluence_score=3)
        
        # Тестируем LONG
        print("🔍 Тест LONG сигнала:")
        print("-"*60)
        result_long = smc.validate_signal(
            signal="LONG",
            df=df,
            ml_confidence=0.75,
            symbol="BTC/USDT"
        )
        
        print(f"Одобрен: {'✅ ДА' if result_long['approved'] else '❌ НЕТ'}")
        print(f"Confluence Score: {result_long['confluence_score']}/6")
        print(f"\n{result_long['recommendation']}")
        print(f"\nДетали:")
        for reason in result_long['reasons']:
            print(f"  {reason}")
        
        print("\n" + "="*60)
        
        # Тестируем SHORT
        print("🔍 Тест SHORT сигнала:")
        print("-"*60)
        result_short = smc.validate_signal(
            signal="SHORT",
            df=df,
            ml_confidence=0.72,
            symbol="BTC/USDT"
        )
        
        print(f"Одобрен: {'✅ ДА' if result_short['approved'] else '❌ НЕТ'}")
        print(f"Confluence Score: {result_short['confluence_score']}/6")
        print(f"\n{result_short['recommendation']}")
        print(f"\nДетали:")
        for reason in result_short['reasons']:
            print(f"  {reason}")
        
        print("\n" + "="*60)
        print("✅ Тест завершён успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_smc_filter()
