#!/usr/bin/env python3
"""
ModelLaunchNotifier - уведомляет о запуске модели с подробной информацией
"""

import json
import os
from datetime import datetime
from telegram_sender import send_telegram_message

def notify_model_launch(symbol, timeframe, model_path, meta_path, used_threshold, threshold_source):
    """
    Отправляет уведомление о запуске модели в консоль и Telegram
    
    Args:
        symbol: торговая пара (например, "ETHUSDT")
        timeframe: таймфрейм (например, "30m") 
        model_path: путь к файлу модели
        meta_path: путь к мета-файлу модели
        used_threshold: используемый порог (например, 0.65)
        threshold_source: источник порога ("MANUAL" или "AUTO")
    """
    
    try:
        # Загружаем метаданные модели
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Извлекаем информацию о модели
        config = meta.get('config', {})
        backtest = meta.get('backtest_results', {})
        
        period = config.get('lookback_days', 'Unknown')
        win_rate = backtest.get('win_rate', 0) * 100 if backtest.get('win_rate') else 0
        profit_factor = backtest.get('profit_factor', 'Unknown')
        total_trades = backtest.get('total_trades', 'Unknown')
        
        # Используем переданные параметры порога (уже определены в Predict-Advanced.py)
        
        # Формируем сообщения
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        console_msg = f"""
🚀 ===============================================
   ЗАПУСК ТОРГОВОЙ МОДЕЛИ: {symbol} {timeframe}
   ===============================================
   📊 Период обучения: {period} дней
   🎯 Порог уверенности: {used_threshold} ({threshold_source})
   📈 Win Rate: {win_rate:.1f}%
   💰 Profit Factor: {profit_factor}
   📋 Сделок в бэктесте: {total_trades}
   📁 Модель: {os.path.basename(model_path)}
   🕐 Время запуска: {timestamp}
   ===============================================
"""

        telegram_msg = f"""🚀 МОДЕЛЬ ЗАПУЩЕНА

{symbol} {timeframe}
📁 Файл: {os.path.basename(model_path)}
 Период: {period}d
 Порог: {used_threshold} ({threshold_source})
 WR: {win_rate:.1f}% | PF: {profit_factor}
 Сделок: {total_trades}
 {timestamp}"""
        
        # Выводим в консоль
        print(console_msg)
        
        # Отправляем в Telegram
        send_telegram_message(telegram_msg)
        
        print(f"✅ Уведомление о запуске {symbol} отправлено в Telegram")
        
    except Exception as e:
        error_msg = f"❌ Ошибка при отправке уведомления о запуске {symbol}: {e}"
        print(error_msg)
        
        # Пытаемся отправить хотя бы базовое уведомление
        try:
            basic_msg = f"🚀 МОДЕЛЬ ЗАПУЩЕНА: {symbol} {timeframe} (детали недоступны)"
            send_telegram_message(basic_msg)
        except:
            pass

def notify_model_error(symbol, timeframe, error_msg):
    """Уведомляет об ошибке запуска модели"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        console_msg = f"""
❌ ===============================================
   ОШИБКА ЗАПУСКА: {symbol} {timeframe}
   ===============================================
   💥 {error_msg}
   🕐 {timestamp}
   ===============================================
"""
        
        telegram_msg = f"""❌ ОШИБКА ЗАПУСКА

{symbol} {timeframe}
💥 {error_msg}
🕐 {timestamp}"""
        
        print(console_msg)
        send_telegram_message(telegram_msg)
        
    except Exception as e:
        print(f"❌ Ошибка при отправке уведомления об ошибке: {e}")

if __name__ == "__main__":
    # Тест
    notify_model_launch(
        "ETHUSDT", 
        "30m", 
        "models_v2/xgb_ETHUSDT_30m_2200d_20250909_002216.json",
        "models_v2/meta_ETHUSDT_30m_2200d_20250909_002216.json",
        0.68,  # Тестовый порог
        "MANUAL"  # Тестовый источник
    )
