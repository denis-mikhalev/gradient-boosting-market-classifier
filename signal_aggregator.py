# Улучшенный SignalAggregator.py с поддержкой Telegram
import json
import glob
import time
import os
import logging
from datetime import datetime
import requests
from dotenv import load_dotenv

# Метрики риска / EV
try:
    from risk_metrics import compute_signal_metrics, format_metrics_block
except Exception:
    compute_signal_metrics = None
    format_metrics_block = None

# Загружаем переменные окружения
load_dotenv()

# Настройки
SIGNALS_DIR = "signals"  # папка для хранения сигналов
CHECK_INTERVAL = 300  # проверка каждые 5 минут (300 секунд)

# Telegram settings из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Проверяем, что секреты загружены
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Не найдены Telegram credentials в .env файле!")

# Настройка логирования
logging.basicConfig(
    filename='signal_aggregator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Создаем папку для сигналов, если её нет
os.makedirs(SIGNALS_DIR, exist_ok=True)

# Для отслеживания уже обработанных сигналов
processed_signals = set()


def send_telegram_notification(message):
    """Отправка уведомления в Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info("Уведомление в Telegram отправлено")
        else:
            logging.error(f"Ошибка отправки в Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Ошибка Telegram: {e}")

def send_email_notification(subject, message):
    """Заготовка для отправки email"""
    # TODO: Добавить код для отправки email
    logging.info(f"Email отправлен: {subject}")

def process_signals():
    """Обработка сигналов из JSON файлов"""
    new_signals = []
    
    # Ищем все JSON файлы в папке сигналов
    for signal_file in glob.glob(os.path.join(SIGNALS_DIR, "*.json")):
        try:
            # Получаем время модификации файла
            mod_time = os.path.getmtime(signal_file)
            
            # Обрабатываем только свежие файлы (не старше 30 минут)
            if time.time() - mod_time > 1800:  # 30 минут
                continue
                
            # Создаем уникальный идентификатор файла + время модификации
            signal_id = f"{signal_file}_{mod_time}"
            
            # Пропускаем уже обработанные сигналы
            if signal_id in processed_signals:
                continue
                
            with open(signal_file, "r") as f:
                signal_data = json.load(f)
                
            # Проверяем, есть ли активный сигнал
            if signal_data.get("signal") in ["LONG", "SHORT"]:
                new_signals.append(signal_data)
                processed_signals.add(signal_id)
                logging.info(f"Найден новый торговый сигнал: {signal_file} - {signal_data.get('signal')}")
            else:
                # Добавляем в processed, чтобы не проверять HOLD сигналы повторно
                processed_signals.add(signal_id)
                
        except Exception as e:
            logging.error(f"Ошибка обработки {signal_file}: {e}")
    
    return new_signals

def format_signal_message(signal):
    """Форматирование сообщения о сигнале (расширено с метриками)."""
    signal_type = signal.get("signal", "HOLD")
    symbol = signal.get("symbol", "?")
    price = signal.get("close") or signal.get("price")
    probs = signal.get("probs", {})
    confidence = max(probs.values()) if probs else signal.get('confidence', 0.0)
    tp = signal.get("take_profit")
    sl = signal.get("stop_loss")
    timeframe = signal.get('timeframe', 'Unknown')
    model_filename = signal.get('model_filename', 'Unknown')
    
    emoji = "🟢" if signal_type == "LONG" else ("🔴" if signal_type == "SHORT" else "⚪")
    message = (
        f"{emoji} <b>{signal_type} сигнал: {symbol}</b>\n"
        f"Таймфрейм: {timeframe}\n"
        f"Цена: {price}\n"
        f"Уверенность: {confidence:.2%}\n"
    )
    
    # Добавляем имя модели
    if model_filename != 'Unknown':
        message += f"Модель: <code>{model_filename}</code>\n"
    
    if tp and sl:
        message += f"Take Profit: {tp}\nStop Loss: {sl}\n"
    
    # Добавляем статистику бэктеста
    backtest_stats = signal.get('backtest_stats', {})
    if backtest_stats:
        total_trades = backtest_stats.get('total_trades', 0)
        tp_count = backtest_stats.get('tp_count', 0)
        sl_count = backtest_stats.get('sl_count', 0)
        time_exit_count = backtest_stats.get('time_exit_count', 0)
        win_rate = backtest_stats.get('win_rate', 0)
        profit_factor = backtest_stats.get('profit_factor', 0)
        starting_equity = backtest_stats.get('starting_equity', 0)
        total_pnl_usd = backtest_stats.get('total_pnl_usd', 0)
        
        # Данные по выходам
        exit_breakdown = backtest_stats.get('exit_pnl_breakdown', {})
        by_exit = exit_breakdown.get('by_exit_reason', {})
        
        # Данные по Stop Loss
        sl_data = by_exit.get('stop_loss', {})
        sl_gross_loss = sl_data.get('gross_loss', 0)
        
        # Данные по Take Profit
        tp_data = by_exit.get('take_profit', {})
        tp_gross_profit = tp_data.get('gross_profit', 0)
        
        # Данные по Time Exit
        te_data = by_exit.get('time_exit', {})
        te_gross_profit = te_data.get('gross_profit', 0)
        te_gross_loss = te_data.get('gross_loss', 0)
        te_net = te_gross_profit - te_gross_loss
        
        if total_trades > 0:
            message += f"\nСтатистика бэктеста:\n"
            message += f"Всего: {total_trades} | TP: {tp_count} | SL: {sl_count} | Timeout: {time_exit_count}\n"
            message += f"WinRate: {win_rate:.1%} | PF: {profit_factor:.2f}\n"
            message += f"TP Profit: +${tp_gross_profit:,.0f} | SL Loss: -${sl_gross_loss:,.0f}\n"
            message += f"Timeout P&L: {'+-'[te_net<0]}${abs(te_net):,.0f}\n"
            message += f"Депозит: ${starting_equity:,.0f} → Total P&L: {'+-'[total_pnl_usd<0]}${abs(total_pnl_usd):,.0f}\n"
    
    # Добавляем информацию о таймауте
    horizon_bars = signal.get('horizon_bars', 6)
    if horizon_bars:
        try:
            from telegram_sender import get_timeout_info
            timeout_info = get_timeout_info(timeframe, horizon_bars)
            if timeout_info:
                message += f"Выход по таймауту: {timeout_info}\n"
        except Exception:
            pass
    
    message += f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Добавляем блок метрик
    if signal_type in ("LONG", "SHORT") and compute_signal_metrics and format_metrics_block:
        metrics_input = {
            'signal': signal_type,
            'symbol': symbol,
            'price': price,
            'take_profit': tp,
            'stop_loss': sl,
            'confidence': confidence
        }
        try:
            m = compute_signal_metrics(metrics_input)
            block = format_metrics_block(m)
            if block:
                message += block
        except Exception as e:
            message += f"\n⚠ Metrics error: {e}"
    return message


def main():
    """Основной цикл работы агрегатора"""
    
    logging.info("Запуск SignalAggregator...")
    print("SignalAggregator запущен. Ожидание сигналов...")
    
    
    while True:
        try:
            # Обработка сигналов
            new_signals = process_signals()
            
            # Отправка уведомлений о новых сигналах
            for signal in new_signals:
                message = format_signal_message(signal)
                print(message.replace('<b>', '').replace('</b>', ''))
                
                # Отправка уведомлений
                send_telegram_notification(message)
                
                # Если хотите также отправлять email
                # send_email_notification(f"{signal['signal']} сигнал: {signal['symbol']}", message)
            
            # Очистка старых записей в processed_signals (каждый час)
            if len(processed_signals) > 1000:
                processed_signals.clear()
                
            # Пауза перед следующей проверкой
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("Остановка SignalAggregator...")
            break
        except Exception as e:
            logging.error(f"Ошибка в основном цикле: {e}")
            time.sleep(CHECK_INTERVAL)  # Продолжаем работу даже при ошибке

if __name__ == "__main__":
    main()