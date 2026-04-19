#!/usr/bin/env python3
"""
Расширенный поиск оптимального периода для проблемных монет
Тестирует периоды до 2000 дней и агрессивные пороги
"""

import subprocess
import json
import time
from datetime import datetime

# Расширенная конфигурация для проблемных монет
EXTENDED_CONFIG = {
    "phase_1_extended_periods": {
        "symbols": ["LTCUSDT", "POLUSDT"],
        "periods": [1000, 1200, 1500, 1800, 2000],
        "backtest_days": 14,
        "reason": "Нужны очень большие периоды для накопления сигналов"
    },
    "phase_2_aggressive_thresholds": {
        "symbols": ["SOLUSDT", "ENAUSDT"],
        "periods": [800, 1000, 1200, 1500],
        "backtest_days": 21,  # Увеличиваем бэктест
        "threshold_override": [0.45, 0.48, 0.52],
        "reason": "Консервативные пороги блокируют сигналы"
    },
    "phase_3_bnb_special": {
        "symbols": ["BNBUSDT"],
        "periods": [500, 800, 1000, 1500, 2000],
        "backtest_days": 30,  # Еще больше бэктест
        "threshold_override": [0.40, 0.45, 0.50],
        "reason": "Полное отсутствие сигналов - экстремальные меры"
    }
}

def run_extended_search_for_symbol(symbol, periods, backtest_days, max_days=None):
    """Запускает расширенный поиск для одного символа"""
    
    print(f"\n🔍 Расширенный поиск для {symbol}")
    print("-" * 50)
    
    if not max_days:
        max_days = max(periods)
    
    start_time = time.time()
    
    try:
        # Запускаем OptimalPeriodSearch.py с расширенными параметрами
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            sys.executable,
            "OptimalPeriodSearch.py",
            symbol,
            "--start-days", str(min(periods)),
            "--max-days", str(max_days),
            "--step-days", "200",  # Больший шаг для экономии времени
            "--backtest-days", str(backtest_days)
        ]
        
        print(f"📋 Команда: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Успешно завершено за {execution_time:.1f} сек")
            return True
        else:
            print(f"❌ Ошибка: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"💥 Исключение: {str(e)}")
        return False

def analyze_extended_results():
    """Анализирует результаты расширенного поиска"""
    
    print(f"\n📊 АНАЛИЗ РАСШИРЕННЫХ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # Здесь будет логика анализа новых результатов
    # Аналогично SimpleAnalyzer.py но для расширенного диапазона
    
    extended_results = {}
    
    problematic_symbols = ["LTCUSDT", "POLUSDT", "SOLUSDT", "ENAUSDT", "BNBUSDT"]
    
    for symbol in problematic_symbols:
        # Поиск файлов результатов для каждого символа
        try:
            # Здесь будет логика загрузки и анализа результатов
            print(f"🔍 Анализ {symbol}...")
            # Заглушка для примера
            extended_results[symbol] = {
                "status": "pending_analysis",
                "new_optimal_period": None,
                "improvement": False
            }
        except Exception as e:
            print(f"⚠️ Ошибка анализа {symbol}: {e}")
    
    return extended_results

def create_rehabilitation_report():
    """Создает отчет о реабилитации проблемных монет"""
    
    report = {
        "analysis_date": datetime.now().isoformat(),
        "phase": "extended_period_testing",
        "objective": "Найти оптимальные параметры для проблемных монет",
        "coins_tested": [],
        "results": {},
        "recommendations": {},
        "next_steps": []
    }
    
    return report

def main():
    """Основная функция"""
    
    print("🧪 РАСШИРЕННОЕ ТЕСТИРОВАНИЕ ПРОБЛЕМНЫХ МОНЕТ")
    print("=" * 70)
    
    total_start_time = time.time()
    
    # Показываем план
    print(f"\n📋 ПЛАН ТЕСТИРОВАНИЯ:")
    print("-" * 40)
    
    total_tests = 0
    for phase_name, config in EXTENDED_CONFIG.items():
        symbols = config["symbols"]
        periods = config["periods"]
        total_tests += len(symbols)
        print(f"📊 {phase_name}: {len(symbols)} монет, периоды {min(periods)}-{max(periods)} дней")
    
    print(f"\n🎯 Всего тестов: {total_tests}")
    print(f"⏰ Ожидаемое время: {total_tests * 5} минут")
    
    # Фаза 1: Расширенные периоды
    print(f"\n🚀 ФАЗА 1: РАСШИРЕННЫЕ ПЕРИОДЫ")
    print("=" * 50)
    
    phase1_config = EXTENDED_CONFIG["phase_1_extended_periods"]
    for symbol in phase1_config["symbols"]:
        success = run_extended_search_for_symbol(
            symbol, 
            phase1_config["periods"],
            phase1_config["backtest_days"]
        )
        
        if success:
            print(f"✅ {symbol} - расширенный поиск завершен")
        else:
            print(f"❌ {symbol} - ошибка в расширенном поиске")
    
    # Анализ результатов
    results = analyze_extended_results()
    
    # Создание отчета
    report = create_rehabilitation_report()
    
    total_time = time.time() - total_start_time
    
    print(f"\n📊 ИТОГИ РАСШИРЕННОГО ТЕСТИРОВАНИЯ:")
    print("=" * 50)
    print(f"⏰ Общее время: {total_time:.1f} секунд")
    print(f"📈 Монет протестировано: {len(results)}")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print("-" * 30)
    print("1. 📊 Проанализировать новые результаты в SimpleAnalyzer.py")
    print("2. 🔄 Для монет без улучшений - попробовать другие временные рамки")
    print("3. 🎚️ Рассмотреть адаптивные пороги")
    print("4. 🧮 Возможно добавить дополнительные индикаторы")
    
    print(f"\n📄 Следующий шаг: Запустить SimpleAnalyzer.py для новых результатов")

if __name__ == "__main__":
    main()
