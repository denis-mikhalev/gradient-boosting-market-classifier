#!/usr/bin/env python3
"""
ModelTrainingMediator.py
Медиатор для автоматического обучения моделей с оптимальными периодами

Функциональность:
1. Проверяет наличие оптимального периода для монеты в results_summary.json
2. Если период найден - запускает обучение
3. Если период не найден - сначала анализирует монету через OptimalPeriodAnalyzer.py
4. После получения оптимального периода запускает обучение модели

Использование:
    python ModelTrainingMediator.py --symbol BTCUSDT
    python ModelTrainingMediator.py --symbol ETHUSDT --timeframe 1h
    python ModelTrainingMediator.py --batch BTCUSDT ETHUSDT ADAUSDT
"""

import json
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

class ModelTrainingMediator:
    def __init__(self, results_dir="optimal_period_analysis/results", verbose=False):
        self.results_dir = results_dir
        self.verbose = verbose
        self.results_summary_file = os.path.join(results_dir, "results_summary.json")
        
        # Пути к скриптам
        self.analyzer_script = "optimal_period_analysis/OptimalPeriodAnalyzer.py"
        self.training_script = "CreateModel-2.py"
        
        # Определяем путь к Python из виртуальной среды
        self.python_executable = self.get_python_executable()
        
        print("🤖 МЕДИАТОР ОБУЧЕНИЯ МОДЕЛЕЙ")
        print("=" * 50)
        print(f"📂 Папка результатов: {self.results_dir}")
        print(f"📊 Файл периодов: {self.results_summary_file}")
        print(f"🐍 Python: {self.python_executable}")
    
    def get_python_executable(self):
        """Определяет путь к Python исполнителю (предпочитает виртуальную среду)"""
        # Проверяем виртуальную среду
        venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            return venv_python
        
        # Если виртуальная среда не найдена, используем системный Python
        return sys.executable
        
    def load_optimal_periods(self):
        """Загружает оптимальные периоды из файла результатов"""
        if not os.path.exists(self.results_summary_file):
            if self.verbose:
                print(f"📂 Файл {self.results_summary_file} не найден")
            return {}
        
        try:
            with open(self.results_summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                periods = data.get("recommendations", {})
                print(f"📋 Загружено {len(periods)} оптимальных периодов")
                return periods
        except Exception as e:
            print(f"❌ Ошибка загрузки периодов: {e}")
            return {}
    
    def get_optimal_period(self, symbol):
        """
        Получает оптимальный период для символа.
        Если период не найден, запускает анализ.
        """
        periods = self.load_optimal_periods()
        
        if symbol in periods:
            period_data = periods[symbol]
            if 'error' in period_data:
                print(f"⚠️  {symbol}: Предыдущий анализ завершился ошибкой: {period_data['error']}")
                print(f"🔄 Повторяем анализ...")
                return self.analyze_symbol(symbol)
            else:
                print(f"✅ {symbol}: Найден оптимальный период {period_data['recommended_days']} дней")
                return period_data
        else:
            print(f"🔍 {symbol}: Период не найден, запускаем анализ...")
            return self.analyze_symbol(symbol)
    
    def analyze_symbol(self, symbol):
        """Запускает анализ символа через OptimalPeriodAnalyzer"""
        print(f"🔄 Анализируем {symbol}...")
        
        try:
            # Запускаем OptimalPeriodAnalyzer для одного символа
            cmd = [
                self.python_executable, 
                self.analyzer_script,
                "--symbol", symbol,
                "--save",
                "--append"  # Добавляем к существующим результатам
            ]
            
            # В Windows лучше не использовать verbose из-за проблем с эмодзи
            # if self.verbose:
            #     cmd.append("--verbose")
            
            if self.verbose:
                print(f"🔧 Команда: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print(f"✅ Анализ {symbol} завершен успешно")
                
                # Перезагружаем результаты
                periods = self.load_optimal_periods()
                if symbol in periods:
                    period_data = periods[symbol]
                    if 'error' not in period_data:
                        print(f"📊 Получен период: {period_data['recommended_days']} дней")
                        return period_data
                    else:
                        print(f"❌ Анализ завершился ошибкой: {period_data['error']}")
                        return None
                else:
                    print(f"❌ Символ {symbol} не найден в результатах после анализа")
                    return None
            else:
                print(f"❌ Ошибка анализа {symbol}:")
                if result.stdout:
                    print(f"stdout: {result.stdout}")
                if result.stderr:
                    print(f"stderr: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Исключение при анализе {symbol}: {e}")
            return None
    
    def train_model(self, symbol, period_data, timeframe="30m"):
        """Запускает обучение модели с оптимальными параметрами"""
        recommended_days = period_data['recommended_days']
        skip_days = period_data.get('skip_days', 0)
        
        print(f"🚀 Запускаем обучение модели для {symbol}")
        print(f"   📅 Период: {recommended_days} дней")
        print(f"   ⏭️  Пропуск: {skip_days} дней")
        print(f"   ⏰ Таймфрейм: {timeframe}")
        
        try:
            # Команда для обучения модели
            cmd = [
                self.python_executable,
                self.training_script,
                "--symbol", symbol,
                "--timeframe", timeframe,
                "--days", str(recommended_days)
            ]
            
            # Добавляем skip_days если он больше 0
            if skip_days > 0:
                # Проверяем, поддерживает ли CreateModel-2.py параметр --skip
                # Если нет, просто показываем рекомендацию
                print(f"💡 Рекомендация: пропустить первые {skip_days} дней данных")
            
            if self.verbose:
                print(f"🔧 Команда обучения: {' '.join(cmd)}")
            
            print(f"⏳ Начинаем обучение модели...")
            
            # Запускаем обучение
            result = subprocess.run(cmd, text=True, encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                print(f"✅ Обучение модели {symbol} завершено успешно!")
                return True
            else:
                print(f"❌ Ошибка обучения модели {symbol} (код возврата: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"❌ Исключение при обучении {symbol}: {e}")
            return False
    
    def process_symbol(self, symbol, timeframe="30m"):
        """Полный цикл: анализ (если нужен) + обучение"""
        print(f"\n🎯 ОБРАБОТКА {symbol}")
        print("-" * 40)
        
        # Получаем оптимальный период
        period_data = self.get_optimal_period(symbol)
        
        if period_data is None:
            print(f"❌ Не удалось получить оптимальный период для {symbol}")
            return False
        
        # Запускаем обучение
        success = self.train_model(symbol, period_data, timeframe)
        
        return success
    
    def process_batch(self, symbols, timeframe="30m"):
        """Обрабатывает список символов"""
        print(f"\n📦 ПАКЕТНАЯ ОБРАБОТКА")
        print(f"🎯 Символы: {', '.join(symbols)}")
        print(f"⏰ Таймфрейм: {timeframe}")
        print("=" * 50)
        
        results = {}
        successful = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Обрабатываем {symbol}...")
            
            success = self.process_symbol(symbol, timeframe)
            results[symbol] = success
            
            if success:
                successful += 1
                print(f"✅ {symbol} - обучение завершено")
            else:
                print(f"❌ {symbol} - обучение неудачно")
        
        # Итоговая статистика
        print(f"\n📊 РЕЗУЛЬТАТЫ ПАКЕТНОЙ ОБРАБОТКИ:")
        print("-" * 40)
        print(f"✅ Успешно: {successful}/{len(symbols)}")
        print(f"❌ Неудачно: {len(symbols) - successful}/{len(symbols)}")
        print(f"📈 Процент успеха: {(successful/len(symbols)*100):.1f}%")
        
        return results

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Медиатор для автоматического обучения моделей с оптимальными периодами'
    )
    
    # Группа для выбора символов
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument('--symbol', '-s', type=str,
                             help='Один символ для обучения (например, BTCUSDT)')
    symbol_group.add_argument('--batch', '-b', type=str, nargs='+',
                             help='Несколько символов (например, --batch BTCUSDT ETHUSDT)')
    symbol_group.add_argument('--file', '-f', type=str,
                             help='Файл со списком символов (один на строку)')
    
    # Дополнительные параметры
    parser.add_argument('--timeframe', '-t', type=str, default='30m',
                       help='Таймфрейм для обучения (по умолчанию: 30m)')
    parser.add_argument('--results-dir', '-r', type=str, default='optimal_period_analysis/results',
                       help='Папка с результатами анализа (по умолчанию: optimal_period_analysis/results)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    return parser.parse_args()

def load_symbols_from_file(filepath):
    """Загружает символы из файла"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            symbols = []
            for line in f:
                symbol = line.strip().upper()
                if symbol and not symbol.startswith('#'):
                    symbols.append(symbol)
        return symbols
    except Exception as e:
        print(f"❌ Ошибка чтения файла {filepath}: {e}")
        return []

def main():
    """Главная функция"""
    args = parse_arguments()
    
    # Создаем медиатор
    mediator = ModelTrainingMediator(
        results_dir=args.results_dir,
        verbose=args.verbose
    )
    
    # Определяем список символов
    symbols = []
    
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.batch:
        symbols = [s.upper() for s in args.batch]
    elif args.file:
        symbols = load_symbols_from_file(args.file)
        if not symbols:
            print(f"❌ Не удалось загрузить символы из файла {args.file}")
            return 1
    
    if not symbols:
        print("❌ Не указаны символы для обработки")
        return 1
    
    # Обрабатываем символы
    if len(symbols) == 1:
        # Один символ
        success = mediator.process_symbol(symbols[0], args.timeframe)
        return 0 if success else 1
    else:
        # Несколько символов
        results = mediator.process_batch(symbols, args.timeframe)
        
        # Возвращаем 0 если все успешно, иначе 1
        all_successful = all(results.values())
        return 0 if all_successful else 1

if __name__ == "__main__":
    exit(main())
