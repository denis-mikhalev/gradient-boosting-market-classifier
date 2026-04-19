#!/usr/bin/env python3
"""
OptimalPeriodSearch.py
Поиск оптимального периода обучения для каждой монеты

Функциональность:
1. Обучает модели с разными периодами (50, 100, 150, ... дней)
2. Фиксированный период бэктеста (7 или 14 дней)
3. Сохраняет все результаты threshold optimization для анализа
4. Поддерживает batch обработку всех символов из JSON watchlist
5. Работает только с JSON форматом watchlist

Использование:
    python OptimalPeriodSea        # Затем запускаем очистку убыточных моделей
        unprofitable_script = Path("CleanupUnprofitableModels.py")
        if unprofitable_script.exists():
            print(f"\n[2/2] Очистка убыточных моделей...")
            cmd = [self.python_executable, str(unprofitable_script)]y                                    # Стандартные настройки (watchlist.json)
    python OptimalPeriodSearch.py --backtest-days 7 --max-days 1000  # 7 дней бэктест, до 1000 дней
    python OptimalPeriodSearch.py --step 30                          # Шаг 30 дней вместо 50
    python OptimalPeriodSearch.py --symbols BTCUSDT,ETHUSDT          # Только определенные символы
    python OptimalPeriodSearch.py --watchlist custom_watchlist.json  # Кастомный JSON файл
"""

import os
import sys
import argparse
import subprocess
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ▶️ Звук завершения: используем ту же библиотеку, что и в предикшне (winsound),
# но другой паттерн, чтобы отличать сигнал окончания от сигнала входа.
try:
    import winsound  # Доступно на Windows
except Exception:
    winsound = None

def _play_completion_sound():
    """Проигрывает отличительный звук завершения обучения/очистки.

    В Predict-Advanced используется одиночный сигнал 1000 Гц на 500 мс.
    Здесь используем три коротких восходящих бипа.
    """
    try:
        if winsound is not None:
            winsound.Beep(1200, 200)
            winsound.Beep(1500, 200)
            winsound.Beep(1800, 350)
        else:
            # Фоллбэк для не-Windows сред (терминальный bell)
            print('\a')
    except Exception:
        # Никогда не ломаем основной поток из-за звука
        pass

class OptimalPeriodSearcher:
    def __init__(self, 
                 watchlist_file='watchlist.json',
                 python_executable=None,
                 training_script='CreateModel-2.py',
                 results_dir='optimal_period_analysis',
                 timeframe='30m',
                 backtest_days=14,
                 start_days=50,
                 max_days=1000,
                 step_days=50,
                 single_period=None,
                 silent=True,
                 verbose=False,
                 quality_preset='balanced',
                 min_trades=5,
                 geom_tp=None,
                 geom_sl=None,
                 horizon=None):
        
        self.watchlist_file = watchlist_file
        self.python_executable = python_executable or sys.executable
        self.training_script = training_script
        self.results_dir = Path(results_dir)
        self.timeframe = timeframe
        self.backtest_days = backtest_days
        self.start_days = start_days
        self.max_days = max_days
        self.step_days = step_days
        self.single_period = single_period
        self.silent = silent
        self.quality_preset = quality_preset
        self.verbose = verbose
        self.min_trades = min_trades
        self.geom_tp = geom_tp
        self.geom_sl = geom_sl
        self.horizon = horizon
        
        # Создаем директорию для результатов
        self.results_dir.mkdir(exist_ok=True)
        
        # Файл для сохранения результатов
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = self.results_dir / f'period_search_results_{timestamp}.json'
        
        # Структура для результатов
        self.results = {
            'metadata': {
                'created': timestamp,
                'timeframe': timeframe,
                'backtest_days': backtest_days,
                'start_days': start_days,
                'max_days': max_days,
                'step_days': step_days,
                'total_symbols': 0,
                'total_periods': 0,
                'completed_experiments': 0
            },
            'experiments': {}
        }
        
    def load_symbols(self, custom_symbols=None):
        """Загружает символы из watchlist или использует переданные"""
        if custom_symbols:
            symbols = [s.strip() for s in custom_symbols.split(',')]
            print(f"📋 Используем переданные символы: {len(symbols)}")
        else:
            symbols = self._load_symbols_from_watchlist()
            
        for i, symbol in enumerate(symbols, 1):
            print(f"  {i:2d}. {symbol}")
            
        return symbols
    
    def _load_symbols_from_watchlist(self):
        """Загружает символы из JSON watchlist"""
        if not os.path.exists(self.watchlist_file):
            print(f"❌ Файл watchlist не найден: {self.watchlist_file}")
            return []
        
        try:
            with open(self.watchlist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Извлекаем только активные монеты из JSON структуры
            symbols = []
            coins = data.get('coins', {})
            
            for symbol, coin_data in coins.items():
                if coin_data.get('active', True):  # По умолчанию считаем активной
                    symbols.append(symbol)
            
            print(f"📊 JSON watchlist содержит {len(coins)} монет, {len(symbols)} активных")
            print(f"📋 Загружено из {self.watchlist_file}: {len(symbols)} символов")
            return symbols
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON в {self.watchlist_file}: {e}")
            return []
        except Exception as e:
            print(f"❌ Ошибка чтения {self.watchlist_file}: {e}")
            return []
    
    def get_period_list(self):
        """Генерирует список периодов для тестирования"""
        # Если задан конкретный период, используем только его
        if self.single_period:
            return [self.single_period]
            
        periods = []
        current = self.start_days
        while current <= self.max_days:
            periods.append(current)
            current += self.step_days
        return periods
    
    def extract_threshold_results(self, model_file):
        """Извлекает результаты threshold optimization из meta-файла модели"""
        try:
            # Имя мета-файла основано на модели, но убираем префикс xgb_
            model_name = os.path.basename(model_file)
            # Заменяем xgb_ на meta_
            meta_name = model_name.replace('xgb_', 'meta_')
            meta_file = os.path.join(os.path.dirname(model_file), meta_name)
            
            if not os.path.exists(meta_file):
                print(f"⚠️ Мета-файл не найден: {meta_file}")
                return None
                
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # Проверяем, есть ли данные оптимизации порогов
            if 'threshold_optimization' in meta_data:
                # Новый формат с полными данными оптимизации
                threshold_opt = meta_data['threshold_optimization']
                all_results = threshold_opt.get('all_results', [])
                
                if all_results:
                    # Анализируем результаты для диагностики
                    total_trades_all = sum(result.get('total_trades', 0) for result in all_results)
                    thresholds_with_trades = [r for r in all_results if r.get('total_trades', 0) > 0]
                    
                    # Находим оптимальный порог
                    optimal_threshold = threshold_opt.get('optimal_threshold')
                    
                    # Находим статистику для оптимального порога
                    optimal_result = None
                    for result in all_results:
                        if result.get('threshold') == optimal_threshold:
                            optimal_result = result
                            break
                    
                    # Если не найден, берем лучший по profit_factor
                    if optimal_result is None:
                        optimal_result = max(all_results, key=lambda x: x.get('profit_factor', 0))
                        optimal_threshold = optimal_result.get('threshold')
                        
                        # Улучшенная диагностика
                        if total_trades_all == 0:
                            print(f"⚠️ Модель не генерирует сделки ни при каком пороге (тестировано {len(all_results)} порогов)")
                            print(f"   Возможные причины: слишком короткий период обучения ({meta_data.get('config', {}).get('lookback_days', '?')} дней) или высокие пороги уверенности")
                        elif len(thresholds_with_trades) == 0:
                            print(f"⚠️ Все пороги показали 0 сделок, используем адаптивный порог {optimal_threshold:.3f}")
                        else:
                            print(f"⚠️ Не найден результат для порога {threshold_opt.get('optimal_threshold')}, используем лучший по PF: {optimal_threshold:.3f}")
                            print(f"   Доступно результатов с торговлей: {len(thresholds_with_trades)}/{len(all_results)}")
                    
                    return {
                        'optimal_threshold': optimal_threshold,
                        'all_thresholds': all_results,  # Полезная информация для анализа порогов
                        'backtest_stats': meta_data.get('backtest_results', {})
                    }
            
            # Fallback к старому формату (только backtest_results)
            backtest_data = meta_data.get('backtest_results', {})
            if not backtest_data:
                print(f"⚠️ Нет данных threshold optimization в мета-файле")
                return None
                
            return {
                'optimal_threshold': backtest_data.get('confidence_threshold'),
                'backtest_stats': backtest_data
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка извлечения threshold data: {e}")
            return None
    
    def _find_model_file(self, symbol, days):
        """Находит файл созданной модели"""
        model_pattern = f"xgb_{symbol}_{self.timeframe}_{days}d_bt{self.backtest_days}d_*.json"
        model_files = list(Path('models_v2').glob(model_pattern))
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            threshold_results = self.extract_threshold_results(str(latest_model))
            model_filename = os.path.basename(str(latest_model))
            return True, threshold_results, model_filename
        else:
            print(f"⚠️ Файл модели не найден для {symbol} ({days}d)")
            return False, None, None
    
    def train_single_period(self, symbol, days):
        """Обучает модель для одного символа с определенным периодом"""
        
        # Теперь проверка данных происходит внутри CreateModel-2.py
        cmd = [
            self.python_executable,
            self.training_script,
            '--symbol', symbol,
            '--timeframe', self.timeframe,
            '--days', str(days),
            '--backtest-days', str(self.backtest_days)
        ]
        
        # Добавляем geometry параметры если указаны
        if self.geom_tp:
            cmd.extend(['--geom-tp', self.geom_tp])
        if self.geom_sl:
            cmd.extend(['--geom-sl', self.geom_sl])
        if self.horizon:
            cmd.extend(['--horizon', str(self.horizon)])
        
        if self.silent:
            cmd.append('--silent')
        
        if self.verbose:
            print(f"🔧 Команда: {' '.join(cmd)}")
        
        print(f"⏳ Обучение {symbol} с {days} днями...")
        start_time = time.time()
        
        try:
            # Устанавливаем кодировку UTF-8 для subprocess
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # В verbose режиме показываем вывод в реальном времени
            if self.verbose:
                result = subprocess.run(
                    cmd, 
                    text=True, 
                    encoding='utf-8', 
                    errors='replace',
                    env=env,
                    # Не захватываем output - пускаем в stdout/stderr напрямую
                    capture_output=False
                )
                
                elapsed = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"✅ {symbol} ({days}d) - завершено за {elapsed:.1f}с")
                    
                    # Ищем созданный файл модели
                    success, threshold_results, model_filename = self._find_model_file(symbol, days)
                    return success, threshold_results, elapsed, model_filename
                else:
                    # В verbose режиме ошибка уже показана напрямую
                    print(f"❌ {symbol} ({days}d) - ошибка (код: {result.returncode})")
                    return False, None, elapsed, None
            
            # В обычном режиме захватываем output для анализа
            else:
                result = subprocess.run(
                    cmd, 
                    text=True, 
                    encoding='utf-8', 
                    errors='replace',
                    env=env,
                    capture_output=True
                )
                
                elapsed = time.time() - start_time
            
                if result.returncode == 0:
                    print(f"✅ {symbol} ({days}d) - завершено за {elapsed:.1f}с")
                    
                    # Ищем созданный файл модели
                    success, threshold_results, model_filename = self._find_model_file(symbol, days)
                    return success, threshold_results, elapsed, model_filename
                else:
                    # Проверяем, не связана ли ошибка с недостаточностью данных
                    error_msg = "Неизвестная ошибка"
                    insufficient_data = False
                    
                    # Проверяем stderr и stdout для всех режимов
                    full_text = ""
                    if result.stderr:
                        error_lines = result.stderr.strip().split('\n')
                        error_msg = error_lines[-1] if error_lines else "Неизвестная ошибка"
                        full_text += result.stderr.strip()
                    
                    if result.stdout:
                        full_text += " " + result.stdout.strip()
                    
                    # Проверяем признаки недостаточности данных
                    if any(keyword in full_text.lower() for keyword in ['не найден с достаточными данными', 'недостаточно данных', 'not found with sufficient data']):
                        insufficient_data = True
                    
                    if insufficient_data:
                        print(f"⚠️ ПРОПУСК: {symbol} ({days}d) - недостаточно данных на всех биржах")
                        return 'insufficient_data', None, elapsed, None
                    else:
                        print(f"❌ {symbol} ({days}d) - ошибка (код: {result.returncode})")
                        if result.stderr:
                            print(f"   Ошибка: {result.stderr.strip()}")
                        
                        return False, None, elapsed, None
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ {symbol} ({days}d) - исключение: {str(e)[:100]}")
            return False, None, elapsed, None
    
    def run_search(self, symbols=None):
        """Запускает поиск оптимального периода"""
        symbols = self.load_symbols(symbols)
        if not symbols:
            print("❌ Нет символов для обработки")
            return
            
        periods = self.get_period_list()
        
        print(f"\n🚀 {'ОБУЧЕНИЕ С ФИКСИРОВАННЫМ ПЕРИОДОМ' if self.single_period else 'ПОИСК ОПТИМАЛЬНОГО ПЕРИОДА'}")
        print("=" * 60)
        print(f"📋 Символов: {len(symbols)}")
        if self.single_period:
            print(f"📅 Период: {self.single_period} дней (фиксированный)")
        else:
            print(f"📅 Периоды: {periods}")
        print(f"📊 Бэктест: {self.backtest_days} дней")
        print(f"⏰ Таймфрейм: {self.timeframe}")
        print(f"🔇 Тихий режим: {'Да' if self.silent else 'Нет'}")
        print(f"🎯 Критерии качества: {self.get_preset_description(self.quality_preset)}")
        print(f"💾 Результаты: {self.results_file}")
        print("=" * 60)
        
        # Обновляем метаданные
        self.results['metadata']['total_symbols'] = len(symbols)
        self.results['metadata']['total_periods'] = len(periods)
        
        total_experiments = len(symbols) * len(periods)
        completed = 0
        skipped = 0
        total_start_time = time.time()
        
        for symbol in symbols:
            print(f"\n📈 СИМВОЛ: {symbol}")
            print("-" * 40)
            
            # Инициализируем результаты для символа
            self.results['experiments'][symbol] = {
                'periods': {},
                'completed': 0,
                'skipped': 0,
                'total_periods': len(periods),
                'best_period': None,
                'best_stats': {}
            }
            
            symbol_start_time = time.time()
            symbol_completed = 0
            symbol_skipped = 0
            
            for period in periods:
                print(f"\n[{completed+1}/{total_experiments}] {symbol} - {period} дней")
                
                success, threshold_data, duration, model_filename = self.train_single_period(symbol, period)
                
                # Сохраняем результат
                period_result = {
                    'days': period,
                    'success': success if success != 'insufficient_data' else False,
                    'duration': duration,
                    'threshold_data': threshold_data,
                    'model_file': model_filename,
                    'timestamp': datetime.now().isoformat(),
                    'skipped_reason': 'insufficient_data' if success == 'insufficient_data' else None
                }
                
                self.results['experiments'][symbol]['periods'][str(period)] = period_result
                
                if success == True:
                    symbol_completed += 1
                elif success == 'insufficient_data':
                    symbol_skipped += 1
                    skipped += 1
                    
                completed += 1
                
                # Сохраняем промежуточные результаты
                self.save_results()
                
            # Статистика по символу
            symbol_elapsed = time.time() - symbol_start_time
            self.results['experiments'][symbol]['completed'] = symbol_completed
            self.results['experiments'][symbol]['skipped'] = symbol_skipped
            self.results['experiments'][symbol]['duration'] = symbol_elapsed
            
            if symbol_skipped > 0:
                print(f"\n📊 {symbol} завершен: {symbol_completed}/{len(periods)} периодов за {symbol_elapsed/60:.1f} мин (пропущено: {symbol_skipped})")
            else:
                print(f"\n📊 {symbol} завершен: {symbol_completed}/{len(periods)} периодов за {symbol_elapsed/60:.1f} мин")
        
        # Финальная статистика
        total_elapsed = time.time() - total_start_time
        successful_experiments = sum(1 for symbol_data in self.results['experiments'].values() 
                                   for period_data in symbol_data['periods'].values() 
                                   if period_data['success'])
        self.results['metadata']['completed_experiments'] = completed
        self.results['metadata']['successful_experiments'] = successful_experiments
        self.results['metadata']['skipped_experiments'] = skipped
        self.results['metadata']['total_duration'] = total_elapsed
        
        print(f"\n📊 ПОИСК ЗАВЕРШЕН")
        print("=" * 60)
        print(f"✅ Успешно: {successful_experiments}/{total_experiments}")
        if skipped > 0:
            print(f"⚠️ Пропущено: {skipped}/{total_experiments} (недостаток данных)")
        print(f"⏱️  Общее время: {total_elapsed/60:.1f} минут")
        if successful_experiments > 0:
            print(f"⚡ Среднее время: {total_elapsed/successful_experiments:.1f}с на успешный эксперимент")
        print(f"💾 Результаты сохранены: {self.results_file}")
        
        # Сохраняем финальные результаты
        self.save_results()
        
        print(f"\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print(f"   1. python WatchlistAutoUpdaterJSON.py {self.results_file}")
        print(f"      # Анализирует результаты, обновляет watchlist.json и автоматически очищает неоптимальные модели")
        print(f"")
        print(f"   💡 СОВЕТ: В следующий раз используйте флаг --auto-update для автоматизации:")
        print(f"   python OptimalPeriodSearch.py --auto-update")
        print(f"")
        print(f"   Альтернативно (раздельные шаги):")
        print(f"   • python WatchlistAutoUpdaterJSON.py {self.results_file} --no-cleanup")
        print(f"     # Только обновить watchlist.json без очистки")
        print(f"   • python CleanupNonOptimalModels.py")
        print(f"     # Ручная очистка неоптимальных моделей")
        
    def save_results(self):
        """Сохраняет результаты в JSON файл"""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения результатов: {e}")
            
    def run_auto_update(self, no_cleanup=False, min_trades=5, keep_top_n=1):
        """Автоматически запускает очистку моделей, затем обновление watchlist"""
        print(f"\n🚀 АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ WATCHLIST")
        print("=" * 60)
        
        # ШАГ 1: СНАЧАЛА очищаем модели по качественным критериям
        if not no_cleanup:
            print(f"🎯 Критерии качества: {self.get_preset_description(self.quality_preset)}")
            print("=" * 60)
            self.run_quality_cleanup()
        else:
            print(f"🔧 Очистка по качеству отключена")
        
        # ШАГ 2: ЗАТЕМ ищем оптимальные периоды среди оставшихся качественных моделей
        updater_script = Path("WatchlistAutoUpdaterJSON.py")
        if not updater_script.exists():
            print(f"❌ Скрипт {updater_script} не найден")
            return
            
        print(f"\n📊 ПОИСК ОПТИМАЛЬНЫХ ПЕРИОДОВ")
        print("=" * 60)
        
        # Команда для запуска
        cmd = [self.python_executable, str(updater_script), str(self.results_file), "--force"]
        
        # Добавляем минимум сделок
        cmd.extend(["--min-trades", str(min_trades)])
        
        # Добавляем количество моделей для сохранения
        cmd.extend(["--keep-top-n", str(keep_top_n)])
        
        # Отключаем автоочистку в WatchlistAutoUpdaterJSON - мы уже очистили
        cmd.append("--no-cleanup")
        
        print(f"🔧 Минимум сделок: {min_trades}")
        print(f"🔧 Сохранить топ-N моделей: {keep_top_n}")
        print(f"🔧 Команда: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=False,  # Показываем вывод в реальном времени
                timeout=600  # 10 минут таймаут
            )
            
            if result.returncode == 0:
                print(f"\n✅ Автообновление завершено успешно!")
                
                # ШАГ 3: ФИНАЛЬНАЯ очистка неоптимальных периодов (только среди качественных моделей)
                if not no_cleanup:
                    self.run_period_cleanup()
            else:
                print(f"\n❌ Автообновление завершилось с ошибкой (код {result.returncode})")
                
                # Даже при ошибке обновления запускаем финальную очистку
                if not no_cleanup:
                    print(f"\n🧹 Запускаем финальную очистку неоптимальных периодов...")
                    self.run_period_cleanup()
                
        except subprocess.TimeoutExpired:
            print(f"\n❌ Таймаут автообновления (>10 мин)")
        except Exception as e:
            print(f"\n❌ Ошибка запуска автообновления: {e}")

    def get_preset_description(self, preset: str) -> str:
        """Возвращает описание пресета качества"""
        presets = {
            'conservative': f"📈 КОНСЕРВАТИВНЫЙ: PF≥1.5, WR≥65%, DD≤15%, Return≥5%, Trades≥{self.min_trades}",
            'balanced': f"⚖️ СБАЛАНСИРОВАННЫЙ: PF≥1.2, WR≥55%, DD≤25%, Return≥2%, Trades≥{self.min_trades}", 
            'aggressive': f"🚀 АГРЕССИВНЫЙ: PF≥1.1, WR≥45%, DD≤35%, Return≥1%, Trades≥{self.min_trades}"
        }
        return presets.get(preset, f"❓ НЕИЗВЕСТНЫЙ ПРЕСЕТ: {preset}")

    def run_quality_cleanup(self):
        """ШАГ 1: Очистка моделей по качественным критериям"""
        print(f"\n💎 [ШАГ 1/3] ОЧИСТКА ПО КАЧЕСТВЕННЫМ КРИТЕРИЯМ")
        print("=" * 60)
        print(f"🎯 {self.get_preset_description(self.quality_preset)}")
        print(f"   (Переопределяем min_trades на {self.min_trades})")
        print("=" * 60)
        
        unprofitable_script = Path("CleanupUnprofitableModels.py")
        if unprofitable_script.exists():
            cmd = [
                self.python_executable, 
                str(unprofitable_script), 
                "--preset", self.quality_preset,
                "--min-trades", str(self.min_trades)  # Передаем наш min_trades
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=False,  # Показываем вывод в реальном времени
                    timeout=300,  # 5 минут
                    encoding='utf-8',
                    errors='replace'
                )
                
                if result.returncode == 0:
                    print(f"✅ Очистка по качественным критериям завершена успешно")
                else:
                    print(f"⚠️ Очистка по качеству: код возврата {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                print(f"❌ Таймаут очистки по качеству (>5 мин)")
            except Exception as e:
                print(f"❌ Ошибка очистки по качеству: {e}")
        else:
            print(f"❌ Скрипт {unprofitable_script} не найден")

    def run_period_cleanup(self):
        """ШАГ 3: Очистка неоптимальных периодов среди качественных моделей"""
        print(f"\n🔄 [ШАГ 3/3] ОЧИСТКА НЕОПТИМАЛЬНЫХ ПЕРИОДОВ")
        print("=" * 60)
        
        cleanup_script = Path("CleanupNonOptimalModels.py")
        if cleanup_script.exists():
            cmd = [self.python_executable, str(cleanup_script)]
            
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=False,  # Показываем вывод в реальном времени
                    timeout=300,  # 5 минут
                    encoding='utf-8',
                    errors='replace'
                )
                
                if result.returncode == 0:
                    print(f"✅ Очистка неоптимальных периодов завершена успешно")
                else:
                    print(f"⚠️ Очистка неоптимальных периодов: код возврата {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                print(f"❌ Таймаут очистки периодов (>5 мин)")
            except Exception as e:
                print(f"❌ Ошибка очистки периодов: {e}")
        else:
            print(f"❌ Скрипт {cleanup_script} не найден")

    def run_cleanup_models(self):
        """УСТАРЕВШИЙ МЕТОД: Запускает очистку неоптимальных и убыточных моделей (старая логика)"""
        print(f"\n🧹 АВТОМАТИЧЕСКАЯ ОЧИСТКА МОДЕЛЕЙ (СТАРАЯ ЛОГИКА)")
        print("=" * 50)
        print(f"🎯 Критерии качества: {self.get_preset_description(self.quality_preset)}")
        print("=" * 50)
        
        # Сначала запускаем обычную очистку неоптимальных моделей
        cleanup_script = Path("CleanupNonOptimalModels.py")
        if cleanup_script.exists():
            print(f"[1/2] Очистка неоптимальных моделей...")
            cmd = [self.python_executable, str(cleanup_script)]
            
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,  # Захватываем вывод для контроля
                    timeout=300,  # 5 минут
                    encoding='utf-8',  # Принудительно UTF-8
                    errors='replace'  # Заменяем проблемные символы
                )
                
                if result.returncode == 0:
                    print(f"УСПЕХ: Очистка неоптимальных моделей завершена")
                else:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Очистка неоптимальных моделей: код {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                print(f"ТАЙМАУТ: Очистка неоптимальных моделей (>5 мин)")
            except Exception as e:
                print(f"ОШИБКА: Очистка неоптимальных моделей: {e}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Скрипт {cleanup_script} не найден")
        
        # Затем запускаем очистку убыточных моделей
        unprofitable_script = Path("CleanupUnprofitableModels.py")
        if unprofitable_script.exists():
            print(f"\n💰 [2/2] Очистка убыточных моделей...")
            print(f"    {self.get_preset_description(self.quality_preset)}")
            cmd = [self.python_executable, str(unprofitable_script), "--preset", self.quality_preset]
            
            try:
                result = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,  # Захватываем вывод
                    timeout=300,  # 5 минут
                    encoding='utf-8',  # Принудительно UTF-8
                    errors='replace'  # Заменяем проблемные символы
                )
                
                if result.returncode == 0:
                    print(f"УСПЕХ: Очистка убыточных моделей завершена")
                    
                    # Показываем только итоговую статистику
                    if result.stdout:
                        lines = result.stdout.split('\n')
                        stats_started = False
                        for line in lines:
                            if "ИТОГОВАЯ СТАТИСТИКА:" in line:
                                stats_started = True
                            if stats_started:
                                print(line)
                else:
                    print(f"ОШИБКА: Очистка убыточных моделей: код {result.returncode}")
                    if result.stderr:
                        print(f"ОШИБКА: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                print(f"ТАЙМАУТ: Очистка убыточных моделей (>5 мин)")
            except Exception as e:
                print(f"ОШИБКА: Очистка убыточных моделей: {e}")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Скрипт {unprofitable_script} не найден")
        
        print(f"\nОчистка моделей завершена")

def main():
    parser = argparse.ArgumentParser(
        description='Поиск оптимального периода обучения для каждого символа',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  OptimalPeriodSearch.py                                    # Стандартные настройки (поиск 50-1000 дней)
  OptimalPeriodSearch.py --period 100                       # Обучение только с 100 днями
  OptimalPeriodSearch.py --period 250 --silent              # Обучение с 250 днями в тихом режиме
  OptimalPeriodSearch.py --backtest-days 7 --max-days 1000  # 7 дней бэктест, поиск до 1000 дней
  OptimalPeriodSearch.py --step 30 --start 30               # Шаг 30 дней, начало с 30
  OptimalPeriodSearch.py --symbols BTCUSDT,ETHUSDT          # Только определенные символы
  OptimalPeriodSearch.py --auto-update                      # С автоматическим обновлением watchlist + очистка
  OptimalPeriodSearch.py --auto-update --no-cleanup         # С обновлением watchlist без очистки
  OptimalPeriodSearch.py --auto-update --min-trades 1       # Для новых монет с малым количеством сделок
  
  # Параметры geometry и horizon:
  OptimalPeriodSearch.py --symbols BTCUSDT --geom-tp 1.0,3.0,0.2 --geom-sl 0.6,1.5,0.15 --horizon 12
  OptimalPeriodSearch.py --symbols NEARUSDT --step 100 --start-days 400 --max-days 1830 --geom-tp 1.0,3.0,0.2
        """)
    
    parser.add_argument('--watchlist', '-w', default='watchlist.json',
                        help='JSON файл со списком символов (по умолчанию: watchlist.json)')
    parser.add_argument('--timeframe', '-t', default='30m',
                        help='Таймфрейм для обучения (по умолчанию: 30m)')
    parser.add_argument('--backtest-days', '-b', type=int, default=14,
                        help='Дней для бэктеста (по умолчанию: 14)')
    parser.add_argument('--period', '-p', type=int,
                        help='Обучить с одним конкретным периодом в днях (вместо диапазона)')
    parser.add_argument('--start-days', '-s', type=int, default=50,
                        help='Начальный период в днях (по умолчанию: 50)')
    parser.add_argument('--max-days', '-m', type=int, default=1000,
                        help='Максимальный период в днях (по умолчанию: 1000)')
    parser.add_argument('--step', type=int, default=50,
                        help='Шаг увеличения периода (по умолчанию: 50)')
    parser.add_argument('--symbols', 
                        help='Конкретные символы через запятую (вместо watchlist)')
    parser.add_argument('--silent', '-q', action='store_true',
                        help='Тихий режим (отключает графики)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Подробный вывод процесса обучения')
    parser.add_argument('--results-dir', default='optimal_period_analysis',
                        help='Директория для сохранения результатов')
    parser.add_argument('--auto-update', '-u', action='store_true',
                        help='Автоматически запустить WatchlistAutoUpdaterJSON.py после анализа')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Отключить автоочистку при автообновлении (передается в WatchlistAutoUpdaterJSON.py)')
    parser.add_argument('--min-trades', type=int, default=5,
                        help='Минимум сделок для автообновления (по умолчанию: 5)')
    parser.add_argument('--keep-top-n', type=int, default=1,
                        help='Количество лучших моделей для сохранения (по умолчанию: 1)')
    parser.add_argument('--quality-preset', choices=['conservative', 'balanced', 'aggressive'], 
                        default='balanced',
                        help='Пресет качества моделей для очистки (по умолчанию: balanced)')
    
    # Параметры geometry и horizon (передаются в CreateModel-2.py)
    parser.add_argument('--geom-tp', type=str,
                        help='Диапазон TP: start,end,step (например: 1.0,3.0,0.2)')
    parser.add_argument('--geom-sl', type=str,
                        help='Диапазон SL: start,end,step (например: 0.6,1.5,0.15)')
    parser.add_argument('--horizon', type=int,
                        help='Горизонт прогноза в барах (например: 12 для 15m = 3 часа)')
    
    args = parser.parse_args()
    
    # Создаем searcher
    searcher = OptimalPeriodSearcher(
        watchlist_file=args.watchlist,
        timeframe=args.timeframe,
        backtest_days=args.backtest_days,
        start_days=args.start_days,
        max_days=args.max_days,
        step_days=args.step,
        single_period=args.period,
        silent=args.silent,
        verbose=args.verbose,
        results_dir=args.results_dir,
        quality_preset=args.quality_preset,
        min_trades=args.min_trades,
        geom_tp=args.geom_tp,
        geom_sl=args.geom_sl,
        horizon=args.horizon
    )
    
    # Запускаем поиск
    searcher.run_search(args.symbols)

    # 🔔 Сигнал после завершения ОБУЧЕНИЯ моделей (до автообновления/очистки)
    _play_completion_sound()
    
    # Автоматически запускаем обновление watchlist если указан флаг
    if args.auto_update:
        searcher.run_auto_update(no_cleanup=args.no_cleanup, min_trades=args.min_trades, keep_top_n=args.keep_top_n)


if __name__ == "__main__":
    main()
