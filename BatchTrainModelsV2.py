#!/usr/bin/env python3
"""
BatchTrainModelsV2 - Улучшенная версия батч обучения с поддержкой:
- JSON формата watchlist.json с периодами и метаданными
- Активации/деактивации монет
- Автоматического определения оптимальных периодов
- Детального логирования процесса
"""

import os
import sys
import time
import subprocess
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

# Импортируем наш парсер
from WatchlistParser import WatchlistParser, WatchlistEntry

@dataclass
class TrainingResult:
    """Результат обучения одной модели"""
    symbol: str
    period: int
    success: bool
    duration: float
    error_message: str = ""
    model_files: List[str] = None

class BatchTrainerV2:
    """Улучшенная система батч обучения моделей"""
    
    def __init__(self, watchlist_path: str = "watchlist.json"):
        self.watchlist_path = watchlist_path
        self.parser = WatchlistParser(watchlist_path)
        self.results: List[TrainingResult] = []
        self.start_time = None
        self.override_period = None  # Переопределение периода из командной строки
        self.backtest_days = None    # Переопределение дней бэктеста
        
    def train_single_model(self, entry: WatchlistEntry) -> TrainingResult:
        """Обучает одну модель"""
        # Используем переопределенный период если он задан
        period = self.override_period if self.override_period else entry.period
        
        print(f"\n🎯 Обучение {entry.symbol} с периодом {period} дней...")
        if entry.comment:
            print(f"   💬 Комментарий: {entry.comment}")
        
        start_time = time.time()
        
        try:
            # Команда для обучения модели через cmd для избежания проблем с эмодзи в PowerShell
            script_dir = os.path.dirname(os.path.abspath(__file__))
            python_args = [sys.executable, "train_model.py",
                           "--symbol", entry.symbol, "--days", str(period), "--silent"]
            
            # Добавляем backtest-days если задан
            if self.backtest_days:
                python_args += ["--backtest-days", str(self.backtest_days)]
            
            cmd = python_args
            
            print(f"   ⚡ Команда: train_model.py --symbol {entry.symbol} --days {period}")
            if self.backtest_days:
                print(f"      📊 Бэктест: {self.backtest_days} дней")
            
            # Запускаем обучение с настройкой кодировки
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir,
                                    env=env, encoding='utf-8', errors='replace')
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ Успешно за {duration:.1f}с")
                
                # Определяем созданные файлы
                model_files = self._find_model_files(entry.symbol)
                
                return TrainingResult(
                    symbol=entry.symbol,
                    period=period,  # Используем фактический период
                    success=True,
                    duration=duration,
                    model_files=model_files
                )
            else:
                error_msg = result.stderr[:200] if result.stderr else "Неизвестная ошибка"
                print(f"   ❌ Ошибка: {error_msg}")
                
                return TrainingResult(
                    symbol=entry.symbol,
                    period=period,  # Используем фактический период
                    success=False,
                    duration=duration,
                    error_message=error_msg
                )
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            print(f"   💥 Исключение: {error_msg}")
            
            return TrainingResult(
                symbol=entry.symbol,
                period=period,  # Используем фактический период
                success=False,
                duration=duration,
                error_message=error_msg
            )
    
    def _find_model_files(self, symbol: str) -> List[str]:
        """Находит файлы модели для символа"""
        models_dir = "models_v2"
        if not os.path.exists(models_dir):
            return []
        
        model_files = []
        for filename in os.listdir(models_dir):
            if symbol in filename and any(filename.startswith(prefix) for prefix in ['xgb_', 'meta_', 'trades_']):
                model_files.append(filename)
        
        return sorted(model_files)
    
    def run_batch_training(self, max_concurrent: int = 1, delay_between: float = 2.0):
        """Запускает батч обучение всех активных моделей"""
        
        # Получаем активные записи
        active_entries = self.parser.get_active_entries()
        
        if not active_entries:
            print("❌ Нет активных монет для обучения!")
            return
        
        self.start_time = time.time()
        
        print("🚀 БАТЧ ОБУЧЕНИЕ МОДЕЛЕЙ V2")
        print("=" * 60)
        print(f"📅 Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 Активных монет: {len(active_entries)}")
        print(f"⏱️ Задержка между обучениями: {delay_between}с")
        print(f"📁 Источник: {self.watchlist_path}")
        
        # Выводим план обучения
        print(f"\n📋 ПЛАН ОБУЧЕНИЯ:")
        total_expected_time = 0
        for i, entry in enumerate(active_entries, 1):
            expected_time = self._estimate_training_time(entry.period)
            total_expected_time += expected_time
            print(f"   {i:2d}. {entry.symbol:10s} - {entry.period:4d} дней (~{expected_time:.0f}мин)")
        
        print(f"\n⏰ Ожидаемое общее время: {total_expected_time:.0f} минут")
        print("=" * 60)
        
        # Запускаем обучение
        successful = 0
        failed = 0
        
        for i, entry in enumerate(active_entries, 1):
            print(f"\n[{i}/{len(active_entries)}] Обучение {entry.symbol}...")
            
            result = self.train_single_model(entry)
            self.results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
            
            # Задержка между обучениями (кроме последнего)
            if i < len(active_entries):
                print(f"   ⏸️ Пауза {delay_between}с...")
                time.sleep(delay_between)
        
        # Финальная сводка
        self._print_final_summary(successful, failed)
        
        # Сохраняем отчет
        self._save_training_report()
    
    def _estimate_training_time(self, period: int) -> float:
        """Оценивает время обучения в минутах на основе периода"""
        # Эмпирические данные: ~1 минута на 500 дней
        base_time = period / 500.0
        return max(1.0, base_time)  # Минимум 1 минута
    
    def _print_final_summary(self, successful: int, failed: int):
        """Выводит финальную сводку"""
        total_time = time.time() - self.start_time
        total_models = successful + failed
        
        print("\n" + "=" * 60)
        print("📊 ИТОГИ БАТЧ ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"✅ Успешно обучено: {successful}/{total_models}")
        print(f"❌ Ошибок: {failed}/{total_models}")
        print(f"📈 Успешность: {successful/total_models*100:.1f}%" if total_models > 0 else "📈 Успешность: 0%")
        print(f"⏰ Общее время: {total_time/60:.1f} минут")
        print(f"⚡ Среднее время на модель: {total_time/total_models:.1f}с" if total_models > 0 else "⚡ Среднее время: 0с")
        
        if successful > 0:
            print(f"\n✅ УСПЕШНО ОБУЧЕННЫЕ МОДЕЛИ:")
            for result in self.results:
                if result.success:
                    files_count = len(result.model_files) if result.model_files else 0
                    print(f"   🔸 {result.symbol:10s} - {result.period:4d} дней ({result.duration:.1f}с, {files_count} файлов)")
        
        if failed > 0:
            print(f"\n❌ ОШИБКИ ОБУЧЕНИЯ:")
            for result in self.results:
                if not result.success:
                    error_short = result.error_message[:50] + "..." if len(result.error_message) > 50 else result.error_message
                    print(f"   🔸 {result.symbol:10s} - {error_short}")
    
    def _save_training_report(self):
        """Сохраняет подробный отчет об обучении"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"batch_training_report_{timestamp}.json"
        
        report_data = {
            "metadata": {
                "timestamp": timestamp,
                "watchlist_file": self.watchlist_path,
                "total_models": len(self.results),
                "successful": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "total_duration": time.time() - self.start_time if self.start_time else 0
            },
            "results": [
                {
                    "symbol": r.symbol,
                    "period": r.period,
                    "success": r.success,
                    "duration": r.duration,
                    "error_message": r.error_message,
                    "model_files": r.model_files or []
                }
                for r in self.results
            ]
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Отчет сохранен: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Ошибка сохранения отчета: {e}")

def main():
    """Главная функция"""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="BatchTrainModelsV2 - Улучшенное батч обучение моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python BatchTrainModelsV2.py                                           # Интерактивный режим
  python BatchTrainModelsV2.py --period 100 --backtest-days 14           # Все монеты с периодом 100
  python BatchTrainModelsV2.py --silent --active-only                    # Только активные, без вопросов
  python BatchTrainModelsV2.py --period 250 --silent --active-only       # Быстрое обучение активных с периодом 250
        """
    )
    
    parser.add_argument('--period', type=int, 
                       help='Период для всех монет (переопределяет значения из watchlist)')
    parser.add_argument('--backtest-days', type=int, 
                       help='Количество дней для бэктеста')
    parser.add_argument('--silent', action='store_true',
                       help='Автоматически запускать без подтверждения')
    parser.add_argument('--active-only', action='store_true',
                       help='Обучать только активные монеты')
    parser.add_argument('--watchlist', default='watchlist.json',
                       help='Путь к файлу watchlist (по умолчанию: watchlist.json)')
    
    args = parser.parse_args()
    
    # Проверяем существование файла
    watchlist_file = args.watchlist
    if not os.path.exists(watchlist_file):
        print(f"❌ Файл {watchlist_file} не найден!")
        print("💡 Создайте файл или укажите правильный путь")
        return
    
    # Создаем и запускаем тренер
    trainer = BatchTrainerV2(watchlist_file)
    
    # Применяем аргументы командной строки
    if args.period:
        trainer.override_period = args.period
        print(f"🔧 Период переопределен на {args.period} дней для всех монет")
    
    if args.backtest_days:
        trainer.backtest_days = args.backtest_days
        print(f"🔧 Количество дней бэктеста установлено: {args.backtest_days}")
    
    # Получаем список монет для обучения
    if args.active_only:
        entries = trainer.parser.get_active_entries()
        print(f"🎯 Режим только активных монет: {len(entries)} монет")
    else:
        entries = trainer.parser.get_active_entries()  # По умолчанию все равно только активные
    
    # Сначала показываем текущую конфигурацию
    if not args.silent:
        trainer.parser.print_summary()
    
    # Проверяем что есть монеты для обучения
    if not entries:
        print("\n❌ Нет активных монет для обучения!")
        return
    
    # Подтверждение запуска (если не silent режим)
    if not args.silent:
        print(f"\n❓ Запустить обучение {len(entries)} моделей? (y/N): ", end="")
        confirmation = input().strip().lower()
        
        if confirmation not in ['y', 'yes', 'да']:
            print("🚫 Обучение отменено")
            return
    else:
        print(f"� Автоматический запуск обучения {len(entries)} моделей...")
    
    # Запускаем обучение
    trainer.run_batch_training()

if __name__ == "__main__":
    main()