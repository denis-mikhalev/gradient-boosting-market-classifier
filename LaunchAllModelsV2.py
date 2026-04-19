#!/usr/bin/env python3
"""
LaunchAllModelsV2 - Улучшенная версия запуска моделей с поддержкой:
- Нового формата watchlist.txt с периодами
- Фильтрации по активным монетам из watchlist
- Выбора лучших моделей для каждого символа
- Детального логирования и мониторинга
"""

import os
import sys
import json
import glob
import time
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Импортируем наш парсер
from WatchlistParser import WatchlistParser, WatchlistEntry

@dataclass
class ModelInfo:
    """Информация о модели"""
    symbol: str
    filepath: str
    period: int
    timeframe: str
    cv_score: float
    win_rate: float
    profit_factor: float
    trades: int
    timestamp: str
    meta_file: str = ""
    trades_file: str = ""

class LauncherV2:
    """Улучшенная система запуска моделей"""
    
    def __init__(self, prediction_interval: int = 60, models_dir: str = "models_v2", 
                 enable_smc_filter: bool = False, smc_min_confluence: int = 3):
        self.python_exe = self._get_python_executable()
        self.predict_script = "Predict-Advanced.py"
        self.models_dir = models_dir
        self.prediction_interval = prediction_interval
        self.enable_smc_filter = enable_smc_filter
        self.smc_min_confluence = smc_min_confluence
        self.watchlist_parser = WatchlistParser()
        
        print(f"🐍 Python executable: {self.python_exe}")
        print(f"📜 Predict script: {self.predict_script}")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"⏱️ Интервал предикшн: {self.prediction_interval//60} мин")
    
    def _get_python_executable(self) -> str:
        """Определяет правильный исполняемый файл Python"""
        venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            return venv_python
        return sys.executable
    
    def find_all_models(self) -> List[ModelInfo]:
        """Находит все доступные модели"""
        if not os.path.exists(self.models_dir):
            print(f"❌ Папка {self.models_dir} не найдена!")
            return []
        
        model_pattern = os.path.join(self.models_dir, "xgb_*.json")
        model_files = glob.glob(model_pattern)
        
        models = []
        
        for model_file in model_files:
            try:
                model_info = self._parse_model_info(model_file)
                if model_info:
                    models.append(model_info)
            except Exception as e:
                print(f"⚠️ Ошибка парсинга {model_file}: {e}")
        
        return models
    
    def _parse_model_info(self, model_file: str) -> Optional[ModelInfo]:
        """Парсит информацию о модели из имени файла и мета-файла"""
        filename = os.path.basename(model_file)
        
        # Парсим имя файла: xgb_SYMBOL_timeframe_period_additional_timestamp.json
        try:
            parts = filename.replace("xgb_", "").replace(".json", "").split("_")
            if len(parts) < 4:
                return None
            
            symbol = parts[0]
            timeframe = parts[1]
            period_str = parts[2]
            
            # Извлекаем период (убираем 'd' в конце)
            period = int(period_str.replace('d', '')) if period_str.endswith('d') else int(period_str)
            
            # Извлекаем timestamp (последняя часть)
            timestamp = parts[-1] if len(parts) >= 5 else ""
            
            # Ищем соответствующий мета-файл
            meta_file = model_file.replace("xgb_", "meta_")
            trades_file = model_file.replace("xgb_", "trades_").replace(".json", ".csv")
            
            # Получаем статистики из мета-файла
            cv_score, win_rate, profit_factor, trades = self._get_model_stats(meta_file)
            
            return ModelInfo(
                symbol=symbol,
                filepath=model_file,
                period=period,
                timeframe=timeframe,
                cv_score=cv_score,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trades=trades,
                timestamp=timestamp,
                meta_file=meta_file,
                trades_file=trades_file
            )
            
        except (ValueError, IndexError) as e:
            print(f"⚠️ Не удалось парсить имя файла {filename}: {e}")
            return None
    
    def _get_model_stats(self, meta_file: str) -> Tuple[float, float, float, int]:
        """Получает статистики модели из мета-файла"""
        try:
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                # Извлекаем статистики из бэктеста
                backtest = meta_data.get('backtest_results', {})
                cv_scores = meta_data.get('cv_scores', [])
                
                # Исправляем обработку cv_scores
                if isinstance(cv_scores, list) and cv_scores:
                    cv_score = sum(cv_scores) / len(cv_scores) * 100
                else:
                    cv_score = 0.0
                
                win_rate = backtest.get('win_rate', 0)
                if win_rate <= 1:
                    win_rate *= 100  # Конвертируем в проценты если нужно
                
                profit_factor = backtest.get('profit_factor', 0)
                trades = backtest.get('total_trades', 0)
                
                return cv_score, win_rate, profit_factor, trades
        except Exception as e:
            print(f"⚠️ Ошибка чтения мета-файла {meta_file}: {e}")
        
        return 0.0, 0.0, 0.0, 0
    
    def filter_models_by_watchlist(self, models: List[ModelInfo], active_only: bool = True) -> List[ModelInfo]:
        """Фильтрует модели по watchlist"""
        if active_only:
            active_symbols = self.watchlist_parser.get_active_symbols()
            filtered = [model for model in models if model.symbol in active_symbols]
        else:
            filtered = models
        
        print(f"🎯 Фильтрация по watchlist: {len(filtered)}/{len(models)} моделей")
        return filtered
    
    def select_best_models(self, models: List[ModelInfo], strategy: str = "latest") -> List[ModelInfo]:
        """Выбирает лучшие модели для каждого символа"""
        symbol_models = {}
        
        # Группируем модели по символам
        for model in models:
            if model.symbol not in symbol_models:
                symbol_models[model.symbol] = []
            symbol_models[model.symbol].append(model)
        
        selected_models = []
        
        for symbol, symbol_model_list in symbol_models.items():
            if strategy == "latest":
                # Выбираем самую новую модель
                best_model = max(symbol_model_list, key=lambda m: m.timestamp)
            elif strategy == "best_performance":
                # Выбираем по лучшему соотношению WR * PF * CV
                best_model = max(symbol_model_list, key=lambda m: m.win_rate * m.profit_factor * m.cv_score)
            elif strategy == "all":
                # Берем все модели
                selected_models.extend(symbol_model_list)
                continue
            else:
                # По умолчанию берем последнюю
                best_model = max(symbol_model_list, key=lambda m: m.timestamp)
            
            selected_models.append(best_model)
        
        print(f"📊 Стратегия выбора '{strategy}': {len(selected_models)} моделей")
        return selected_models
    
    def print_models_summary(self, models: List[ModelInfo]):
        """Выводит сводку по моделям"""
        print("📊 ОБЗОР ВЫБРАННЫХ МОДЕЛЕЙ:")
        print("=" * 80)
        print(f"📈 Найдено: {len(models)} моделей для {len(set(m.symbol for m in models))} символов")
        
        # Группируем по символам
        symbol_groups = {}
        for model in models:
            if model.symbol not in symbol_groups:
                symbol_groups[model.symbol] = []
            symbol_groups[model.symbol].append(model)
        
        for symbol, symbol_models in symbol_groups.items():
            print(f"\\n🔸 {symbol}: {len(symbol_models)} моделей")
            for model in symbol_models:
                pf_str = f"{model.profit_factor:.1f}" if model.profit_factor < 100 else "100.0+"
                print(f"   ✅ {model.timeframe} {model.period}d {model.timestamp} - "
                      f"CV:{model.cv_score:.1f}% WR:{model.win_rate:.1f}% PF:{pf_str} T:{model.trades}")
    
    def launch_models(self, models: List[ModelInfo], dry_run: bool = False):
        """Запускает выбранные модели"""
        if not models:
            print("❌ Нет моделей для запуска!")
            return
        
        print(f"\\n🚀 ЗАПУСК {len(models)} МОДЕЛЕЙ:")
        print("=" * 60)
        
        if dry_run:
            print("🧪 DRY RUN MODE - модели не будут запущены")
        
        successful_launches = 0
        failed_launches = 0
        
        for i, model in enumerate(models, 1):
            print(f"[{i:2d}/{len(models)}] {model.symbol} {model.timeframe} ({model.period}d {model.timestamp})")
            pf_str = f"{model.profit_factor:.1f}" if model.profit_factor < 100 else "100.0+"
            print(f"          ✅ CV:{model.cv_score:.1f}% WR:{model.win_rate:.1f}% PF:{pf_str} Trades:{model.trades}")
            
            if not dry_run:
                success = self._launch_single_model(model)
                if success:
                    successful_launches += 1
                    print(f"          ✅ Запущен")
                else:
                    failed_launches += 1
                    print(f"          ❌ Ошибка запуска")
            else:
                print(f"          🧪 Пропущен (dry run)")
                successful_launches += 1
        
        print("\\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ ЗАПУСКА:")
        print(f"✅ Успешно запущено: {successful_launches}")
        print(f"❌ Ошибки запуска: {failed_launches}")
        print(f"📈 Общий успех: {successful_launches}/{len(models)} ({successful_launches/len(models)*100:.1f}%)")
        
        if not dry_run and successful_launches > 0:
            self._launch_heartbeat_manager()
    
    def _launch_single_model(self, model: ModelInfo) -> bool:
        """Запускает одну модель"""
        try:
            cmd = [
                self.python_exe,
                self.predict_script,
                model.filepath,
                "--interval", str(self.prediction_interval)
            ]
            
            # Добавляем SMC фильтр если включен
            if self.enable_smc_filter:
                cmd.extend(["--enable_smc_filter"])
                cmd.extend(["--smc_min_confluence", str(self.smc_min_confluence)])
            
            # Запускаем в отдельном процессе
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            time.sleep(0.5)  # Небольшая пауза
            return True
            
        except Exception as e:
            print(f"💥 Ошибка запуска {model.symbol}: {e}")
            return False
    
    def _launch_heartbeat_manager(self):
        """Запускает HeartbeatManager"""
        try:
            print("\\n💓 Запуск HeartbeatManager...")
            cmd = [self.python_exe, "HeartbeatManager.py"]
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            print("✅ HeartbeatManager запущен")
        except Exception as e:
            print(f"⚠️ Ошибка запуска HeartbeatManager: {e}")

def main():
    parser = argparse.ArgumentParser(description="Запуск моделей с поддержкой нового watchlist")
    parser.add_argument("--strategy", choices=["latest", "best_performance", "all"], 
                       default="latest", help="Стратегия выбора моделей")
    parser.add_argument("--active-only", action="store_true", default=True,
                       help="Только активные монеты из watchlist")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Показать план без реального запуска")
    parser.add_argument("--interval", type=int, default=60,
                       help="Интервал предикшн в секундах")
    parser.add_argument("--enable_smc_filter", action="store_true",
                       help="Включить SMC (Smart Money Concepts) фильтр для всех моделей")
    parser.add_argument("--smc_min_confluence", type=int, default=3, choices=[2, 3, 4, 5],
                       help="Минимальный confluence score для SMC фильтра (по умолчанию: 3)")
    
    args = parser.parse_args()
    
    print("🚀 ЗАПУСК МОДЕЛЕЙ V2")
    print("=" * 60)
    if args.enable_smc_filter:
        print(f"🛡️ SMC ФИЛЬТР ВКЛЮЧЕН (min confluence: {args.smc_min_confluence}/6)")
    else:
        print("⚠️ SMC ФИЛЬТР ОТКЛЮЧЕН (только ML сигналы)")
    print("=" * 60)
    
    launcher = LauncherV2(
        prediction_interval=args.interval,
        enable_smc_filter=args.enable_smc_filter,
        smc_min_confluence=args.smc_min_confluence
    )
    
    # Показываем текущий watchlist
    launcher.watchlist_parser.print_summary()
    
    # Находим все модели
    all_models = launcher.find_all_models()
    print(f"\\n📁 Найдено моделей в {launcher.models_dir}: {len(all_models)}")
    
    # Фильтруем по watchlist
    filtered_models = launcher.filter_models_by_watchlist(all_models, args.active_only)
    
    # Выбираем лучшие
    selected_models = launcher.select_best_models(filtered_models, args.strategy)
    
    # Показываем сводку
    launcher.print_models_summary(selected_models)
    
    # Запускаем
    if not args.dry_run:
        print(f"\\n❓ Запустить {len(selected_models)} моделей? (y/N): ", end="")
        confirmation = input().strip().lower()
        
        if confirmation in ['y', 'yes', 'да']:
            launcher.launch_models(selected_models, dry_run=False)
        else:
            print("🚫 Запуск отменен")
    else:
        launcher.launch_models(selected_models, dry_run=True)

if __name__ == "__main__":
    main()