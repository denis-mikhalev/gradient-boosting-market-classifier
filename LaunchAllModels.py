#!/usr/bin/env python3
"""
Запуск ВСЕХ доступных моделей без фильтрации по качеству.

Этот скрипт автоматически запускает все модели, найденные в папке models_v2/,
даже если у них низкий Win Rate или другие показатели качества.
Полезно для реального тестирования множества моделей с разными параметрами.

Автор: CryptoAI Team
"""

import os
import sys
import json
import glob
import time
import argparse
import subprocess
from datetime import datetime

class AllModelsLauncher:
    def __init__(self, prediction_interval=60, enable_smc_filter=False, smc_min_confluence=3):
        # Пути к исполняемым файлам и скриптам
        self.python_exe = sys.executable
        self.predict_script = "Predict-Advanced.py"
        self.models_dir = "models_v2"
        self.prediction_interval = prediction_interval
        self.enable_smc_filter = enable_smc_filter
        self.smc_min_confluence = smc_min_confluence
        
        # Убедимся, что используем правильный Python из виртуальной среды
        venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            self.python_exe = venv_python
        
        print(f"🐍 Python executable: {self.python_exe}")
        print(f"📜 Predict script: {self.predict_script}")
        print(f"📁 Models directory: {self.models_dir}")

    def find_all_models(self):
        """Найти ВСЕ модели в папке models_v2 без фильтрации"""
        if not os.path.exists(self.models_dir):
            print(f"❌ Папка {self.models_dir} не найдена!")
            return []
        
        # Ищем все .json файлы моделей (начинающиеся с xgb_)
        model_pattern = os.path.join(self.models_dir, "xgb_*.json")
        model_files = glob.glob(model_pattern)
        
        models = []
        
        for model_file in model_files:
            try:
                # Извлекаем информацию из имени файла
                basename = os.path.basename(model_file)
                # Формат: xgb_SYMBOL_TIMEFRAME_PERIOD_TIMESTAMP.json
                parts = basename.replace("xgb_", "").replace(".json", "").split("_")
                
                if len(parts) < 3:
                    print(f"⚠️ Неожиданный формат файла: {basename}")
                    continue
                
                symbol = parts[0]
                timeframe = parts[1]
                
                # Находим period (может быть с bt информацией)
                period_info = ""
                timestamp = ""
                
                # Ищем период и timestamp
                for i, part in enumerate(parts[2:], start=2):
                    if part.isdigit() and len(part) == 14:  # timestamp
                        timestamp = part
                        period_info = "_".join(parts[2:i])
                        break
                    elif i == len(parts) - 1:  # последняя часть
                        if part.isdigit() and len(part) == 14:
                            timestamp = part
                            period_info = "_".join(parts[2:i])
                        else:
                            period_info = "_".join(parts[2:])
                
                # Ищем соответствующий meta файл
                meta_pattern = model_file.replace("xgb_", "meta_")
                if not os.path.exists(meta_pattern):
                    print(f"⚠️ Meta файл не найден для {basename}")
                    continue
                
                # Создаем display_symbol (без слэша для CCXT)
                display_symbol = symbol
                if "/" not in symbol and symbol.endswith("USDT"):
                    # Конвертируем BTCUSDT -> BTC/USDT для отображения
                    base = symbol.replace("USDT", "")
                    display_symbol = f"{base}/USDT"
                
                model_info = {
                    "model_file": model_file,
                    "meta_file": meta_pattern,
                    "symbol": symbol,
                    "display_symbol": display_symbol,
                    "timeframe": timeframe,
                    "period": period_info,
                    "timestamp": timestamp,
                    "full_name": basename
                }
                
                models.append(model_info)
                
            except Exception as e:
                print(f"❌ Ошибка обработки {model_file}: {e}")
                continue
        
        # Сортируем модели по символу и времени создания
        models.sort(key=lambda x: (x["symbol"], x["timestamp"]))
        
        return models

    def get_model_info(self, model):
        """Получить подробную информацию о модели"""
        try:
            with open(model["meta_file"], 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # CV Scores
            cv_scores = meta_data.get('cv_scores', [])
            avg_accuracy = 0.0
            avg_f1 = 0.0
            
            if cv_scores:
                accuracies = [fold[0] for fold in cv_scores if len(fold) >= 1]
                f1_scores = [fold[1] for fold in cv_scores if len(fold) >= 2]
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                if f1_scores:
                    avg_f1 = sum(f1_scores) / len(f1_scores)
            
            # Backtest Results
            backtest_results = meta_data.get('backtest_results', {})
            win_rate = backtest_results.get('win_rate', 0.0)
            profit_factor = backtest_results.get('profit_factor', 1.0)
            total_trades = backtest_results.get('total_trades', 0)
            
            # Обрабатываем inf значения
            if isinstance(profit_factor, str) and profit_factor.lower() == 'inf':
                profit_factor = 100.0
            elif profit_factor == float('inf'):
                profit_factor = 100.0
            
            # Optimal threshold
            optimal_threshold = meta_data.get('optimal_threshold', 0.7)
            
            return {
                "cv_accuracy": avg_accuracy,
                "cv_f1": avg_f1,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": total_trades,
                "optimal_threshold": optimal_threshold,
                "has_backtest": bool(backtest_results)
            }
            
        except Exception as e:
            print(f"❌ Ошибка чтения метаданных {model['symbol']}: {e}")
            return {
                "cv_accuracy": 0.0,
                "cv_f1": 0.0,
                "win_rate": 0.0,
                "profit_factor": 1.0,
                "total_trades": 0,
                "optimal_threshold": 0.7,
                "has_backtest": False
            }

    def group_models_by_symbol(self, models):
        """Группировать модели по символам"""
        grouped = {}
        for model in models:
            symbol = model["symbol"]
            if symbol not in grouped:
                grouped[symbol] = []
            grouped[symbol].append(model)
        return grouped

    def display_models_overview(self, models):
        """Показать обзор всех найденных моделей"""
        print("\n📊 ОБЗОР ВСЕХ ДОСТУПНЫХ МОДЕЛЕЙ:")
        print("=" * 80)
        
        grouped = self.group_models_by_symbol(models)
        
        total_models = len(models)
        total_symbols = len(grouped)
        
        print(f"📈 Найдено: {total_models} моделей для {total_symbols} символов")
        print()
        
        for symbol, symbol_models in sorted(grouped.items()):
            print(f"🔸 {symbol}: {len(symbol_models)} моделей")
            
            for model in symbol_models:
                info = self.get_model_info(model)
                
                # Формируем краткое описание
                period_short = model["period"].replace("_", " ")
                timestamp_short = model["timestamp"][:8]  # YYYYMMDD
                
                # Статус качества
                cv_acc = info["cv_accuracy"] * 100
                quality_status = "✅" if cv_acc >= 35 else "⚠️" if cv_acc >= 25 else "❌"
                
                # Информация о бэктесте
                if info["has_backtest"]:
                    backtest_info = f"WR:{info['win_rate']:.1%} PF:{info['profit_factor']:.1f} T:{info['total_trades']}"
                else:
                    backtest_info = "No backtest"
                
                print(f"   {quality_status} {model['timeframe']} {period_short} ({timestamp_short}) - "
                      f"CV:{cv_acc:.1f}% {backtest_info}")
        
        print("=" * 80)

    def launch_all_models(self):
        """Запустить ВСЕ найденные модели"""
        print("🚀 ЗАПУСК ВСЕХ ДОСТУПНЫХ МОДЕЛЕЙ")
        print("=" * 60)
        print("⚠️  ВНИМАНИЕ: Запускаются ВСЕ модели без фильтрации по качеству!")
        print("🎯 Это полезно для реального тестирования разных стратегий")
        if self.enable_smc_filter:
            print(f"🛡️ SMC ФИЛЬТР ВКЛЮЧЕН (min confluence: {self.smc_min_confluence}/6)")
        else:
            print("⚠️ SMC ФИЛЬТР ОТКЛЮЧЕН (только ML сигналы)")
        print("=" * 60)
        
        all_models = self.find_all_models()
        if not all_models:
            print("❌ Модели не найдены!")
            return
        
        print(f"📊 Найдено {len(all_models)} моделей")
        
        # Показываем обзор
        self.display_models_overview(all_models)
        
        # Информация о запуске
        print(f"\n🚀 Запускаем все {len(all_models)} моделей...")
        print("⚠️  Каждая модель откроет отдельное окно терминала")
        
        launched_count = 0
        failed_count = 0
        
        print(f"\n🚀 ЗАПУСК {len(all_models)} МОДЕЛЕЙ:")
        print("=" * 60)
        
        for i, model in enumerate(all_models, 1):
            info = self.get_model_info(model)
            
            # Информация о запуске
            period_display = model["period"].replace("_", " ")
            print(f"[{i:2d}/{len(all_models)}] {model['symbol']} {model['timeframe']} ({period_display})")
            
            # Показываем качество модели
            cv_acc = info["cv_accuracy"] * 100
            quality_indicator = "✅" if cv_acc >= 35 else "⚠️" if cv_acc >= 25 else "❌"
            
            if info["has_backtest"]:
                print(f"          {quality_indicator} CV:{cv_acc:.1f}% WR:{info['win_rate']:.1%} "
                      f"PF:{info['profit_factor']:.1f} Trades:{info['total_trades']}")
            else:
                print(f"          {quality_indicator} CV:{cv_acc:.1f}% (100% training)")
            
            # Команда запуска
            cmd = [
                "start",
                f'"{model["symbol"]} {model["timeframe"]} ({period_display})"',
                "cmd", "/k",
                f'"{self.python_exe}"',
                self.predict_script,
                "--meta", model["meta_file"],
                "--model", model["model_file"], 
                "--symbol", model["display_symbol"],
                "--tf", model["timeframe"],
                "--interval", str(self.prediction_interval)
            ]
            
            # Добавляем SMC фильтр если включен
            if self.enable_smc_filter:
                cmd.extend(["--enable_smc_filter"])
                cmd.extend(["--smc_min_confluence", str(self.smc_min_confluence)])
            
            try:
                subprocess.run(" ".join(cmd), shell=True, check=True)
                print(f"          ✅ Запущен")
                launched_count += 1
                time.sleep(0.5)  # Короткая пауза между запусками
                
            except subprocess.CalledProcessError as e:
                print(f"          ❌ Ошибка: {e}")
                failed_count += 1
        
        # Запускаем HeartbeatManager
        print(f"\n💓 Запуск HeartbeatManager...")
        try:
            heartbeat_cmd = [
                "start", '"Heartbeat Manager"', "cmd", "/k",
                f'"{self.python_exe}"', "HeartbeatManager.py"
            ]
            subprocess.run(" ".join(heartbeat_cmd), shell=True, check=True)
            print("✅ HeartbeatManager запущен")
        except Exception as e:
            print(f"❌ Ошибка запуска HeartbeatManager: {e}")
        
        # Итоги
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ ЗАПУСКА:")
        print(f"✅ Успешно запущено: {launched_count}")
        print(f"❌ Ошибки запуска: {failed_count}")
        print(f"📈 Общий успех: {launched_count}/{len(all_models)} ({launched_count/len(all_models)*100:.1f}%)")
        print("="*60)
        
        if launched_count > 0:
            print(f"\n🎉 Запущено {launched_count} моделей!")
            print("📱 Все терминалы будут работать параллельно")
            print("💡 Используйте model_thresholds.json для настройки порогов")
            print("📊 Мониторинг через HeartbeatManager и SignalAggregator")
        else:
            print("\n❌ Не удалось запустить ни одной модели!")

    def show_models_list(self):
        """Показать список моделей без запуска"""
        all_models = self.find_all_models()
        if not all_models:
            print("❌ Модели не найдены!")
            return
        
        self.display_models_overview(all_models)
        
        # Статистика по качеству
        good_models = 0
        medium_models = 0
        poor_models = 0
        no_backtest = 0
        
        for model in all_models:
            info = self.get_model_info(model)
            cv_acc = info["cv_accuracy"] * 100
            
            if not info["has_backtest"]:
                no_backtest += 1
            elif cv_acc >= 35:
                good_models += 1
            elif cv_acc >= 25:
                medium_models += 1
            else:
                poor_models += 1
        
        print(f"\n📈 СТАТИСТИКА КАЧЕСТВА:")
        print(f"✅ Хорошие модели (CV ≥35%): {good_models}")
        print(f"⚠️  Средние модели (CV 25-35%): {medium_models}")
        print(f"❌ Слабые модели (CV <25%): {poor_models}")
        print(f"🎯 100% обучение (без бэктеста): {no_backtest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск всех доступных торговых моделей")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Интервал предикшн в секундах (по умолчанию: 60)")
    parser.add_argument("--list", action="store_true", help="Только показать список моделей")
    parser.add_argument("--enable_smc_filter", action="store_true", 
                       help="Включить SMC (Smart Money Concepts) фильтр для всех моделей")
    parser.add_argument("--smc_min_confluence", type=int, default=3, choices=[2, 3, 4, 5],
                       help="Минимальный confluence score для SMC фильтра (по умолчанию: 3)")
    args = parser.parse_args()
    
    launcher = AllModelsLauncher(
        prediction_interval=args.interval,
        enable_smc_filter=args.enable_smc_filter,
        smc_min_confluence=args.smc_min_confluence
    )
    
    # Проверяем аргументы командной строки
    if args.list:
        launcher.show_models_list()
    else:
        interval_text = f"{args.interval} сек" if args.interval < 60 else f"{args.interval//60} мин"
        print(f"⏱️ Интервал предикшн: {interval_text}")
        if args.enable_smc_filter:
            print(f"🛡️ SMC Filter: ENABLED (min confluence: {args.smc_min_confluence}/6)")
        launcher.launch_all_models()
