#!/usr/bin/env python3
"""
Автоматический лаунчер торговых моделей
Сканирует папку models_v2 и запускает все доступные модели
"""

import os
import glob
import subprocess
import time
import json
import sys
import argparse
from datetime import datetime
import json
from pathlib import Path

class ModelAutoLauncher:
    def __init__(self, prediction_interval=60):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.script_dir, "models_v2")
        self.python_exe = sys.executable
        self.prediction_interval = prediction_interval
        self.predict_script = "predict.py"

    def convert_symbol_to_ccxt_format(self, symbol):
        """
        Автоматическое преобразование символа в CCXT формат
        BTCUSDT -> BTC/USDT
        ETHUSDT -> ETH/USDT
        Поддерживает любые новые символы автоматически!
        """
        if symbol.endswith("USDT"):
            base = symbol[:-4]  # Убираем USDT
            return f"{base}/USDT"
        elif symbol.endswith("BUSD"):
            base = symbol[:-4]  # Убираем BUSD  
            return f"{base}/BUSD"
        elif symbol.endswith("BTC"):
            base = symbol[:-3]  # Убираем BTC
            return f"{base}/BTC"
        else:
            # Если формат неизвестен, возвращаем как есть
            return symbol

    def find_available_models(self):
        """Найти все доступные модели в папке models_v2"""
        if not os.path.exists(self.models_dir):
            print(f"Папка моделей не найдена: {self.models_dir}")
            return []
        
        # Найти все файлы моделей
        model_files = glob.glob(os.path.join(self.models_dir, "xgb_*.json"))
        models = []
        
        for model_file in model_files:
            # Извлечь информацию из имени файла: 
            # xgb_SYMBOL_30m_1234d.json (старый формат)
            # xgb_SYMBOL_30m_1234d_20250907_195124.json (новый формат с версионированием)
            filename = os.path.basename(model_file)
            parts = filename.replace("xgb_", "").replace(".json", "").split("_")
            
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1] 
                period = parts[2]
                
                # Проверяем, есть ли версионная информация (timestamp)
                version_timestamp = None
                if len(parts) >= 5:  # SYMBOL_TF_PERIOD_DATE_TIME
                    version_timestamp = f"{parts[3]}_{parts[4]}"
                elif len(parts) == 4:  # Возможно старый формат или только дата
                    # Проверяем, выглядит ли как timestamp
                    if len(parts[3]) == 8 and parts[3].isdigit():  # YYYYMMDD
                        version_timestamp = parts[3]
                
                # Найти соответствующий meta файл
                meta_file = model_file.replace("xgb_", "meta_")
                
                if os.path.exists(meta_file):
                    # Автоматически преобразовать символ в CCXT формат
                    display_symbol = self.convert_symbol_to_ccxt_format(symbol)
                    
                    models.append({
                        "symbol": symbol,
                        "display_symbol": display_symbol,
                        "timeframe": timeframe,
                        "period": period,
                        "version_timestamp": version_timestamp,
                        "model_file": model_file,
                        "meta_file": meta_file,
                        "filename": filename
                    })
        
        return models

    def get_latest_models(self, models):
        """Выбрать самые новые модели для каждого символа"""
        latest_models = {}
        
        for model in models:
            symbol = model["symbol"]
            period_days = int(model["period"].replace("d", ""))
            
            if symbol not in latest_models:
                latest_models[symbol] = model
            else:
                current_period = int(latest_models[symbol]["period"].replace("d", ""))
                # Выбираем модель с большим периодом (более свежая)
                if period_days > current_period:
                    latest_models[symbol] = model
        
        return list(latest_models.values())

    def check_model_quality(self, meta_file):
        """Проверить качество модели по meta файлу"""
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # Критерии качества модели
            min_accuracy = 0.35  # Минимальная точность 35%
            min_f1 = 0.30       # Минимальный F1-score 30%
            
            # Получить CV scores [accuracy, f1] для каждого fold
            cv_scores = meta.get('cv_scores', [])
            if not cv_scores:
                return False, "Нет CV scores в meta файле"
            
            # Вычислить средние метрики по всем folds
            accuracies = [fold[0] for fold in cv_scores if len(fold) >= 1]
            f1_scores = [fold[1] for fold in cv_scores if len(fold) >= 2]
            
            if not accuracies or not f1_scores:
                return False, "Некорректные CV scores"
            
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            
            if avg_accuracy >= min_accuracy and avg_f1 >= min_f1:
                return True, f"Качество OK: Acc={avg_accuracy:.1%}, F1={avg_f1:.3f}"
            else:
                return False, f"Низкое качество: Acc={avg_accuracy:.1%}, F1={avg_f1:.3f}"
                
        except Exception as e:
            return False, f"Ошибка чтения meta: {e}"

    def get_model_quality_score(self, model):
        """Получить оценку качества модели по meta файлу (accuracy)"""
        try:
            with open(model["meta_file"], 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # Получаем CV scores [accuracy, f1] для каждого fold
            cv_scores = meta.get('cv_scores', [])
            if not cv_scores:
                return 0.0
            
            # Вычисляем среднюю точность по всем folds
            accuracies = [fold[0] for fold in cv_scores if len(fold) >= 1]
            if not accuracies:
                return 0.0
            
            return sum(accuracies) / len(accuracies)
                    
        except Exception as e:
            print(f"Ошибка при получении качества: {e}")
            return 0.0

    def get_backtest_metrics(self, model_info):
        """Получить Win Rate, Profit Factor и Total Trades из meta файла"""
        try:
            meta_file = model_info.get('meta_file')
            if not meta_file or not os.path.exists(meta_file):
                return 0.0, 1.0, 0
            
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # Читаем результаты бэктеста из метаданных
            backtest_results = meta_data.get('backtest_results', {})
            if not backtest_results:
                print(f"❌ Модель {model_info['symbol']}_{model_info['period']} без результатов бэктеста")
                return 0.0, 1.0, 0
            
            win_rate = backtest_results.get('win_rate', 0.0)
            profit_factor = backtest_results.get('profit_factor', 1.0)
            total_trades = backtest_results.get('total_trades', 0)
            
            # Обрабатываем бесконечный profit_factor
            if isinstance(profit_factor, str) and profit_factor.lower() == 'inf':
                profit_factor = 100.0
            elif profit_factor == float('inf'):
                profit_factor = 100.0
            
            return win_rate, profit_factor, total_trades
                
        except Exception as e:
            print(f"❌ Ошибка чтения метаданных {model_info.get('symbol', 'unknown')}: {e}")
            return 0.0, 1.0, 0

    def group_models_by_symbol(self, models):
        """Группирует модели по символам для последующего выбора лучшей"""
        grouped = {}
        for model in models:
            symbol = model["symbol"]
            if symbol not in grouped:
                grouped[symbol] = []
            grouped[symbol].append(model)
        return grouped

    def get_best_models(self, models):
        """Выбирает лучшие модели для каждого символа на основе качества"""
        # Группируем модели по символу
        grouped_models = self.group_models_by_symbol(models)
        best_models = []
        
        print("\n🔍 АНАЛИЗ И ВЫБОР ЛУЧШИХ МОДЕЛЕЙ:")
        print("=" * 60)
        
        for symbol, symbol_models in grouped_models.items():
            print(f"\n📊 {symbol}:")
            
            # Сначала сортируем по качеству CV (точности)
            sorted_by_cv = sorted(
                symbol_models, 
                key=lambda x: self.get_model_quality_score(x), 
                reverse=True
            )
            
            # Отбираем модели с хорошим CV (>0.35)
            good_cv_models = [m for m in sorted_by_cv 
                              if self.get_model_quality_score(m) > 0.35]
            
            if not good_cv_models:
                # Если нет моделей с хорошим CV, берем лучшую доступную
                if sorted_by_cv:
                    best_models.append(sorted_by_cv[0])
                    cv_score = self.get_model_quality_score(sorted_by_cv[0])
                    print(f"   ⚠️  Низкий CV, используем лучшую: {sorted_by_cv[0]['period']}d (CV: {cv_score:.1%})")
                continue
            
            # Среди моделей с хорошим CV выбираем лучшую по торговым показателям
            best_model = None
            best_score = -1
            
            print(f"   📈 Доступные модели с хорошим CV:")
            
            for model in good_cv_models:
                cv_score = self.get_model_quality_score(model)
                
                # Анализируем торговые показатели
                win_rate, pf, total_trades = self.get_backtest_metrics(model)
                
                # Проверяем минимальное количество сделок (исключаем ненадёжные модели)
                if total_trades < 40:
                    # Формируем описание модели с версией
                    model_desc = f"{model['period']}"
                    if model.get('version_timestamp'):
                        model_desc += f"_{model['version_timestamp']}"
                    print(f"      ⚠️  {model_desc}: CV={cv_score:.1%}, WR={win_rate:.1%}, PF={pf:.2f}, Trades={total_trades} (недостаточно сделок)")
                    
                    # Записываем метрики в модель но помечаем как ненадёжную
                    model["win_rate"] = win_rate
                    model["profit_factor"] = pf
                    model["total_trades"] = total_trades
                    model["quality_score"] = 0.0  # Низкая оценка для ненадёжных моделей
                    continue
                
                # Комбинированная оценка качества: WR * PF * CV
                quality_score = win_rate * pf * cv_score
                
                # Формируем описание модели с версией
                model_desc = f"{model['period']}"
                if model.get('version_timestamp'):
                    model_desc += f"_{model['version_timestamp']}"
                
                print(f"      🔸 {model_desc}: CV={cv_score:.1%}, WR={win_rate:.1%}, PF={pf:.2f}, Score={quality_score:.3f}")
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_model = model
                
                # Записываем метрики в модель для отображения
                model["win_rate"] = win_rate
                model["profit_factor"] = pf
                model["total_trades"] = total_trades
                model["quality_score"] = quality_score
            
            # Выбираем лучшую модель
            if best_model:
                wr = best_model.get("win_rate", 0) * 100
                pf = best_model.get("profit_factor", 0)
                cv = self.get_model_quality_score(best_model) * 100
                score = best_model.get("quality_score", 0)
                
                # Формируем описание выбранной модели
                selected_desc = f"{best_model['period']}"
                if best_model.get('version_timestamp'):
                    selected_desc += f"_{best_model['version_timestamp']}"
                
                print(f"   ✅ ВЫБРАНА: {selected_desc}")
                print(f"      📊 CV: {cv:.1f}%, WR: {wr:.1f}%, PF: {pf:.2f}, Итоговый балл: {score:.3f}")
                
                best_models.append(best_model)
            else:
                # Если торговые показатели не найдены, проверяем качество модели
                best_candidate = good_cv_models[0]
                cv_score = self.get_model_quality_score(best_candidate)
                
                # Проверяем, есть ли у модели достаточно сделок
                win_rate, pf, total_trades = self.get_backtest_metrics(best_candidate)
                
                if total_trades < 40:
                    # Модель ненадёжная - не выбираем её
                    fallback_desc = f"{best_candidate['period']}"
                    if best_candidate.get('version_timestamp'):
                        fallback_desc += f"_{best_candidate['version_timestamp']}"
                    
                    print(f"   ❌ Модель {fallback_desc} ненадёжна (сделок: {total_trades} < 40), пропускаем")
                    # Не добавляем в best_models - символ остается без модели
                else:
                    # Модель надёжная, но без торговых данных - выбираем по CV
                    fallback_desc = f"{best_candidate['period']}"
                    if best_candidate.get('version_timestamp'):
                        fallback_desc += f"_{best_candidate['version_timestamp']}"
                    
                    print(f"   ⚠️  Нет торговых данных, выбор по CV: {fallback_desc} (CV: {cv_score:.1%})")
                    best_models.append(best_candidate)
        
        print("=" * 60)
        return best_models

    def save_cleanup_recommendations(self, all_models, selected_models):
        """Сохраняет рекомендации по очистке моделей"""
        try:
            # Группируем все модели по символам
            all_grouped = self.group_models_by_symbol(all_models)
            selected_grouped = self.group_models_by_symbol(selected_models)
            
            recommendations = {
                "timestamp": datetime.now().isoformat(),
                "analysis_results": {},
                "models_to_keep": [],
                "models_to_remove": []
            }
            
            # Анализируем каждый символ
            for symbol, symbol_models in all_grouped.items():
                # Получаем выбранную модель для этого символа
                selected_model = next((m for m in selected_models if m["symbol"] == symbol), None)
                
                # Сортируем все модели символа по качеству
                symbol_models_with_scores = []
                for model in symbol_models:
                    cv_score = self.get_model_quality_score(model)
                    win_rate, pf, total_trades = self.get_backtest_metrics(model)
                    
                    # Применяем фильтр по минимальному количеству сделок
                    if total_trades < 40:
                        quality_score = 0.0  # Низкая оценка для ненадёжных моделей
                    else:
                        quality_score = win_rate * pf * cv_score
                    
                    model_info = {
                        "model_file": model["model_file"],
                        "meta_file": model["meta_file"],
                        "symbol": symbol,
                        "period": model["period"],
                        "cv_score": cv_score,
                        "win_rate": win_rate,
                        "profit_factor": pf,
                        "total_trades": total_trades,
                        "quality_score": quality_score,
                        "is_selected": model == selected_model
                    }
                    
                    if model.get('version_timestamp'):
                        model_info["version_timestamp"] = model["version_timestamp"]
                    
                    symbol_models_with_scores.append(model_info)
                
                # Сортируем по качеству (лучшие сначала)
                symbol_models_with_scores.sort(key=lambda x: x["quality_score"], reverse=True)
                
                # Оставляем топ-3 модели по качеству + выбранную модель
                models_to_keep = []
                models_to_remove = []
                
                # Всегда сохраняем выбранную модель
                if selected_model:
                    selected_info = next((m for m in symbol_models_with_scores if m["is_selected"]), None)
                    if selected_info:
                        models_to_keep.append(selected_info)
                
                # Добавляем еще топ-2 модели (если они не совпадают с выбранной)
                kept_count = 1 if selected_model else 0
                for model_info in symbol_models_with_scores:
                    if kept_count >= 3:  # Максимум 3 модели на символ
                        models_to_remove.append(model_info)
                    elif not model_info["is_selected"]:  # Не дублируем выбранную
                        # Проверяем качество модели - исключаем откровенно плохие
                        if (model_info["win_rate"] == 0 and model_info["profit_factor"] <= 1.0) or \
                           model_info["quality_score"] < 0.1 or model_info["cv_score"] < 0.30 or \
                           model_info["total_trades"] < 40:  # Ненадёжные модели с малым количеством сделок
                            models_to_remove.append(model_info)
                        else:
                            models_to_keep.append(model_info)
                            kept_count += 1
                    elif not selected_model:  # Если выбранной модели нет
                        models_to_keep.append(model_info)
                        kept_count += 1
                
                # Остальные модели на удаление
                for model_info in symbol_models_with_scores[3:]:
                    if model_info not in models_to_keep:
                        models_to_remove.append(model_info)
                
                recommendations["analysis_results"][symbol] = {
                    "total_models": len(symbol_models),
                    "selected_model": selected_info["model_file"] if selected_model else None,
                    "models_analyzed": symbol_models_with_scores
                }
                
                recommendations["models_to_keep"].extend(models_to_keep)
                recommendations["models_to_remove"].extend(models_to_remove)
            
            # Сохраняем рекомендации
            recommendations_file = "cleanup_recommendations.json"
            with open(recommendations_file, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Сохранены рекомендации по очистке: {recommendations_file}")
            print(f"   📊 К сохранению: {len(recommendations['models_to_keep'])} моделей")
            print(f"   🗑️ К удалению: {len(recommendations['models_to_remove'])} моделей")
            
        except Exception as e:
            print(f"Ошибка сохранения рекомендаций: {e}")

    # Улучшенная функция для выбора лучшей модели
    def choose_best_model_version(self, models_list):
        # Шаг 1: Отбор моделей с приемлемым CV (>0.35)
        good_cv_models = [m for m in models_list 
                         if self.get_model_quality_score(m) > 0.35]
        
        if not good_cv_models:
            return models_list[0] if models_list else None
        
        # Шаг 2: Анализ торговых показателей для выбора лучшей модели
        best_model = None
        best_score = -1
        
        for model in good_cv_models:
            try:
                # Анализируем Win Rate и Profit Factor
                win_rate, pf, total_trades = self.get_backtest_metrics(model)
                # Комбинированная оценка качества
                score = win_rate * pf
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                pass
        
        return best_model if best_model else good_cv_models[0]

    def launch_models(self):
        """Запустить лучшие доступные модели"""
        print("Сканирование моделей в папке models_v2...")
        
        all_models = self.find_available_models()
        if not all_models:
            print("Модели не найдены!")
            return
        
        print(f"Найдено {len(all_models)} моделей")
        
        # Выбираем ЛУЧШИЕ модели по качеству (не просто последние)
        best_models = self.get_best_models(all_models)
        print(f"Отобрано {len(best_models)} лучших моделей")
        
        # Сохраняем рекомендации для очистки
        self.save_cleanup_recommendations(all_models, best_models)
        
        launched_count = 0
        skipped_count = 0
        
        print("\n" + "="*60)
        print("ЗАПУСК ТОРГОВЫХ МОДЕЛЕЙ")
        print("="*60)
        
        for model in sorted(best_models, key=lambda x: x["symbol"]):
            # Проверить качество модели
            is_good, quality_msg = self.check_model_quality(model["meta_file"])
            
            if not is_good:
                print(f"ПРОПУСК {model['symbol']}: {quality_msg}")
                skipped_count += 1
                continue
            
            # Дополнительная информация о периоде
            print(f"ЗАПУСК {model['symbol']} ({model['period']}, {model['timeframe']})")
            
            # Запустить модель
            cmd = [
                "start",
                f'"{model["symbol"]} {model["timeframe"]}"',
                "cmd", "/k",
                f'"{self.python_exe}"',
                self.predict_script,
                "--meta", model["meta_file"],
                "--model", model["model_file"], 
                "--symbol", model["display_symbol"],
                "--tf", model["timeframe"],
                "--interval", str(self.prediction_interval)
            ]
            
            try:
                subprocess.run(" ".join(cmd), shell=True, check=True)
                launched_count += 1
                time.sleep(1)  # Пауза между запусками
                
            except subprocess.CalledProcessError as e:
                print(f"ОШИБКА {model['symbol']}: {e}")
                skipped_count += 1
        
        # Запустить HeartbeatManager
        print("\nЗапуск HeartbeatManager...")
        try:
            heartbeat_cmd = [
                "start", '"Heartbeat Manager"', "cmd", "/k",
                f'"{self.python_exe}"', "HeartbeatManager.py"
            ]
            subprocess.run(" ".join(heartbeat_cmd), shell=True, check=True)
            print("HeartbeatManager запущен")
        except Exception as e:
            print(f"Ошибка запуска HeartbeatManager: {e}")
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ЗАПУСКА:")
        print(f"Запущено моделей: {launched_count}")
        print(f"Пропущено моделей: {skipped_count}")
        print(f"Общий успех: {launched_count}/{len(best_models)}")
        print("="*60)
        
        if launched_count > 0:
            print("\nТорговая система успешно запущена!")
            print("Все терминалы останутся открытыми и будут работать непрерывно")
        else:
            print("\nНе удалось запустить ни одной модели!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автоматический запуск лучших торговых моделей")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Интервал предикшн в секундах (по умолчанию: 60)")
    args = parser.parse_args()
    
    interval_text = f"{args.interval} сек" if args.interval < 60 else f"{args.interval//60} мин"
    print(f"⏱️ Интервал предикшн: {interval_text}")
    
    launcher = ModelAutoLauncher(prediction_interval=args.interval)
    launcher.launch_models()
    
    input("\nНажмите Enter для выхода...")
