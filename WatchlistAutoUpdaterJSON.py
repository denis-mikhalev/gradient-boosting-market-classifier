#!/usr/bin/env python3
"""
WatchlistAutoUpdaterJSON.py
Автоматический updater для watchlist.json с точечными обновлениями

Функциональность:
1. Читает результаты анализа периодов
2. Обновляет ТОЛЬКО проанализированные монеты
3. Сохраняет все остальные данные без изменений
4. Добавляет детальную историю оптимизаций
5. Создает отчеты и экспорты
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Импортируем наш JSON менеджер
from WatchlistJSONManager import WatchlistJSONManager

class WatchlistAutoUpdaterJSON:
    """Автоматический updater для JSON watchlist с точечными обновлениями"""
    
    def __init__(self, watchlist_path: str = "watchlist.json", auto_cleanup: bool = True):
        self.watchlist_path = watchlist_path
        self.json_manager = WatchlistJSONManager(watchlist_path)
        self.auto_cleanup = auto_cleanup
        
    def analyze_periods_from_results(self, results_file: str, 
                                   primary_metric: str = "profit_factor",
                                   min_trades: int = 5,
                                   require_profitable: bool = True,
                                   keep_top_n: int = 1) -> dict:
        """Анализирует файл результатов и находит оптимальные периоды
        
        Args:
            require_profitable: Если True, исключает убыточные модели (PF < 1.0 или Return <= 0)
            keep_top_n: Количество лучших моделей для сохранения (по умолчанию: 1)
        """
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки {results_file}: {e}")
            return {}
        
        experiments = data.get('experiments', {})
        if not experiments:
            print("❌ Нет экспериментов в файле")
            return {}
        
        
        print(f"📊 Найдено {len(experiments)} символов для анализа")
        
        # 🔍 КРИТИЧНО: Ищем ВСЕ символы на диске, не только из experiments
        all_symbols_on_disk = self._find_all_symbols_on_disk()
        print(f"💾 Найдено {len(all_symbols_on_disk)} символов на диске: {', '.join(sorted(all_symbols_on_disk))}")
        
        # Объединяем символы из experiments и с диска
        all_symbols = set(experiments.keys()) | all_symbols_on_disk
        print(f"🔄 Всего символов для обработки: {len(all_symbols)}")
        
        results = {}
        
        for symbol in sorted(all_symbols):
            # Получаем данные из experiments (если есть)
            symbol_data = experiments.get(symbol, {})
            periods_data = symbol_data.get('periods', {})
            
            # 🔍 ВСЕГДА ищем существующие модели на диске
            existing_models = self._find_existing_models(symbol)
            
            # Если нет ни новых данных, ни существующих моделей - пропускаем
            if not periods_data and not existing_models:
                continue
            
            # КРИТИЧНО: Объединяем модели БЕЗ перезаписи одинаковых периодов
            # Сохраняем лучшую модель для каждого периода
            all_periods_data = {}
            
            # Извлекаем числовые периоды из periods_data (новые модели)
            new_periods_numeric = set()
            for period_key in periods_data.keys():
                # Извлекаем числовой период (например: "720" или "720_timestamp")
                clean_period = period_key.split('_')[0] if '_' in period_key else period_key
                if clean_period.isdigit():
                    new_periods_numeric.add(int(clean_period))
            
            # Сначала добавляем существующие модели, КРОМЕ тех, что только что создали
            for period, model_data in existing_models.items():
                # Извлекаем числовой период из ключа существующей модели
                clean_period = period.split('_')[0] if '_' in period else period
                if clean_period.isdigit():
                    period_num = int(clean_period)
                    # Пропускаем, если этот период есть в новых моделях
                    if period_num in new_periods_numeric:
                        continue
                
                all_periods_data[f"existing_{period}"] = model_data
            
            # Затем добавляем новые модели с префиксом
            for period, model_data in periods_data.items():
                all_periods_data[f"new_{period}"] = model_data
            
            if existing_models and periods_data:
                print(f"🔄 {symbol}: объединяем {len(existing_models)} существующих + {len(periods_data)} новых моделей")
            elif existing_models:
                print(f"📁 {symbol}: анализируем {len(existing_models)} существующих моделей")
            elif periods_data:
                print(f"🆕 {symbol}: анализируем {len(periods_data)} новых моделей")
            
            analysis = self._analyze_symbol_periods(
                symbol, all_periods_data, primary_metric, min_trades, require_profitable, keep_top_n
            )
            
            if analysis:
                results[symbol] = analysis
        
        return results
    
    def _find_all_symbols_on_disk(self, models_dir: str = "models_v2") -> set:
        """Находит все символы, для которых есть модели на диске"""
        models_path = Path(models_dir)
        if not models_path.exists():
            return set()
        
        symbols = set()
        
        # Ищем все мета-файлы
        meta_files = list(models_path.glob("meta_*.json"))
        
        for meta_file in meta_files:
            # Извлекаем символ из имени файла: meta_BTCUSDT_30m_... -> BTCUSDT
            filename = meta_file.stem  # meta_BTCUSDT_30m_2000d_bt30d_20250920_170407
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == 'meta':
                symbol = parts[1]
                symbols.add(symbol)
        
        return symbols
    
    def _check_watchlist_models_on_disk(self, symbol: str, models_path: Path) -> dict:
        """Проверяет, существуют ли модели из watchlist.json на диске"""
        try:
            watchlist_data = self.json_manager.load_watchlist()
            coin_data = watchlist_data.get('coins', {}).get(symbol, {})
            
            if not coin_data or not coin_data.get('active', True):
                return {}
            
            models_in_watchlist = coin_data.get('models', [])
            if not models_in_watchlist:
                # Старый формат - пробуем best_model_file
                best_model = coin_data.get('best_model_file')
                if best_model:
                    models_in_watchlist = [{'model_file': best_model}]
            
            result = {}
            for model_info in models_in_watchlist:
                model_file = model_info.get('model_file')
                if not model_file:
                    continue
                
                # Проверяем существование файла
                model_path = models_path / model_file
                if model_path.exists():
                    result[model_file] = True
                else:
                    print(f"   ⚠️ Модель из watchlist не найдена на диске: {model_file}")
                    result[model_file] = False
            
            return result
            
        except Exception as e:
            print(f"   ❌ Ошибка проверки watchlist моделей: {e}")
            return {}
    
    def _find_existing_models(self, symbol: str, models_dir: str = "models_v2") -> dict:
        """Находит все существующие модели для символа на диске"""
        models_path = Path(models_dir)
        if not models_path.exists():
            return {}
        
        existing_models = {}
        
        # Ищем все мета-файлы для данного символа
        pattern = f"meta_{symbol}_*.json"
        meta_files = list(models_path.glob(pattern))
        
        # Проверяем, есть ли модели из watchlist.json на диске
        watchlist_models_on_disk = self._check_watchlist_models_on_disk(symbol, models_path)
        
        # Если модели из watchlist отсутствуют на диске, выводим предупреждение
        missing_count = sum(1 for exists in watchlist_models_on_disk.values() if not exists)
        if missing_count > 0:
            print(f"   ⚠️ {symbol}: {missing_count} модель(ей) из watchlist.json отсутствуют на диске - будут заменены новыми")
        
        print(f"🔎 Поиск существующих моделей {symbol}: найдено {len(meta_files)} мета-файлов")
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                # Извлекаем информацию о модели
                version_info = meta_data.get('version_info', {})
                base_name = version_info.get('base_name', '')
                backtest_results = meta_data.get('backtest_results', {})
                
                if not base_name or not backtest_results:
                    continue
                
                # Извлекаем период из base_name (например: BTCUSDT_30m_2000d_bt30d_20250920_170407 -> 2000)
                name_parts = base_name.split('_')
                period_part = None
                for part in name_parts:
                    if part.endswith('d') and part[:-1].isdigit():
                        period_part = part[:-1]  # Убираем 'd'
                        break
                
                if not period_part:
                    continue
                
                # Используем уникальный ключ: period + timestamp
                timestamp = version_info.get('timestamp', '')
                unique_key = f"{period_part}_{timestamp}" if timestamp else period_part
                
                # Формируем данные в том же формате что и для новых моделей
                model_data = {
                    'success': True,
                    'threshold_data': {
                        'backtest_stats': backtest_results
                    },
                    'source': 'existing_model',
                    'model_file': f"xgb_{base_name}.json",
                    'meta_file': meta_file.name,
                    'created_at': version_info.get('created_at', '')
                }
                
                existing_models[unique_key] = model_data
                print(f"   📄 Найдена модель {period_part}д ({timestamp}): PF={backtest_results.get('profit_factor', 0):.2f}, "
                      f"Return={backtest_results.get('total_return_pct', 0)*100:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Ошибка чтения {meta_file.name}: {e}")
                continue
        
        return existing_models
    
    def _analyze_symbol_periods(self, symbol: str, periods_data: dict, 
                               primary_metric: str, min_trades: int, 
                               require_profitable: bool = True, keep_top_n: int = 1) -> dict:
        """Анализирует все периоды для одного символа и возвращает топ-N лучших моделей"""
        
        all_periods = []
        
        # Счетчики для статистики
        backtest_count = 0
        skipped_count = 0  # Модели без backtest_stats
        
        print(f"🔍 Анализ {symbol}: {len(periods_data)} периодов...")
        
        # Анализируем каждый период
        for period_str, period_data in periods_data.items():
            if not period_data.get('success', False):
                continue
            
            # Получаем статистики ТОЛЬКО из детального бэктеста
            threshold_data = period_data.get('threshold_data', {})
            backtest_stats = threshold_data.get('backtest_stats', {})
            
            # Требуем обязательное наличие backtest_stats
            if not backtest_stats:
                print(f"❌ ОШИБКА: Нет backtest_stats для {symbol} период {period_str} - ПРОПУСКАЕМ модель!")
                print(f"   Эта модель не прошла полный бэктест и не может быть использована.")
                skipped_count += 1
                continue
            
            # Используем только данные детального бэктеста
            backtest_count += 1
            win_rate = backtest_stats.get('win_rate', 0) * 100  # Конвертируем в проценты
            profit_factor = backtest_stats.get('profit_factor', 0)
            trades = backtest_stats.get('total_trades', 0)
            threshold = backtest_stats.get('confidence_threshold', 0.5)
            total_return_pct = backtest_stats.get('total_return_pct', 0)
            data_source = "backtest_stats"
            
            # Проверяем минимальное количество сделок
            if trades < min_trades:
                continue
            
            # 🚨 КРИТИЧНО: Фильтруем убыточные модели (если требуется)
            if require_profitable and (profit_factor < 1.0 or total_return_pct <= 0):
                print(f"🚨 УБЫТОЧНАЯ модель: {symbol} период {period_str} "
                      f"(PF={profit_factor:.2f}, Return={total_return_pct*100:.1f}%) - ПРОПУСКАЕМ!")
                skipped_count += 1
                continue
            
            model_file = period_data.get('model_file', '')
            
            # 🔍 КРИТИЧНО: Проверяем существование файла модели на диске
            if model_file:
                model_path = Path("models_v2") / model_file
                if not model_path.exists():
                    print(f"⚠️ ПРОПУСК: {symbol} период {period_str} - файл модели не найден: {model_file}")
                    skipped_count += 1
                    continue
            
            # Извлекаем числовой период из ключа (убираем префиксы existing_ или new_)
            clean_period_str = period_str
            if period_str.startswith('existing_'):
                clean_period_str = period_str[9:]  # Убираем 'existing_'
            elif period_str.startswith('new_'):
                clean_period_str = period_str[4:]   # Убираем 'new_'
            
            # Извлекаем период из ключа с timestamp (например: 2000_20250920_170407 -> 2000)
            if '_' in clean_period_str:
                clean_period_str = clean_period_str.split('_')[0]
            
            # Сохраняем информацию о периоде
            period_info = {
                'period': int(clean_period_str),
                'period_key': period_str,  # Сохраняем оригинальный ключ для отладки
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trades': trades,
                'threshold': threshold,
                'model_file': model_file,
                'data_source': data_source  # Отслеживаем источник данных
            }
            all_periods.append(period_info)
            
            # Определяем значение по метрике
            if primary_metric == "profit_factor":
                # Обрабатываем infinity
                if profit_factor == float('inf'):
                    value = 999.0
                else:
                    value = profit_factor
            elif primary_metric == "win_rate":
                value = win_rate
            elif primary_metric == "trades":
                value = trades
            else:  # score
                # Простой комплексный скор
                if profit_factor == float('inf'):
                    pf_norm = 999.0
                else:
                    pf_norm = profit_factor
                
                wr_norm = win_rate / 100.0 if win_rate > 1 else win_rate
                value = wr_norm * pf_norm * (trades / 10.0)
            
            # Сохраняем значение метрики для каждого периода
            period_info['metric_value'] = value
        
        if not all_periods:
            if skipped_count > 0:
                print(f"❌ {symbol}: Не найдено подходящих периодов "
                      f"(использовано: {backtest_count}, пропущено: {skipped_count})")
            else:
                print(f"❌ {symbol}: Не найдено подходящих периодов (использовано: {backtest_count})")
            return {}
        
        # Сортируем все модели по значению метрики (по убыванию)
        all_periods.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Выбираем топ-N моделей (просто берём N лучших, без проверки на уникальность периодов)
        top_models = all_periods[:keep_top_n]
        
        # Выводим статистику
        if keep_top_n == 1:
            best_stats = top_models[0]
            print(f"✅ {symbol}: Период {best_stats['period']} | "
                  f"Использовано моделей: {backtest_count} | Пропущено: {skipped_count} | "
                  f"Источник: {best_stats.get('data_source', 'unknown')}")
        else:
            print(f"✅ {symbol}: Топ-{len(top_models)} моделей:")
            for i, model in enumerate(top_models, 1):
                print(f"   {i}. Период {model['period']} | "
                      f"PF={model['profit_factor']:.2f} | WR={model['win_rate']:.1f}% | "
                      f"Trades={model['trades']}")
            print(f"   Использовано моделей: {backtest_count} | Пропущено: {skipped_count}")
        
        # Генерируем объяснение для лучшей модели
        best_stats = top_models[0]
        reason = self._generate_reason(best_stats, primary_metric)
        
        # Всегда возвращаем единый формат - массив моделей
        return {
            'top_models': [
                {
                    'period': model['period'],
                    'threshold': model['threshold'],
                    'win_rate': model['win_rate'],
                    'profit_factor': model['profit_factor'],
                    'trades': model['trades'],
                    'metric_value': model['metric_value'],
                    'model_file': model.get('model_file', ''),
                    'data_source': model.get('data_source', 'unknown')
                }
                for model in top_models
            ],
            'reason': reason,
            'all_periods_tested': all_periods,
            'primary_metric': primary_metric,
            'analysis_date': datetime.now().isoformat(),
            'data_source_stats': {
                'backtest_count': backtest_count,
                'skipped_count': skipped_count,
                'total_periods': len(periods_data)
            }
        }
    
    def _generate_reason(self, best_stats: dict, metric: str) -> str:
        """Генерирует объяснение выбора"""
        
        if metric == "profit_factor":
            if best_stats['profit_factor'] >= 999:
                return f"Максимальный PF (∞), {best_stats['trades']} сделок"
            else:
                return f"Лучший PF ({best_stats['profit_factor']:.1f}), {best_stats['trades']} сделок"
        
        elif metric == "win_rate":
            return f"Максимальный WR ({best_stats['win_rate']:.1f}%), {best_stats['trades']} сделок"
        
        elif metric == "trades":
            return f"Максимум сделок ({best_stats['trades']}), PF {best_stats['profit_factor']:.1f}"
        
        else:  # score
            return f"Лучший баланс (Score {best_stats['metric_value']:.2f}), {best_stats['trades']} сделок"
    
    def update_watchlist_from_analysis(self, analysis_results: dict, 
                                     analysis_source: str,
                                     dry_run: bool = False) -> dict:
        """Обновляет watchlist только для проанализированных монет"""
        
        if not analysis_results:
            print("❌ Нет данных для обновления")
            return {}
        
        if dry_run:
            print("🧪 DRY RUN: Показываем изменения без применения")
        
        # Получаем текущие данные
        current_data = self.json_manager.load_watchlist()
        current_coins = current_data.get('coins', {})
        
        update_log = {}
        changes_count = 0
        
        print(f"🔄 {'DRY RUN: ' if dry_run else ''}Точечное обновление watchlist...")
        print("=" * 60)
        
        # Обновляем только проанализированные монеты
        for symbol, analysis in analysis_results.items():
            reason = analysis['reason']
            
            # Извлекаем массив моделей (единый формат)
            top_models = analysis['top_models']
            new_periods = [m['period'] for m in top_models]
            new_period_str = ', '.join([f"{p}д" for p in new_periods])
            primary_period = new_periods[0]  # Лучшая модель
            
            if symbol in current_coins:
                # Монета существует
                current_coin = current_coins[symbol]
                old_period = current_coin.get('period', 0)
                
                if old_period != primary_period:
                    # Период изменился
                    if len(new_periods) > 1:
                        update_log[symbol] = f"Обновлен: {old_period}д → [{new_period_str}] ({reason})"
                        print(f"🔄 {symbol}: {old_period}д → топ-{len(new_periods)} [{new_period_str}] ({reason})")
                    else:
                        update_log[symbol] = f"Обновлен: {old_period}д → {new_period_str} ({reason})"
                        print(f"🔄 {symbol}: {old_period}д → {new_period_str} ({reason})")
                    changes_count += 1
                else:
                    # Период подтвержден
                    if len(new_periods) > 1:
                        update_log[symbol] = f"Подтвержден: топ-{len(new_periods)} [{new_period_str}] ({reason})"
                        print(f"✅ {symbol}: топ-{len(new_periods)} [{new_period_str}] подтверждено ({reason})")
                    else:
                        update_log[symbol] = f"Подтвержден: {new_period_str} ({reason})"
                        print(f"✅ {symbol}: {new_period_str} подтверждено ({reason})")
            else:
                # Новая монета
                if len(new_periods) > 1:
                    update_log[symbol] = f"Добавлена: топ-{len(new_periods)} [{new_period_str}] ({reason})"
                    print(f"➕ {symbol}: Добавлена топ-{len(new_periods)} [{new_period_str}] ({reason})")
                else:
                    update_log[symbol] = f"Добавлена: {new_period_str} ({reason})"
                    print(f"➕ {symbol}: Добавлена с периодом {new_period_str} ({reason})")
                changes_count += 1
        
        # Показываем неизмененные монеты
        unchanged_coins = []
        for symbol, coin in current_coins.items():
            if symbol not in analysis_results and coin.get('active', True):
                unchanged_coins.append(symbol)
        
        if unchanged_coins:
            print(f"\\n➖ НЕИЗМЕНЕННЫЕ МОНЕТЫ ({len(unchanged_coins)}):")
            for symbol in unchanged_coins[:5]:  # Показываем первые 5
                coin = current_coins[symbol]
                period = coin.get('period', 0)
                print(f"   {symbol}: {period}д (нет данных для анализа)")
            
            if len(unchanged_coins) > 5:
                print(f"   ... и еще {len(unchanged_coins) - 5}")
        
        if changes_count == 0:
            print("\\n✅ Никаких изменений не требуется")
            return update_log
        
        if not dry_run:
            # Выполняем реальное обновление
            success_log = self.json_manager.update_coins_from_analysis(
                analysis_results, analysis_source
            )
            
            if success_log:
                print(f"\\n✅ Watchlist обновлен! Изменений: {changes_count}")
                
                # Создаем экспорт для проверки
                export_path = f"watchlist_after_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                if self.json_manager.export_to_txt(export_path):
                    print(f"📄 TXT экспорт создан: {export_path}")
                
                return success_log
            else:
                print(f"\\n❌ Ошибка обновления watchlist")
                return {}
        else:
            print(f"\\n🧪 DRY RUN: Было бы изменений: {changes_count}")
            return update_log
    
    def print_analysis_summary(self, analysis_results: dict):
        """Выводит сводку анализа с учетом текущего состояния"""
        
        print("\\n📊 СВОДКА АНАЛИЗА ОПТИМАЛЬНЫХ ПЕРИОДОВ")
        print("=" * 60)
        print(f"📈 Проанализировано символов: {len(analysis_results)}")
        
        if not analysis_results:
            print("❌ Нет данных для анализа")
            return
        
        # Получаем текущие данные
        current_data = self.json_manager.load_watchlist()
        current_coins = current_data.get('coins', {})
        
        # Группируем по типам изменений
        updates_needed = []
        confirmations = []
        new_coins = []
        
        for symbol, analysis in analysis_results.items():
            reason = analysis['reason']
            
            # Поддержка обоих форматов (одна модель и топ-N)
            if 'top_models' in analysis:
                new_period = analysis['top_models'][0]['period']
                model_count = len(analysis['top_models'])
                period_display = f"топ-{model_count}"
            else:
                new_period = analysis['period']
                period_display = f"{new_period}д"
            
            if symbol in current_coins:
                old_period = current_coins[symbol].get('period', 0)
                if old_period != new_period:
                    updates_needed.append((symbol, old_period, period_display, reason))
                else:
                    confirmations.append((symbol, period_display, reason))
            else:
                new_coins.append((symbol, period_display, reason))
        
        if updates_needed:
            print(f"\\n🔄 ТРЕБУЮТ ОБНОВЛЕНИЯ ({len(updates_needed)}):")
            for symbol, old_period, new_display, reason in updates_needed:
                print(f"   {symbol}: {old_period}д → {new_display} ({reason})")
        
        if confirmations:
            print(f"\\n✅ ПОДТВЕРЖДЕНЫ ({len(confirmations)}):")
            for symbol, period_display, reason in confirmations:
                print(f"   {symbol}: {period_display} ({reason})")
        
        if new_coins:
            print(f"\\n➕ НОВЫЕ МОНЕТЫ ({len(new_coins)}):")
            for symbol, period_display, reason in new_coins:
                print(f"   {symbol}: {period_display} ({reason})")
        
        # Статистика текущего watchlist
        active_count = current_data['metadata'].get('total_active_coins', 0)
        inactive_count = current_data['metadata'].get('total_inactive_coins', 0)
        
        print(f"\\n📋 ТЕКУЩИЙ WATCHLIST:")
        print(f"   🟢 Активных монет: {active_count}")
        print(f"   🔴 Отключенных монет: {inactive_count}")
        print(f"   📊 Монет в анализе: {len(analysis_results)}")
    
    def run_full_update(self, results_file: str, primary_metric: str = "profit_factor",
                       min_trades: int = 5, dry_run: bool = False, 
                       force: bool = False, require_profitable: bool = True, keep_top_n: int = 1) -> bool:
        """Запускает полное точечное обновление watchlist"""
        
        print("🚀 ТОЧЕЧНОЕ ОБНОВЛЕНИЕ JSON WATCHLIST")
        print("=" * 60)
        print(f"📁 Файл результатов: {results_file}")
        print(f"📋 Watchlist: {self.watchlist_path}")
        print(f"📊 Основная метрика: {primary_metric}")
        print(f"🔢 Минимум сделок: {min_trades}")
        print(f"🏆 Сохранить топ-N моделей: {keep_top_n}")
        if dry_run:
            print("🧪 Режим: DRY RUN (только показать изменения)")
        
        # Анализируем результаты
        analysis_results = self.analyze_periods_from_results(
            results_file, primary_metric, min_trades, require_profitable, keep_top_n
        )
        
        if not analysis_results:
            print("❌ Нет данных для обновления")
            return False
        
        # Сохраняем детальный лог анализа
        self.save_optimal_period_analysis_log(
            analysis_results, results_file, primary_metric, min_trades
        )
        
        # Показываем сводку
        self.print_analysis_summary(analysis_results)
        
        # Подтверждение
        if not force and not dry_run:
            print("\\n❓ Обновить watchlist с найденными оптимальными периодами? (y/N): ", end="")
            confirmation = input().strip().lower()
            
            if confirmation not in ['y', 'yes', 'да']:
                print("🚫 Обновление отменено")
                return False
        
        # Выполняем обновление
        update_log = self.update_watchlist_from_analysis(
            analysis_results, 
            results_file,
            dry_run
        )
        
        if update_log and not dry_run:
            print(f"\\n🎉 Точечное обновление watchlist завершено!")
            
            # Показываем итоговую сводку
            self.json_manager.print_summary()
            
            # Запускаем автоочистку после успешного обновления
            self.run_cleanup_if_enabled()
            
        elif dry_run:
            print(f"\\n🧪 DRY RUN завершен - запустите без --dry-run для применения")
        
        return len(update_log) > 0
        
    def run_cleanup_if_enabled(self):
        """Запускает очистку неоптимальных моделей если включена"""
        if not self.auto_cleanup:
            print("\\n🧹 Автоочистка отключена")
            return
            
        cleanup_script = Path("CleanupNonOptimalModelsSimple.py")
        
        if not cleanup_script.exists():
            print(f"\\n⚠️  Скрипт очистки {cleanup_script} не найден, пропускаем автоочистку")
            return
            
        print(f"\\n🧹 АВТОМАТИЧЕСКАЯ ОЧИСТКА НЕОПТИМАЛЬНЫХ МОДЕЛЕЙ")
        print("=" * 60)
        print("Запускаем очистку неоптимальных периодов...")
        
        try:
            # Запускаем очистку с --force (без подтверждения)
            cmd = [sys.executable, str(cleanup_script), "--force"]
            
            print(f"🔧 Команда: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 минут таймаут
            )
            
            if result.returncode == 0:
                print("✅ Очистка неоптимальных моделей завершена успешно!")
                if result.stdout:
                    # Показываем краткую сводку из stdout
                    lines = result.stdout.strip().split('\\n')
                    for line in lines[-10:]:  # Последние 10 строк
                        if any(keyword in line for keyword in ['удалено', 'очищено', 'файлов', 'Итого']):
                            print(f"   {line}")
            else:
                print(f"❌ Ошибка при очистке (код {result.returncode})")
                if result.stderr:
                    print(f"Ошибка: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("❌ Таймаут при выполнении очистки (>5 мин)")
        except Exception as e:
            print(f"❌ Исключение при запуске очистки: {e}")
            
        print("=" * 60)
    
    def save_optimal_period_analysis_log(self, analysis_results: dict, 
                                       results_source: str,
                                       primary_metric: str = "profit_factor",
                                       min_trades: int = 5) -> str:
        """Сохраняет детальный лог анализа оптимальных периодов
        
        Args:
            analysis_results: Результаты анализа символов (из analyze_periods_from_results)
            results_source: Источник данных (имя файла period_search_results)
            primary_metric: Основная метрика для сравнения
            min_trades: Минимальное количество сделок
            
        Returns:
            str: Путь к созданному файлу лога
        """
        try:
            # Создаем имя файла с timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"optimal_period_analysis_log_{timestamp}.json"
            log_path = os.path.join("optimal_period_analysis_logs", log_filename)
            
            # Создаем директорию если её нет
            os.makedirs("optimal_period_analysis_logs", exist_ok=True)
            
            # Формируем детальную структуру лога
            log_data = {
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source_file": results_source,
                    "primary_metric": primary_metric,
                    "min_trades": min_trades,
                    "total_symbols_analyzed": len(analysis_results)
                },
                "analysis_summary": {
                    "symbols_count": len(analysis_results),
                    "symbols_analyzed": list(analysis_results.keys())
                },
                "detailed_analysis": {},
                "models_discovered": {
                    "new_models": {},
                    "existing_models": {},
                    "total_models_count": 0
                },
                "optimization_results": {}
            }
            
            # Собираем детальную информацию по каждому символу
            total_models_tested = 0
            total_new_models = 0
            total_existing_models = 0
            
            for symbol, analysis in analysis_results.items():
                # Основная информация об анализе
                symbol_data = {
                    "selected_period": analysis.get('period'),
                    "selection_reason": analysis.get('reason'),
                    "optimal_metrics": {
                        "profit_factor": analysis.get('profit_factor'),
                        "win_rate": analysis.get('win_rate'),
                        "trades": analysis.get('trades'),
                        "threshold": analysis.get('threshold'),
                        "metric_value": analysis.get('metric_value')
                    },
                    "model_info": {
                        "model_file": analysis.get('model_file'),
                        "data_source": analysis.get('data_source')
                    },
                    "analysis_stats": analysis.get('data_source_stats', {}),
                    "all_periods_tested": analysis.get('all_periods_tested', [])
                }
                
                # Подсчитываем модели
                all_periods = analysis.get('all_periods_tested', [])
                total_models_tested += len(all_periods)
                
                new_models = []
                existing_models = []
                
                for period_info in all_periods:
                    model_detail = {
                        "period": period_info.get('period'),
                        "profit_factor": period_info.get('profit_factor'),
                        "win_rate": period_info.get('win_rate'),
                        "trades": period_info.get('trades'),
                        "threshold": period_info.get('threshold'),
                        "model_file": period_info.get('model_file'),
                        "data_source": period_info.get('data_source')
                    }
                    
                    # Определяем тип модели по data_source или period_key
                    period_key = period_info.get('period_key', '')
                    data_source = period_info.get('data_source', '')
                    
                    if 'existing' in period_key or data_source == 'existing_model':
                        existing_models.append(model_detail)
                        total_existing_models += 1
                    else:
                        new_models.append(model_detail)
                        total_new_models += 1
                
                # Сохраняем информацию о моделях для символа
                log_data["models_discovered"]["new_models"][symbol] = new_models
                log_data["models_discovered"]["existing_models"][symbol] = existing_models
                
                log_data["detailed_analysis"][symbol] = symbol_data
                
                # Результат оптимизации
                log_data["optimization_results"][symbol] = {
                    "optimal_period": analysis.get('period'),
                    "total_periods_tested": len(all_periods),
                    "new_models_tested": len(new_models),
                    "existing_models_tested": len(existing_models),
                    "selection_criteria": primary_metric,
                    "meets_min_trades": analysis.get('trades', 0) >= min_trades
                }
            
            # Обновляем общую статистику
            log_data["models_discovered"]["total_models_count"] = total_models_tested
            log_data["analysis_summary"].update({
                "total_models_tested": total_models_tested,
                "new_models_tested": total_new_models,
                "existing_models_tested": total_existing_models
            })
            
            # Сохраняем лог
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"📝 Детальный лог анализа сохранен: {log_path}")
            print(f"   📊 Проанализировано символов: {len(analysis_results)}")
            print(f"   🔬 Всего моделей протестировано: {total_models_tested}")
            print(f"   🆕 Новых моделей: {total_new_models}")
            print(f"   📁 Существующих моделей: {total_existing_models}")
            
            return log_path
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении лога анализа: {e}")
            return ""

def main():
    parser = argparse.ArgumentParser(description="Точечное обновление JSON watchlist")
    parser.add_argument("results_file", help="JSON файл с результатами анализа")
    parser.add_argument("--watchlist", default="watchlist.json", help="Путь к JSON watchlist")
    parser.add_argument("--metric", choices=["profit_factor", "win_rate", "trades", "score"], 
                       default="profit_factor", help="Основная метрика для выбора")
    parser.add_argument("--min-trades", type=int, default=5, help="Минимальное количество сделок")
    parser.add_argument("--keep-top-n", type=int, default=1, help="Количество лучших моделей для сохранения (по умолчанию: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Показать изменения без записи")
    parser.add_argument("--force", action="store_true", help="Обновить без подтверждения")
    parser.add_argument("--no-cleanup", action="store_true", help="Отключить автоочистку неоптимальных моделей")
    parser.add_argument("--allow-unprofitable", action="store_true", 
                       help="Разрешить выбор убыточных моделей (PF < 1.0 или Return <= 0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"❌ Файл результатов не найден: {args.results_file}")
        return
    
    if not os.path.exists(args.watchlist):
        print(f"❌ Watchlist файл не найден: {args.watchlist}")
        print(f"💡 Создайте его с помощью: python ConvertWatchlistToJSON.py")
        return
    
    # Создаем обновлятель с настройкой автоочистки
    auto_cleanup = not args.no_cleanup
    updater = WatchlistAutoUpdaterJSON(args.watchlist, auto_cleanup=auto_cleanup)
    
    success = updater.run_full_update(
        args.results_file,
        args.metric,
        args.min_trades,
        args.dry_run,
        args.force,
        not args.allow_unprofitable,  # Инвертируем флаг
        args.keep_top_n
    )
    
    if success:
        print("\\n🎯 Обновление завершено успешно!")
    else:
        print("\\n❌ Обновление завершилось с ошибками")

if __name__ == "__main__":
    main()