#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CleanupUnprofitableModels.py
Удаляет убыточные модели из models_v2/ на основе их метаданных

Функциональность:
1. Сканирует все meta-файлы в models_v2/
2. Анализирует backtest_results каждой модели
3. Удаляет модели с profit_factor < 1.0 или total_return_pct <= 0
4. Создает детальный отчет об удаленных файлах
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import re

# Принудительно устанавливаем UTF-8 для вывода
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class UnprofitableModelsCleanup:
    """Очистка убыточных и низкокачественных моделей на основе их результатов"""
    
    def __init__(self, models_dir: str = "models_v2",
                 min_profit_factor: float = 1.2,
                 min_win_rate: float = 0.55,
                 max_drawdown: float = 0.25,
                 min_total_return: float = 0.02,
                 min_trades: int = 10):
        """
        Инициализация с настраиваемыми критериями качества
        
        Args:
            models_dir: директория с моделями
            min_profit_factor: минимальный profit factor (по умолчанию 1.2)
            min_win_rate: минимальный win rate (по умолчанию 55%)
            max_drawdown: максимальный drawdown (по умолчанию 25%)
            min_total_return: минимальная доходность (по умолчанию 2%)
            min_trades: минимальное количество сделок (по умолчанию 10)
        """
        self.models_dir = Path(models_dir)
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.min_total_return = min_total_return
        self.min_trades = min_trades
        
    def analyze_model_profitability(self, meta_file: Path) -> dict:
        """
        Анализирует прибыльность и качество модели по мета-файлу
        """
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            backtest_results = meta_data.get('backtest_results', {})
            if not backtest_results:
                return {'profitable': False, 'reason': 'no_backtest_data'}
            
            profit_factor = backtest_results.get('profit_factor', 0)
            total_return_pct = backtest_results.get('total_return_pct', 0)
            win_rate = backtest_results.get('win_rate', 0)
            total_trades = backtest_results.get('total_trades', 0)
            max_drawdown_pct = backtest_results.get('max_drawdown_pct', 1.0)  # По умолчанию 100% если нет данных
            # Приводим drawdown к процентам, если он в долях (например, 0.0202 -> 2.02)
            if max_drawdown_pct <= 1.0:
                max_drawdown_pct = max_drawdown_pct * 100
            
            # Проверяем каждый критерий качества
            failed_criteria = []
            
            if profit_factor < self.min_profit_factor:
                failed_criteria.append(f"PF={profit_factor:.2f}<{self.min_profit_factor}")
            
            if total_return_pct <= 0:
                failed_criteria.append(f"Return={total_return_pct*100:.1f}%<=0")
            elif total_return_pct < self.min_total_return:
                failed_criteria.append(f"Return={total_return_pct*100:.1f}%<{self.min_total_return*100:.1f}%")
            
            if win_rate < self.min_win_rate:
                failed_criteria.append(f"WinRate={win_rate*100:.1f}%<{self.min_win_rate*100:.1f}%")
            
            if max_drawdown_pct > self.max_drawdown * 100:
                failed_criteria.append(f"Drawdown={max_drawdown_pct:.2f}%>{self.max_drawdown*100:.2f}%")
            
            if total_trades < self.min_trades:
                failed_criteria.append(f"Trades={total_trades}<{self.min_trades}")
            
            # Модель считается качественной только если проходит ВСЕ критерии
            is_profitable = len(failed_criteria) == 0
            
            reason = 'high_quality' if is_profitable else f"failed_criteria: {', '.join(failed_criteria)}"
            
            return {
                'profitable': is_profitable,
                'profit_factor': profit_factor,
                'total_return_pct': total_return_pct * 100,  # В процентах
                'win_rate': win_rate * 100,  # В процентах
                'total_trades': total_trades,
                'max_drawdown_pct': max_drawdown_pct,  # Уже в процентах
                'failed_criteria': failed_criteria,
                'reason': reason
            }
            
        except Exception as e:
            return {'profitable': False, 'reason': f'error: {e}'}
    
    def find_model_files(self, base_name: str) -> List[Path]:
        """Находит все файлы связанные с моделью по базовому имени"""
        files = []
        
        # Паттерны файлов модели
        patterns = [
            f"xgb_{base_name}.json",      # Основной файл модели
            f"meta_{base_name}.json",     # Мета-файл
            f"trades_{base_name}.csv",    # Файл торговых данных
        ]
        
        for pattern in patterns:
            matching_files = list(self.models_dir.glob(pattern))
            files.extend(matching_files)
        
        return files
    
    def run_cleanup(self, dry_run: bool = False) -> dict:
        """Запускает очистку убыточных моделей"""
        print("🧹 ОЧИСТКА МОДЕЛЕЙ ПО КАЧЕСТВЕННЫМ КРИТЕРИЯМ")
        print("=" * 70)
        print(f"📁 Директория: {self.models_dir}")
        print(f"🔧 Режим: {'🔍 ПРОСМОТР (DRY RUN)' if dry_run else '🗑️ УДАЛЕНИЕ'}")
        print()
        print("📋 КРИТЕРИИ КАЧЕСТВА:")
        print(f"   • Profit Factor ≥ {self.min_profit_factor}")
        print(f"   • Win Rate ≥ {self.min_win_rate*100:.0f}%") 
        print(f"   • Max Drawdown ≤ {self.max_drawdown*100:.0f}%")
        print(f"   • Total Return ≥ {self.min_total_return*100:.0f}%")
        print(f"   • Min Trades ≥ {self.min_trades}")
        print("=" * 70)
        print()
        
        if not self.models_dir.exists():
            print(f"❌ ОШИБКА: Директория {self.models_dir} не найдена")
            return {'success': False}
        
        # Ищем все мета-файлы
        meta_files = list(self.models_dir.glob("meta_*.json"))
        
        if not meta_files:
            print(f"❌ ОШИБКА: Мета-файлы не найдены в {self.models_dir}")
            return {'success': False}
        
        print(f"📊 Найдено {len(meta_files)} моделей для анализа")
        print("🔍 Начинаем анализ качества...")
        print()
        
        stats = {
            'total_analyzed': 0,
            'profitable': 0,
            'unprofitable': 0,
            'errors': 0,
            'deleted_files': 0,
            'deleted_models': 0
        }
        
        deleted_models = []
        
        for meta_file in sorted(meta_files):
            # Извлекаем базовое имя модели
            meta_name = meta_file.stem  # meta_BTCUSDT_30m_200d_bt30d_20250920_135319
            base_name = meta_name[5:]   # BTCUSDT_30m_200d_bt30d_20250920_135319
            
            # Извлекаем символ и период из имени для более понятного вывода
            parts = base_name.split('_')
            symbol = parts[0] if parts else 'UNKNOWN'
            period = 'Unknown'
            for part in parts:
                if part.endswith('d') and part[:-1].isdigit():
                    period = part
                    break
            
            print(f"🔍 Анализ: {symbol} ({period}) - {base_name}")
            
            # Анализируем прибыльность
            analysis = self.analyze_model_profitability(meta_file)
            stats['total_analyzed'] += 1
            
            if analysis['reason'].startswith('error'):
                print(f"   💥 ОШИБКА анализа: {analysis['reason']}")
                stats['errors'] += 1
                continue
            
            if analysis['reason'] == 'no_backtest_data':
                print(f"   ⚠️ ПРЕДУПРЕЖДЕНИЕ: Нет данных бэктеста - пропускаем")
                stats['errors'] += 1
                continue
            
            if analysis['profitable']:
                # Модель высокого качества - показываем детальную статистику
                print(f"   ✅ ВЫСОКОЕ КАЧЕСТВО - СОХРАНЯЕМ:")
                print(f"      📊 PF={analysis['profit_factor']:.2f}, Return={analysis['total_return_pct']:.1f}%, WinRate={analysis['win_rate']:.1f}%")
                print(f"      📈 Drawdown={analysis['max_drawdown_pct']:.1f}%, Trades={analysis['total_trades']}")
                stats['profitable'] += 1
            else:
                # Модель низкого качества - показываем причины отклонения
                print(f"   ❌ НИЗКОЕ КАЧЕСТВО - УДАЛЯЕМ:")
                print(f"      📊 PF={analysis['profit_factor']:.2f}, Return={analysis['total_return_pct']:.1f}%, WinRate={analysis['win_rate']:.1f}%")
                print(f"      📈 Drawdown={analysis['max_drawdown_pct']:.1f}%, Trades={analysis['total_trades']}")
                if analysis['failed_criteria']:
                    print(f"      🚫 НЕ ПРОШЛА КРИТЕРИИ: {', '.join(analysis['failed_criteria'])}")
                    
                    # Объясняем требования пресета
                    print(f"      📋 ТРЕБУЕТСЯ: PF≥{self.min_profit_factor}, WR≥{self.min_win_rate*100:.0f}%, " +
                          f"DD≤{self.max_drawdown*100:.0f}%, Return≥{self.min_total_return*100:.0f}%, Trades≥{self.min_trades}")
                
                stats['unprofitable'] += 1
                
                # Находим все файлы этой модели
                model_files = self.find_model_files(base_name)
                
                if model_files:
                    deleted_models.append({
                        'base_name': base_name,
                        'symbol': symbol,
                        'period': period,
                        'files': [str(f) for f in model_files],
                        'analysis': analysis
                    })
                    
                    if dry_run:
                        print(f"   🗑️ БУДЕТ УДАЛЕНО: {len(model_files)} файлов")
                        for f in model_files:
                            print(f"      - {f.name}")
                    else:
                        print(f"   🗑️ УДАЛЯЕМ: {len(model_files)} файлов")
                        deleted_count = 0
                        for f in model_files:
                            try:
                                f.unlink()
                                print(f"      ✅ УДАЛЕН: {f.name}")
                                stats['deleted_files'] += 1
                                deleted_count += 1
                            except Exception as e:
                                print(f"      ❌ ОШИБКА УДАЛЕНИЯ: {f.name}: {e}")
                        
                        if deleted_count > 0:
                            stats['deleted_models'] += 1
                        print(f"   🎯 {symbol} ({period}): удалено {deleted_count}/{len(model_files)} файлов")
            
            print()
        
        # Итоговая статистика
        print("🏁 ИТОГОВАЯ СТАТИСТИКА:")
        print("=" * 50)
        print(f"📊 Проанализировано моделей: {stats['total_analyzed']}")
        print(f"✅ Качественных (сохранено): {stats['profitable']}")
        print(f"❌ Некачественных: {stats['unprofitable']}")
        print(f"💥 Ошибок анализа: {stats['errors']}")
        
        if stats['unprofitable'] > 0:
            print()
            print("🗑️ ДЕТАЛИ УДАЛЕНИЯ:")
            print("-" * 30)
            
            # Группируем по символам для лучшего отображения
            by_symbol = {}
            for model in deleted_models:
                symbol = model['symbol']
                if symbol not in by_symbol:
                    by_symbol[symbol] = []
                by_symbol[symbol].append(model)
            
            for symbol, models in by_symbol.items():
                print(f"📈 {symbol}: удалено {len(models)} моделей")
                for model in models:
                    analysis = model['analysis']
                    print(f"   • {model['period']}: {', '.join(analysis['failed_criteria'])}")
        
        print()
        if dry_run:
            print(f"🔍 РЕЖИМ ПРОСМОТРА: К удалению {len(deleted_models)} моделей")
            print("💡 Для реального удаления запустите без --dry-run")
        else:
            print(f"✅ УДАЛЕНИЕ ЗАВЕРШЕНО:")
            print(f"   🗑️ Удалено моделей: {stats['deleted_models']}")
            print(f"   📁 Удалено файлов: {stats['deleted_files']}")
            
            if stats['deleted_models'] > 0:
                print(f"   💾 Освобождено места: ~{stats['deleted_files'] * 0.5:.1f} MB")
        
        return {
            'success': True,
            'stats': stats,
            'deleted_models': deleted_models
        }

def main():
    parser = argparse.ArgumentParser(description='Очистка убыточных и низкокачественных моделей')
    parser.add_argument('--models-dir', default='models_v2', help='Директория с моделями')
    parser.add_argument('--dry-run', action='store_true', help='Показать что будет удалено без фактического удаления')
    
    # Критерии качества моделей
    parser.add_argument('--min-profit-factor', type=float, default=1.2, 
                       help='Минимальный profit factor (по умолчанию: 1.2)')
    parser.add_argument('--min-win-rate', type=float, default=0.55,
                       help='Минимальный win rate в долях (по умолчанию: 0.55 = 55%%)')
    parser.add_argument('--max-drawdown', type=float, default=0.25,
                       help='Максимальный drawdown в долях (по умолчанию: 0.25 = 25%%)')
    parser.add_argument('--min-total-return', type=float, default=0.02,
                       help='Минимальная доходность в долях (по умолчанию: 0.02 = 2%%)')
    parser.add_argument('--min-trades', type=int, default=10,
                       help='Минимальное количество сделок (по умолчанию: 10)')
    
    # Пресеты для разных уровней строгости
    parser.add_argument('--preset', choices=['conservative', 'balanced', 'aggressive'], 
                       help='Предустановленные критерии качества')
    
    args = parser.parse_args()
    
    # Применяем пресеты если указаны
    # ВАЖНО: min_trades из аргументов имеет приоритет над пресетом
    min_trades_from_args = args.min_trades  # Сохраняем значение до применения пресета
    
    if args.preset == 'conservative':
        # Очень строгие критерии для консервативной торговли
        args.min_profit_factor = 1.5
        args.min_win_rate = 0.65
        args.max_drawdown = 0.15
        args.min_total_return = 0.05
        # Не перезаписываем min_trades если он был явно передан (не равен дефолту 10)
        if min_trades_from_args == 10:  # Дефолтное значение
            args.min_trades = 30
        print(f"📈 Применен КОНСЕРВАТИВНЫЙ пресет: PF≥1.5, WR≥65%, DD≤15%, Return≥5%, Trades≥{args.min_trades}")
    elif args.preset == 'balanced':
        # Сбалансированные критерии
        args.min_profit_factor = 1.2
        args.min_win_rate = 0.55
        args.max_drawdown = 0.25
        args.min_total_return = 0.02
        if min_trades_from_args == 10:
            args.min_trades = 30
        print(f"⚖️ Применен СБАЛАНСИРОВАННЫЙ пресет: PF≥1.2, WR≥55%, DD≤25%, Return≥2%, Trades≥{args.min_trades}")
    elif args.preset == 'aggressive':
        # Менее строгие критерии для агрессивной торговли
        args.min_profit_factor = 1.1
        args.min_win_rate = 0.45
        args.max_drawdown = 0.35
        args.min_total_return = 0.01
        if min_trades_from_args == 10:
            args.min_trades = 30
        print(f"🚀 Применен АГРЕССИВНЫЙ пресет: PF≥1.1, WR≥45%, DD≤35%, Return≥1%, Trades≥{args.min_trades}")
    
    cleanup = UnprofitableModelsCleanup(
        args.models_dir,
        min_profit_factor=args.min_profit_factor,
        min_win_rate=args.min_win_rate,
        max_drawdown=args.max_drawdown,
        min_total_return=args.min_total_return,
        min_trades=args.min_trades
    )
    
    # Показываем текущие критерии только если запущено напрямую (не из OptimalPeriodSearch)
    if not args.preset or len(sys.argv) <= 3:  # Если мало аргументов, значит запущено напрямую
        print(f"\n🎯 КРИТЕРИИ КАЧЕСТВА МОДЕЛЕЙ:")
        print(f"   Profit Factor ≥ {args.min_profit_factor}")
        print(f"   Win Rate ≥ {args.min_win_rate*100:.1f}%")
        print(f"   Max Drawdown ≤ {args.max_drawdown*100:.1f}%")
        print(f"   Total Return ≥ {args.min_total_return*100:.1f}%")
        print(f"   Min Trades ≥ {args.min_trades}")
    
    result = cleanup.run_cleanup(args.dry_run)
    
    if result['success']:
        print(f"\nОчистка завершена!")
        if args.dry_run:
            print(f"СОВЕТ: Запустите без --dry-run для фактического удаления")
    else:
        print(f"\nОчистка завершилась с ошибками")

if __name__ == "__main__":
    main()