#!/usr/bin/env python3
"""
CleanupNonOptimalModels.py
Удаляет модели с неоптимальными периодами, оставляя только оптимальные из watchlist.json

Функциональность:
1. Читает оптимальные периоды из watchlist.json
2. Сканирует models_v2/ и находит все модели
3. Удаляет модели с периодами, которые НЕ совпадают с оптимальными
4. Создает детальный отчет об удаленных файлах
5. Опциональный dry-run режим для проверки
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import re

class NonOptimalModelsCleanup:
    """Очистка неоптимальных моделей на основе watchlist.json"""
    
    def __init__(self, models_dir: str = "models_v2", watchlist_path: str = "watchlist.json"):
        self.models_dir = Path(models_dir)
        self.watchlist_path = Path(watchlist_path)
        
    def load_optimal_models(self) -> Dict[str, List[str]]:
        """Загружает оптимальные модели из watchlist.json"""
        try:
            with open(self.watchlist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            optimal_models = {}
            for symbol, coin_data in data.get('coins', {}).items():
                if coin_data.get('active', True):  # Только активные монеты
                    # Извлекаем все model_file из массива моделей
                    if 'models' in coin_data and coin_data['models']:
                        model_files = [m['model_file'] for m in coin_data['models'] if 'model_file' in m]
                        if model_files:
                            optimal_models[symbol] = model_files
            
            print(f"Загружено оптимальных моделей из watchlist:")
            total_models = 0
            for symbol, model_files in sorted(optimal_models.items()):
                if len(model_files) > 1:
                    print(f"   {symbol}: топ-{len(model_files)} моделей")
                    for i, mf in enumerate(model_files, 1):
                        print(f"      {i}. {mf}")
                else:
                    print(f"   {symbol}: {model_files[0]}")
                total_models += len(model_files)
            
            print(f"\nВсего: {len(optimal_models)} символов, {total_models} моделей")
            return optimal_models
            
        except Exception as e:
            print(f"Ошибка загрузки {self.watchlist_path}: {e}")
            return {}
    
    def scan_models(self) -> Dict[str, List[tuple]]:
        """Сканирует models_v2/ и группирует модели по символам и периодам (расширенный парсинг)"""
        if not self.models_dir.exists():
            print(f"❌ Папка {self.models_dir} не найдена")
            return {}
        models_by_symbol = {}
        # Расширенный паттерн: xgb_SYMBOL_30m_XXXd_...
        pattern = r'xgb_([A-Z0-9]+USDT)_[^_]+_(\d+)d_.*\.json$'
        for file_path in self.models_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.json' and file_path.name.startswith('xgb_'):
                match = re.match(pattern, file_path.name)
                if match:
                    symbol = match.group(1)
                    period = int(match.group(2))
                    if symbol not in models_by_symbol:
                        models_by_symbol[symbol] = []
                    models_by_symbol[symbol].append((file_path, period))
        print(f"\n📁 Найдено моделей:")
        for symbol, models in models_by_symbol.items():
            periods = sorted([period for _, period in models])
            print(f"   {symbol}: {len(models)} файлов ({periods})")
        return models_by_symbol
    
    def identify_files_to_remove(self, optimal_models: Dict[str, List[str]], 
                                models_by_symbol: Dict[str, List[tuple]]) -> List[Path]:
        """Определяет файлы для удаления (включая связанные meta и trades файлы)"""
        files_to_remove = []
        
        print(f"\n🔍 Анализ моделей для удаления:")
        
        for symbol, models in models_by_symbol.items():
            optimal_model_files = optimal_models.get(symbol)
            if optimal_model_files is None:
                print(f"   ⚠️ {symbol}: нет в активных монетах watchlist, пропускаем все модели")
                continue
            
            for file_path, period in models:
                # Сохраняем все модели из списка optimal_model_files
                if file_path.name in optimal_model_files:
                    if len(optimal_model_files) > 1:
                        print(f"   ✅ {symbol}: {file_path.name} → ОСТАВИТЬ (топ-{len(optimal_model_files)} модель)")
                    else:
                        print(f"   ✅ {symbol}: {file_path.name} → ОСТАВИТЬ (оптимальная модель)")
                else:
                    # Удаляем все остальные модели для символа
                    files_to_remove.append(file_path)
                    base_name = file_path.stem
                    # meta файл
                    if base_name.startswith('xgb_'):
                        meta_name = base_name.replace('xgb_', 'meta_', 1) + '.json'
                        meta_path = file_path.parent / meta_name
                        if meta_path.exists():
                            files_to_remove.append(meta_path)
                    # trades файл
                    if base_name.startswith('xgb_'):
                        trades_name = base_name.replace('xgb_', 'trades_', 1) + '.csv'
                        trades_path = file_path.parent / trades_name
                        if trades_path.exists():
                            files_to_remove.append(trades_path)
                    print(f"   🗑️ {symbol}: {file_path.name} → УДАЛИТЬ (не оптимальная)")
        
        return files_to_remove
    
    def remove_files(self, files_to_remove: List[Path], dry_run: bool = False) -> int:
        """Удаляет неоптимальные файлы"""
        if not files_to_remove:
            print("✅ Нет файлов для удаления")
            return 0
        
        if dry_run:
            print(f"\n🧪 DRY RUN: Было бы удалено {len(files_to_remove)} файлов:")
            for file_path in files_to_remove:
                print(f"   - {file_path.name}")
            return 0
        
        removed_count = 0
        print(f"\n🗑️ Удаление файлов:")
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                print(f"   ✅ Удален: {file_path.name}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Ошибка удаления {file_path.name}: {e}")
        
        print(f"\n📊 Удалено файлов: {removed_count}/{len(files_to_remove)}")
        return removed_count
    
    def generate_report(self, optimal_models: Dict[str, str], 
                       files_removed: int) -> None:
        """Генерирует отчет об очистке"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = [
            f"# Отчет об очистке неоптимальных моделей",
            f"Дата: {timestamp}",
            f"Источник оптимальных моделей: {self.watchlist_path}",
            f"",
            f"## Статистика:",
            f"- Активных монет в watchlist: {len(optimal_models)}",
            f"- Удалено файлов: {files_removed}",
            f"",
            f"## Оптимальные модели:",
        ]
        
        for symbol, model_file in sorted(optimal_models.items()):
            report.append(f"- {symbol}: {model_file}")
        
        report_path = f"cleanup_nonoptimal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Отчет сохранен: {report_path}")
    
    def run_cleanup(self, dry_run: bool = False, force: bool = False) -> bool:
        """Запускает полную очистку"""
        print("ОЧИСТКА НЕОПТИМАЛЬНЫХ МОДЕЛЕЙ")
        print("=" * 60)
        
        # 1. Загружаем оптимальные модели
        optimal_models = self.load_optimal_models()
        if not optimal_models:
            return False
        
        # 2. Сканируем модели
        models_by_symbol = self.scan_models()
        if not models_by_symbol:
            return False
        
        # 3. Определяем файлы для удаления
        files_to_remove = self.identify_files_to_remove(optimal_models, models_by_symbol)
        
        if not files_to_remove:
            print("\n✅ Все модели уже оптимальны!")
            return True
        
        # 4. Показываем статистику
        print(f"\n📊 ИТОГО К УДАЛЕНИЮ: {len(files_to_remove)} файлов")
        
        # 5. Подтверждение
        if not dry_run and not force:
            print(f"\n❓ Удалить {len(files_to_remove)} неоптимальных файлов? (y/N): ", end="")
            confirmation = input().strip().lower()
            
            if confirmation not in ['y', 'yes', 'да']:
                print("🚫 Очистка отменена")
                return False
        
        # 6. Удаляем файлы
        removed_count = self.remove_files(files_to_remove, dry_run)
        
        # 7. Генерируем отчет
        if not dry_run:
            self.generate_report(optimal_models, removed_count)
        
        if dry_run:
            print("\n🧪 DRY RUN завершен - никаких изменений не внесено")
        else:
            print(f"\n✅ Очистка завершена! Удалено: {removed_count} файлов")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Очистка неоптимальных моделей")
    parser.add_argument("--models-dir", default="models_v2", help="Папка с моделями")
    parser.add_argument("--watchlist", default="watchlist.json", help="Путь к watchlist")
    parser.add_argument("--dry-run", action="store_true", help="Показать что будет удалено")
    parser.add_argument("--force", action="store_true", help="Удалить без подтверждения")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.watchlist):
        print(f"❌ Файл {args.watchlist} не найден")
        return False
    
    cleanup = NonOptimalModelsCleanup(args.models_dir, args.watchlist)
    return cleanup.run_cleanup(args.dry_run, args.force)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)