#!/usr/bin/env python3
"""
WatchlistAdapter.py
Адаптер для обеспечения совместимости между TXT и JSON форматами watchlist

Функциональность:
1. Автоматическое определение формата (TXT или JSON)
2. Унифицированный интерфейс для чтения данных
3. Обратная совместимость с существующими скриптами
4. Плавный переход на JSON формат
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

# Импортируем оба парсера
from WatchlistParser import WatchlistParser, WatchlistEntry
from WatchlistJSONManager import WatchlistJSONManager

@dataclass 
class UnifiedWatchlistEntry:
    """Унифицированная запись watchlist для совместимости"""
    symbol: str
    period: int
    active: bool = True
    comment: str = ""
    source_format: str = "unknown"  # txt или json
    
    def to_watchlist_entry(self) -> WatchlistEntry:
        """Конвертирует в старый формат WatchlistEntry"""
        return WatchlistEntry(
            symbol=self.symbol,
            period=self.period,
            comment=self.comment,
            is_active=self.active
        )

class WatchlistAdapter:
    """Адаптер для работы с watchlist в любом формате"""
    
    def __init__(self, watchlist_path: str = None):
        """
        Инициализация адаптера
        
        Args:
            watchlist_path: Путь к файлу. Если None, автоматически определяет формат
        """
        
        self.watchlist_path = self._determine_watchlist_path(watchlist_path)
        self.format = self._detect_format()
        
        # Инициализируем соответствующий парсер
        if self.format == "json":
            self.json_manager = WatchlistJSONManager(self.watchlist_path)
            self.txt_parser = None
        else:
            self.txt_parser = WatchlistParser(self.watchlist_path)
            self.json_manager = None
        
        print(f"📋 Используется watchlist: {self.watchlist_path} (формат: {self.format.upper()})")
    
    def _determine_watchlist_path(self, provided_path: str) -> str:
        """Определяет путь к watchlist файлу"""
        
        if provided_path:
            return provided_path
        
        # Приоритет: JSON > TXT
        json_path = "watchlist.json"
        txt_path = "watchlist.txt"
        
        if os.path.exists(json_path):
            return json_path
        elif os.path.exists(txt_path):
            return txt_path
        else:
            # Возвращаем JSON как предпочтительный для создания нового
            return json_path
    
    def _detect_format(self) -> str:
        """Определяет формат файла"""
        
        if not os.path.exists(self.watchlist_path):
            # Определяем по расширению
            if self.watchlist_path.endswith('.json'):
                return "json"
            else:
                return "txt"
        
        # Пытаемся определить по содержимому
        try:
            with open(self.watchlist_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Проверяем JSON
            if content.startswith('{'):
                json.loads(content)  # Проверяем валидность
                return "json"
            else:
                return "txt"
                
        except json.JSONDecodeError:
            return "txt"
        except Exception:
            # По расширению как fallback
            if self.watchlist_path.endswith('.json'):
                return "json"
            else:
                return "txt"
    
    def get_all_entries(self) -> List[UnifiedWatchlistEntry]:
        """Возвращает все записи в унифицированном формате"""
        
        entries = []
        
        if self.format == "json":
            data = self.json_manager.load_watchlist()
            coins = data.get('coins', {})
            
            for symbol, coin_data in coins.items():
                entry = UnifiedWatchlistEntry(
                    symbol=symbol,
                    period=coin_data.get('period', 0),
                    active=coin_data.get('active', True),
                    comment=coin_data.get('comment', ''),
                    source_format="json"
                )
                entries.append(entry)
        
        else:  # TXT
            txt_entries = self.txt_parser.parse_watchlist()
            for txt_entry in txt_entries:
                entry = UnifiedWatchlistEntry(
                    symbol=txt_entry.symbol,
                    period=txt_entry.period,
                    active=txt_entry.is_active,
                    comment=txt_entry.comment,
                    source_format="txt"
                )
                entries.append(entry)
        
        return entries
    
    def get_active_entries(self) -> List[UnifiedWatchlistEntry]:
        """Возвращает только активные записи"""
        return [entry for entry in self.get_all_entries() if entry.active]
    
    def get_inactive_entries(self) -> List[UnifiedWatchlistEntry]:
        """Возвращает только неактивные записи"""
        return [entry for entry in self.get_all_entries() if not entry.active]
    
    def get_symbol_period(self, symbol: str) -> Optional[int]:
        """Возвращает период для символа"""
        
        if self.format == "json":
            return self.json_manager.get_coin_period(symbol)
        else:
            return self.txt_parser.get_symbol_period(symbol)
    
    def get_symbols_with_periods(self) -> Dict[str, int]:
        """Возвращает словарь символ -> период для активных монет"""
        
        result = {}
        for entry in self.get_active_entries():
            result[entry.symbol] = entry.period
        
        return result
    
    def get_legacy_entries(self) -> List[WatchlistEntry]:
        """
        Возвращает записи в старом формате WatchlistEntry
        Для обратной совместимости с существующими скриптами
        """
        
        unified_entries = self.get_all_entries()
        return [entry.to_watchlist_entry() for entry in unified_entries]
    
    def print_summary(self):
        """Выводит сводку watchlist"""
        
        all_entries = self.get_all_entries()
        active_entries = [e for e in all_entries if e.active]
        inactive_entries = [e for e in all_entries if not e.active]
        
        print(f"\\n📊 СВОДКА WATCHLIST ({self.format.upper()})")
        print("=" * 50)
        print(f"📁 Файл: {self.watchlist_path}")
        print(f"📈 Активных монет: {len(active_entries)}")
        print(f"📉 Отключенных монет: {len(inactive_entries)}")
        print(f"📋 Всего монет: {len(all_entries)}")
        
        if active_entries:
            print(f"\\n🟢 АКТИВНЫЕ МОНЕТЫ (показаны первые 5):")
            for i, entry in enumerate(sorted(active_entries, key=lambda x: x.symbol)):
                if i >= 5:
                    break
                comment_short = entry.comment[:40] if entry.comment else ""
                print(f"   {entry.symbol:12} {entry.period:4}д # {comment_short}")
            
            if len(active_entries) > 5:
                print(f"   ... и еще {len(active_entries) - 5}")
    
    def recommend_migration(self):
        """Рекомендует миграцию на JSON если используется TXT"""
        
        if self.format == "txt":
            print(f"\\n💡 РЕКОМЕНДАЦИЯ: Миграция на JSON формат")
            print("=" * 50)
            print("🚀 Преимущества JSON формата:")
            print("  ✅ Точечные обновления без потери данных")
            print("  ✅ История изменений и метаданные")
            print("  ✅ Лучшая производительность")
            print("  ✅ Расширенная валидация")
            print()
            print("🔄 Команда для миграции:")
            print(f"  python ConvertWatchlistToJSON.py --txt {self.watchlist_path} --report")
            print()
            print("📄 После миграции будет доступен экспорт в TXT для просмотра")

def create_compatible_adapter(watchlist_path: str = None) -> WatchlistAdapter:
    """
    Создает адаптер с автоматическим определением формата
    Функция для простой замены в существующих скриптах
    """
    return WatchlistAdapter(watchlist_path)

# Для обратной совместимости - можно заменить импорт
def get_watchlist_entries(watchlist_path: str = None) -> List[WatchlistEntry]:
    """
    Возвращает записи в старом формате WatchlistEntry
    Полная замена для WatchlistParser.parse_watchlist()
    """
    adapter = WatchlistAdapter(watchlist_path)
    return adapter.get_legacy_entries()

def get_active_symbols_with_periods(watchlist_path: str = None) -> Dict[str, int]:
    """
    Возвращает словарь активных символов с периодами
    Удобная функция для быстрого получения данных
    """
    adapter = WatchlistAdapter(watchlist_path)
    return adapter.get_symbols_with_periods()

def main():
    """Демонстрация работы адаптера"""
    
    print("🔀 ДЕМОНСТРАЦИЯ WATCHLIST ADAPTER")
    print("=" * 60)
    
    # Автоматическое определение формата
    adapter = WatchlistAdapter()
    
    # Показываем сводку
    adapter.print_summary()
    
    # Рекомендуем миграцию если нужно
    adapter.recommend_migration()
    
    # Тестируем совместимость
    print(f"\\n🧪 ТЕСТИРОВАНИЕ СОВМЕСТИМОСТИ:")
    symbols_periods = adapter.get_symbols_with_periods()
    print(f"📊 Активных пар: {len(symbols_periods)}")
    
    # Показываем несколько пар
    for i, (symbol, period) in enumerate(symbols_periods.items()):
        if i >= 3:
            break
        print(f"   {symbol}: {period}д")

if __name__ == "__main__":
    main()