#!/usr/bin/env python3
"""
WatchlistParser - Утилита для работы с JSON форматом watchlist.json
Поддерживает:
- Указание периода для каждой монеты
- Активация/деактивация монет
- Комментарии и метаданные
- Отслеживание истории оптимизации
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class WatchlistEntry:
    """Запись в watchlist"""
    symbol: str
    period: int
    comment: str = ""
    is_active: bool = True

class WatchlistParser:
    """Парсер JSON формата watchlist.json"""
    
    def __init__(self, watchlist_path: str = "watchlist.json"):
        self.watchlist_path = watchlist_path
        self.default_period = 365  # По умолчанию если период не указан
        
    def parse_watchlist(self) -> List[WatchlistEntry]:
        """Парсит JSON файл watchlist.json"""
        if not os.path.exists(self.watchlist_path):
            print(f"❌ Файл {self.watchlist_path} не найден")
            return []

        entries = []
        
        try:
            with open(self.watchlist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Обрабатываем секцию coins
            if 'coins' not in data:
                print(f"❌ Неверный формат JSON: отсутствует секция 'coins'")
                return []
                
            for symbol, config in data['coins'].items():
                try:
                    # Пропускаем неактивные монеты
                    if not config.get('active', True):
                        continue
                        
                    # Получаем период (может отсутствовать у некоторых монет)
                    period = config.get('period', self.default_period)
                    comment = config.get('comment', '')
                    
                    entry = WatchlistEntry(
                        symbol=symbol,
                        period=period,
                        comment=comment,
                        is_active=True
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    print(f"⚠️ Ошибка обработки монеты {symbol}: {e}")
                    
            return entries
                
        except Exception as e:
            print(f"❌ Ошибка чтения файла {self.watchlist_path}: {e}")
            return []
        
    def get_active_symbols(self) -> List[str]:
        """Возвращает список активных символов"""
        entries = self.parse_watchlist()
        return [entry.symbol for entry in entries if entry.is_active]
    
    def get_active_entries(self) -> List[WatchlistEntry]:
        """Возвращает список активных записей"""
        entries = self.parse_watchlist()
        return [entry for entry in entries if entry.is_active]
    
    def get_symbol_config(self, symbol: str) -> Optional[WatchlistEntry]:
        """Получает конфигурацию для конкретного символа"""
        entries = self.parse_watchlist()
        for entry in entries:
            if entry.symbol == symbol and entry.is_active:
                return entry
        return None
    
    def get_symbol_period(self, symbol: str) -> int:
        """Получает период для символа или возвращает дефолтный"""
        config = self.get_symbol_config(symbol)
        return config.period if config else self.default_period
    
    def print_summary(self):
        """Выводит сводку по watchlist"""
        entries = self.parse_watchlist()
        active_entries = [e for e in entries if e.is_active]
        inactive_entries = [e for e in entries if not e.is_active]
        
        print("📋 СВОДКА ПО WATCHLIST")
        print("=" * 50)
        print(f"📁 Файл: {self.watchlist_path}")
        print(f"📊 Всего записей: {len(entries)}")
        print(f"✅ Активных: {len(active_entries)}")
        print(f"❌ Отключенных: {len(inactive_entries)}")
        
        if active_entries:
            print(f"\n✅ АКТИВНЫЕ МОНЕТЫ:")
            for entry in active_entries:
                comment_str = f" # {entry.comment}" if entry.comment else ""
                print(f"   🔸 {entry.symbol} ({entry.period} дней){comment_str}")
        
        if inactive_entries:
            print(f"\n❌ ОТКЛЮЧЕННЫЕ МОНЕТЫ:")
            for entry in inactive_entries:
                comment_str = f" # {entry.comment}" if entry.comment else ""
                print(f"   🔸 {entry.symbol} ({entry.period} дней){comment_str}")

def main():
    """Тестирование парсера"""
    parser = WatchlistParser()
    
    print("🧪 ТЕСТИРОВАНИЕ WATCHLIST PARSER")
    print("=" * 60)
    
    # Выводим сводку
    parser.print_summary()
    
    # Примеры использования
    print(f"\n🔍 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
    print(f"Активные символы: {parser.get_active_symbols()}")
    
    test_symbols = ["BTCUSDT", "ADAUSDT", "POLUSDT"]
    for symbol in test_symbols:
        period = parser.get_symbol_period(symbol)
        config = parser.get_symbol_config(symbol)
        status = "активна" if config and config.is_active else "неактивна"
        print(f"{symbol}: {period} дней ({status})")

if __name__ == "__main__":
    main()