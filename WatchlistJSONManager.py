#!/usr/bin/env python3
"""
WatchlistJSONManager.py
Менеджер для работы с watchlist в JSON формате

Функциональность:
1. Структурированное хранение данных о монетах
2. Точечные обновления без потери данных
3. История изменений и метаданные
4. Валидация и проверка целостности
5. Экспорт в человеко-читаемые форматы
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import shutil

@dataclass
class CoinEntry:
    """Запись о монете в watchlist"""
    symbol: str
    period: int
    active: bool = True
    comment: str = ""
    last_updated: str = ""
    analysis_source: str = "manual"
    previous_period: Optional[int] = None
    optimization_history: List[Dict] = None
    best_model_file: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()
        if self.optimization_history is None:
            self.optimization_history = []

@dataclass 
class WatchlistMetadata:
    """Метаданные watchlist"""
    version: str = "1.0"
    last_updated: str = ""
    last_analysis_file: str = ""
    total_active_coins: int = 0
    total_inactive_coins: int = 0
    last_optimization_date: str = ""
    created_date: str = ""
    format_description: str = "JSON format for market classifier watchlist with optimization tracking"
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()
        if not self.created_date:
            self.created_date = datetime.now().isoformat()

class WatchlistJSONManager:
    """Менеджер для работы с JSON watchlist"""
    
    def __init__(self, watchlist_path: str = "watchlist.json"):
        self.watchlist_path = Path(watchlist_path)
        self.backup_dir = Path("watchlist_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def load_watchlist(self) -> Dict[str, Any]:
        """Загружает watchlist из JSON файла"""
        
        if not self.watchlist_path.exists():
            # Создаем пустой watchlist
            return self._create_empty_watchlist()
        
        try:
            with open(self.watchlist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Валидируем и исправляем структуру
            was_fixed = self._validate_watchlist_structure(data)
            
            # Если были исправления, сохраняем исправленный файл
            if was_fixed:
                self.save_watchlist(data)
                print("💾 Исправленный watchlist сохранен")
            
            return data
            
        except Exception as e:
            print(f"❌ Критическая ошибка загрузки watchlist: {e}")
            print("🔄 Создаем новый пустой watchlist")
            return self._create_empty_watchlist()
    
    def _create_empty_watchlist(self) -> Dict[str, Any]:
        """Создает пустую структуру watchlist"""
        
        metadata = WatchlistMetadata()
        
        return {
            "metadata": asdict(metadata),
            "coins": {},
            "analysis_history": [],
            "schema": {
                "coin_entry": {
                    "symbol": "string (required)",
                    "period": "integer (required)", 
                    "active": "boolean (default: true)",
                    "comment": "string (optional)",
                    "last_updated": "ISO datetime string",
                    "analysis_source": "string (manual/auto)",
                    "previous_period": "integer (optional)",
                    "optimization_history": "array of optimization records",
                    "best_model_file": "string (optional, path to best model)"
                },
                "metadata": {
                    "version": "schema version",
                    "last_updated": "last modification time",
                    "last_analysis_file": "source of last analysis",
                    "total_active_coins": "count of active coins",
                    "total_inactive_coins": "count of inactive coins"
                }
            }
        }
    
    def _validate_watchlist_structure(self, data: Dict[str, Any]) -> bool:
        """Валидирует и исправляет структуру watchlist"""
        
        was_fixed = False
        
        # Проверяем обязательные ключи
        if "metadata" not in data:
            print("⚠️ Отсутствует metadata, создаем новый")
            data["metadata"] = WatchlistMetadata().__dict__
            was_fixed = True
        
        if "coins" not in data:
            print("⚠️ Отсутствует coins, создаем пустой")
            data["coins"] = {}
            was_fixed = True
        
        # Исправляем монеты вместо удаления
        fixed_coins = 0
        for symbol, coin_data in data.get("coins", {}).items():
            if "period" not in coin_data:
                print(f"⚠️ У монеты {symbol} отсутствует период, устанавливаем значение по умолчанию 400")
                coin_data["period"] = 400  # Значение по умолчанию
                fixed_coins += 1
            elif not isinstance(coin_data.get("period"), int):
                print(f"⚠️ У монеты {symbol} период не число, исправляем")
                try:
                    coin_data["period"] = int(coin_data["period"])
                except:
                    coin_data["period"] = 400
                fixed_coins += 1
                
            # Добавляем отсутствующие поля
            if "active" not in coin_data:
                coin_data["active"] = True
                fixed_coins += 1
            if "comment" not in coin_data:
                coin_data["comment"] = "Восстановлено при загрузке"
                fixed_coins += 1
            if "last_updated" not in coin_data:
                coin_data["last_updated"] = datetime.now().isoformat()
                fixed_coins += 1
            if "best_model_file" not in coin_data:
                coin_data["best_model_file"] = ""
                fixed_coins += 1
        
        if fixed_coins > 0:
            print(f"🔧 Исправлено {fixed_coins} полей в watchlist")
            was_fixed = True
        
        return was_fixed
    
    def add_or_update_coin(self, symbol: str, period: int, 
                          comment: str = "", active: bool = True,
                          analysis_source: str = "manual") -> bool:
        """Добавляет или обновляет монету"""
        
        data = self.load_watchlist()
        current_time = datetime.now().isoformat()
        
        # Проверяем, существует ли монета
        if symbol in data["coins"]:
            # Обновляем существующую
            old_coin = data["coins"][symbol]
            old_period = old_coin.get("period")
            
            # Сохраняем историю оптимизации если период изменился
            if old_period != period:
                if "optimization_history" not in old_coin:
                    old_coin["optimization_history"] = []
                
                old_coin["optimization_history"].append({
                    "date": current_time,
                    "old_period": old_period,
                    "new_period": period,
                    "source": analysis_source,
                    "comment": comment
                })
                
                old_coin["previous_period"] = old_period
            
            # Обновляем данные
            old_coin.update({
                "period": period,
                "active": active,
                "comment": comment,
                "last_updated": current_time,
                "analysis_source": analysis_source
            })
            
            print(f"✅ Обновлена {symbol}: {old_period}д → {period}д")
        else:
            # Добавляем новую монету
            coin_entry = CoinEntry(
                symbol=symbol,
                period=period,
                active=active,
                comment=comment,
                last_updated=current_time,
                analysis_source=analysis_source
            )
            
            data["coins"][symbol] = asdict(coin_entry)
            print(f"➕ Добавлена {symbol}: {period}д")
        
        # Обновляем метаданные
        self._update_metadata(data, analysis_source)
        
        return self.save_watchlist(data)
    
    def update_coins_from_analysis(self, analysis_results: Dict[str, Dict], 
                                  analysis_source: str = "") -> Dict[str, str]:
        """Обновляет только проанализированные монеты"""
        
        data = self.load_watchlist()
        update_log = {}
        
        if not analysis_results:
            print("❌ Нет данных для обновления")
            return update_log
        
        current_time = datetime.now().isoformat()
        
        # Добавляем запись в историю анализа
        if "analysis_history" not in data:
            data["analysis_history"] = []
        
        data["analysis_history"].append({
            "date": current_time,
            "source": analysis_source,
            "coins_analyzed": list(analysis_results.keys()),
            "total_coins": len(analysis_results)
        })
        
        # Обновляем только проанализированные монеты
        for symbol, analysis_data in analysis_results.items():
            comment = analysis_data.get("reason", "Оптимизирован")
            
            # Извлекаем массив моделей (единый формат)
            top_models = analysis_data.get('top_models', [])
            if not top_models:
                continue
            
            new_period = top_models[0]['period']  # Период лучшей модели
            model_file = top_models[0]['model_file']
            
            # Сохраняем массив моделей
            models_data = []
            for model in top_models:
                models_data.append({
                    "period": model['period'],
                    "model_file": model['model_file'],
                    "threshold": model.get('threshold', 0.6),
                    "profit_factor": model.get('profit_factor', 0),
                    "win_rate": model.get('win_rate', 0),
                    "trades": model.get('trades', 0),
                    "metric_value": model.get('metric_value', 0)
                })
            
            # Получаем текущие данные монеты (если есть)
            current_coin = data["coins"].get(symbol, {})
            old_period = current_coin.get("period")
            
            if symbol in data["coins"]:
                # Обновляем существующую монету
                if old_period != new_period:
                    # Период изменился
                    if "optimization_history" not in current_coin:
                        current_coin["optimization_history"] = []
                    
                    history_entry = {
                        "date": current_time,
                        "old_period": old_period,
                        "new_period": new_period,
                        "source": analysis_source,
                        "comment": comment,
                        "metrics": {
                            "profit_factor": analysis_data.get("profit_factor", 0),
                            "win_rate": analysis_data.get("win_rate", 0),
                            "trades": analysis_data.get("trades", 0),
                            "threshold": analysis_data.get("threshold", 0.6),
                            "model_file": model_file
                        }
                    }
                    
                    history_entry["top_models_count"] = len(models_data)
                    
                    current_coin["optimization_history"].append(history_entry)
                    
                    current_coin["previous_period"] = old_period
                    current_coin["period"] = new_period
                    current_coin["best_model_file"] = model_file
                    
                    update_log[symbol] = f"Обновлен: {old_period}д → топ-{len(models_data)} моделей"
                else:
                    # Период подтвержден, но обновляем модель
                    if model_file:
                        current_coin["best_model_file"] = model_file
                    
                    update_log[symbol] = f"Подтвержден: топ-{len(models_data)} моделей"
                
                # Всегда сохраняем массив моделей
                current_coin["models"] = models_data
                
                # Обновляем общие данные
                current_coin.update({
                    "comment": comment,
                    "last_updated": current_time,
                    "analysis_source": analysis_source
                })
            else:
                # Добавляем новую монету
                coin_entry = CoinEntry(
                    symbol=symbol,
                    period=new_period,
                    active=True,
                    comment=comment,
                    last_updated=current_time,
                    analysis_source=analysis_source,
                    best_model_file=model_file
                )
                
                # Добавляем массив моделей к новой монете (если есть)
                if models_data:
                    coin_entry['models'] = models_data
                
                data["coins"][symbol] = asdict(coin_entry)
                update_log[symbol] = f"Добавлена: {new_period}д"
        
        # Обновляем метаданные
        data["metadata"]["last_analysis_file"] = analysis_source
        data["metadata"]["last_optimization_date"] = current_time
        self._update_metadata(data, analysis_source)
        
        # Сохраняем
        if self.save_watchlist(data):
            return update_log
        else:
            return {}
    
    def get_active_coins(self) -> Dict[str, Dict]:
        """Возвращает только активные монеты"""
        data = self.load_watchlist()
        return {symbol: coin for symbol, coin in data["coins"].items() 
                if coin.get("active", True)}
    
    def get_inactive_coins(self) -> Dict[str, Dict]:
        """Возвращает только неактивные монеты"""
        data = self.load_watchlist()
        return {symbol: coin for symbol, coin in data["coins"].items() 
                if not coin.get("active", True)}
    
    def get_coin_period(self, symbol: str) -> Optional[int]:
        """Возвращает период для конкретной монеты"""
        data = self.load_watchlist()
        coin = data["coins"].get(symbol)
        return coin.get("period") if coin else None
    
    def deactivate_coin(self, symbol: str, reason: str = "") -> bool:
        """Деактивирует монету"""
        data = self.load_watchlist()
        
        if symbol not in data["coins"]:
            print(f"❌ Монета {symbol} не найдена")
            return False
        
        data["coins"][symbol]["active"] = False
        data["coins"][symbol]["comment"] = reason
        data["coins"][symbol]["last_updated"] = datetime.now().isoformat()
        
        self._update_metadata(data)
        return self.save_watchlist(data)
    
    def _update_metadata(self, data: Dict[str, Any], analysis_source: str = ""):
        """Обновляет метаданные"""
        
        active_count = sum(1 for coin in data["coins"].values() 
                          if coin.get("active", True))
        inactive_count = len(data["coins"]) - active_count
        
        data["metadata"].update({
            "last_updated": datetime.now().isoformat(),
            "total_active_coins": active_count,
            "total_inactive_coins": inactive_count
        })
        
        if analysis_source:
            data["metadata"]["last_analysis_file"] = analysis_source
    
    def create_backup(self) -> str:
        """Создает бэкап текущего watchlist (ОТКЛЮЧЕНО: используем git для истории)"""
        # Бэкапы отключены по требованию: история изменений ведется в git
        return ""
    
    def save_watchlist(self, data: Dict[str, Any]) -> bool:
        """Сохраняет watchlist в JSON файл"""
        
        try:
            # Бэкапы отключены — сохраняем напрямую
            with open(self.watchlist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения watchlist: {e}")
            return False
    
    def export_to_txt(self, output_path: str = "watchlist_export.txt") -> bool:
        """Экспортирует JSON в человеко-читаемый TXT формат"""
        
        data = self.load_watchlist()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Заголовок
                metadata = data["metadata"]
                f.write(f"# Market Classifier Watchlist (exported from JSON)\n")
                f.write(f"# Последнее обновление: {metadata.get('last_updated', 'N/A')}\n")
                f.write(f"# Версия: {metadata.get('version', '1.0')}\n")
                f.write(f"# Последний анализ: {metadata.get('last_analysis_file', 'N/A')}\n")
                f.write(f"#\n")
                f.write(f"# Формат: SYMBOL PERIOD # COMMENT\n")
                f.write(f"\n")
                
                # Активные монеты
                active_coins = self.get_active_coins()
                if active_coins:
                    f.write("# ============ АКТИВНЫЕ МОНЕТЫ ============\n")
                    for symbol in sorted(active_coins.keys()):
                        coin = active_coins[symbol]
                        comment = coin.get("comment", "")
                        period = coin.get("period", 0)
                        comment_str = f" # {comment}" if comment else ""
                        f.write(f"{symbol} {period}{comment_str}\n")
                
                # Неактивные монеты  
                inactive_coins = self.get_inactive_coins()
                if inactive_coins:
                    f.write("\n# ============ ОТКЛЮЧЕННЫЕ МОНЕТЫ ============\n")
                    for symbol in sorted(inactive_coins.keys()):
                        coin = inactive_coins[symbol]
                        comment = coin.get("comment", "")
                        period = coin.get("period", 0)
                        comment_str = f" # {comment}" if comment else ""
                        f.write(f"#{symbol} {period}{comment_str}\n")
                
                # Статистика
                f.write(f"\n# ============ СТАТИСТИКА ============\n")
                f.write(f"# Активных монет: {metadata.get('total_active_coins', 0)}\n")
                f.write(f"# Отключенных монет: {metadata.get('total_inactive_coins', 0)}\n")
                f.write(f"# Всего монет: {len(data['coins'])}\n")
                
                # История анализов
                if "analysis_history" in data and data["analysis_history"]:
                    f.write(f"\n# ============ ИСТОРИЯ АНАЛИЗОВ ============\n")
                    for analysis in data["analysis_history"][-3:]:  # Последние 3
                        date = analysis.get("date", "")[:19]  # Убираем микросекунды
                        source = analysis.get("source", "")
                        count = analysis.get("total_coins", 0)
                        f.write(f"# {date}: {source} ({count} монет)\n")
            
            print(f"📄 TXT экспорт создан: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка экспорта в TXT: {e}")
            return False
    
    def print_summary(self):
        """Выводит сводку по watchlist"""
        
        data = self.load_watchlist()
        metadata = data["metadata"]
        
        print("\n📊 СВОДКА WATCHLIST")
        print("=" * 50)
        print(f"📁 Файл: {self.watchlist_path}")
        print(f"📅 Последнее обновление: {metadata.get('last_updated', 'N/A')[:19]}")
        print(f"📈 Активных монет: {metadata.get('total_active_coins', 0)}")
        print(f"📉 Отключенных монет: {metadata.get('total_inactive_coins', 0)}")
        print(f"🔍 Последний анализ: {metadata.get('last_analysis_file', 'N/A')}")
        
        # Показываем несколько активных монет
        active_coins = self.get_active_coins()
        if active_coins:
            print(f"\n🟢 АКТИВНЫЕ МОНЕТЫ (показаны первые 5):")
            for i, (symbol, coin) in enumerate(sorted(active_coins.items())):
                if i >= 5:
                    break
                period = coin.get("period", 0)
                comment = coin.get("comment", "")[:50]  # Ограничиваем длину
                print(f"   {symbol:12} {period:4}д # {comment}")
            
            if len(active_coins) > 5:
                print(f"   ... и еще {len(active_coins) - 5}")

def main():
    """Демонстрация работы с JSON watchlist"""
    
    print("🚀 ДЕМОНСТРАЦИЯ JSON WATCHLIST MANAGER")
    print("=" * 60)
    
    manager = WatchlistJSONManager("watchlist.json")
    
    # Показываем текущее состояние
    manager.print_summary()
    
    # Создаем экспорт в TXT
    if manager.export_to_txt("watchlist_readable.txt"):
        print("\n✅ TXT экспорт создан для просмотра")

if __name__ == "__main__":
    main()