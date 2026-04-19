#!/usr/bin/env python3
"""
CompareOptimalPeriodSelection.py
Сравнивает выбор оптимального периода на основе optimal_stats vs backtest_stats
"""

import json
import os
from pathlib import Path
from WatchlistAutoUpdaterJSON import WatchlistAutoUpdaterJSON

def compare_period_selection():
    """Сравнивает выбор периода между старой и новой логикой"""
    
    # Ищем последний файл с результатами
    search_dir = Path("optimal_period_analysis")
    if not search_dir.exists():
        search_dir = Path(".")
    
    files = list(search_dir.glob("period_search_results_*.json"))
    if not files:
        print("❌ Не найдено файлов period_search_results")
        return
    
    latest_file = sorted(files)[-1]
    print(f"📄 Анализируем: {latest_file}")
    
    # Создаем updater
    updater = WatchlistAutoUpdaterJSON()
    
    # Загружаем данные
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return
    
    experiments = data.get('experiments', {})
    
    print("\n🔄 Сравнение логики выбора периодов:")
    print("=" * 80)
    
    for symbol, symbol_data in experiments.items():
        periods_data = symbol_data.get('periods', {})
        if not periods_data:
            continue
        
        print(f"\n🪙 {symbol}:")
        
        # Имитируем старую логику (optimal_stats)
        old_choice = simulate_old_logic(periods_data)
        
        # Новая логика (backtest_stats) 
        new_choice = updater._analyze_symbol_periods(
            symbol, periods_data, "profit_factor", 30
        )
        
        if old_choice and new_choice:
            print(f"  📊 СТАРАЯ ЛОГИКА (optimal_stats):")
            print(f"      Период: {old_choice['period']}")
            print(f"      WR: {old_choice['win_rate']:.1f}%, PF: {old_choice['profit_factor']:.2f}")
            print(f"      Trades: {old_choice['trades']}")
            
            print(f"  🎯 НОВАЯ ЛОГИКА (backtest_stats):")
            print(f"      Период: {new_choice['period']}")
            print(f"      WR: {new_choice['win_rate']:.1f}%, PF: {new_choice['profit_factor']:.2f}")
            print(f"      Trades: {new_choice['trades']}")
            print(f"      Источник: {new_choice.get('data_source', 'unknown')}")
            
            if old_choice['period'] != new_choice['period']:
                print(f"  ⚠️  ИЗМЕНЕНИЕ: {old_choice['period']} → {new_choice['period']}")
            else:
                print(f"  ✅ Период остался тот же: {old_choice['period']}")
        
        print("-" * 60)

def simulate_old_logic(periods_data):
    """Имитирует старую логику выбора на основе optimal_stats"""
    
    best_period = None
    best_pf = 0
    best_stats = None
    
    for period_str, period_data in periods_data.items():
        if not period_data.get('success', False):
            continue
        
        threshold_data = period_data.get('threshold_data', {})
        stats = threshold_data.get('optimal_stats', {})
        
        if not stats:
            continue
        
        trades = stats.get('trades', 0)
        if trades < 30:  # min_trades
            continue
        
        profit_factor = stats.get('profit_factor', 0)
        if profit_factor == float('inf'):
            profit_factor = 999.0
        
        if profit_factor > best_pf:
            best_pf = profit_factor
            best_period = int(period_str)
            best_stats = {
                'period': best_period,
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': profit_factor,
                'trades': trades,
                'threshold': stats.get('threshold', 0.5)
            }
    
    return best_stats

if __name__ == "__main__":
    compare_period_selection()