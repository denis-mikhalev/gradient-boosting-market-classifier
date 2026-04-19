#!/usr/bin/env python3
"""
UpdateWatchlistWithModelInfo.py
Обновляет watchlist с информацией о файлах моделей из рекомендаций
"""

import json
import sys
from datetime import datetime

def update_watchlist_with_models(recommendations_file, watchlist_file='watchlist.json'):
    """Обновляет watchlist с информацией о файлах моделей"""
    
    # Загружаем рекомендации
    with open(recommendations_file, 'r', encoding='utf-8') as f:
        rec_data = json.load(f)

    # Загружаем watchlist
    with open(watchlist_file, 'r', encoding='utf-8') as f:
        wl_data = json.load(f)

    # Бэкап отключен: история хранится в git

    # Применяем рекомендации
    recommendations = rec_data.get('recommendations', {})
    updated_count = 0

    for symbol, rec in recommendations.items():
        if symbol in wl_data['coins']:
            old_period = wl_data['coins'][symbol]['period']
            new_period = rec['period']
            model_file = rec.get('model_file')
            
            # Обновляем основной блок монеты
            wl_data['coins'][symbol]['period'] = new_period
            
            # Добавляем информацию о модели сразу после периода
            if model_file:
                wl_data['coins'][symbol]['best_model_file'] = model_file
            
            wl_data['coins'][symbol]['comment'] = rec['comment']
            wl_data['coins'][symbol]['last_updated'] = datetime.now().isoformat()
            wl_data['coins'][symbol]['analysis_source'] = rec_data.get('metadata', {}).get('source_file', 'unknown')
            
            # Добавляем в историю оптимизации
            if 'optimization_history' not in wl_data['coins'][symbol]:
                wl_data['coins'][symbol]['optimization_history'] = []
            
            optimization_entry = {
                'date': datetime.now().isoformat(),
                'old_period': old_period,
                'new_period': new_period,
                'source': rec_data.get('metadata', {}).get('source_file', 'unknown'),
                'comment': rec['comment'],
                'metrics': rec['metrics']
            }
            
            # Добавляем файл модели в историю оптимизации
            if model_file:
                optimization_entry['best_model_file'] = model_file
                
            wl_data['coins'][symbol]['optimization_history'].append(optimization_entry)
            
            model_info = f" (модель: {model_file})" if model_file else ""
            print(f'✅ {symbol}: {old_period}д → {new_period}д{model_info}')
            print(f'   📊 {rec["metrics"]["trades"]} сделок, PF={rec["metrics"]["profit_factor"]:.2f}, WR={rec["metrics"]["win_rate"]:.1%}')
            updated_count += 1

    # Обновляем метаданные
    wl_data['metadata']['last_updated'] = datetime.now().isoformat()
    wl_data['metadata']['last_analysis_file'] = rec_data.get('metadata', {}).get('source_file', 'unknown')

    # Сохраняем обновленный watchlist
    with open(watchlist_file, 'w', encoding='utf-8') as f:
        json.dump(wl_data, f, indent=2, ensure_ascii=False)

    print(f'\n🎯 Обновлено {updated_count} монет в {watchlist_file}')
    return updated_count

def main():
    if len(sys.argv) != 2:
        print("Использование: python UpdateWatchlistWithModelInfo.py <recommendations_file.json>")
        sys.exit(1)
        
    recommendations_file = sys.argv[1]
    update_watchlist_with_models(recommendations_file)

if __name__ == "__main__":
    main()