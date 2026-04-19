"""Сравнение волатильности на разных таймфреймах"""
import sys
sys.path.insert(0, 'rl_system')
from data_loader import DataLoader
import pandas as pd

loader = DataLoader('data/cache')

timeframes = [
    ('1d', 365, '1 день'),
    ('12h', 365, '12 часов'),
    ('8h', 365, '8 часов'),
    ('6h', 365, '6 часов'),
    ('4h', 365, '4 часа'),
    ('1h', 180, '1 час')
]

print('\n📊 СРАВНЕНИЕ ВОЛАТИЛЬНОСТИ ПО ТАЙМФРЕЙМАМ')
print('='*80)

results = []

for tf, days, name in timeframes:
    try:
        df = loader.load_data('BTCUSDT', tf, days)
        
        # Последние 100 баров
        recent = df.tail(100).copy()
        recent['volatility'] = ((recent['high'] - recent['low']) / recent['open'] * 100)
        recent['daily_change'] = ((recent['close'] - recent['open']) / recent['open'] * 100).abs()
        
        avg_vol = recent['volatility'].mean()
        avg_change = recent['daily_change'].mean()
        max_vol = recent['volatility'].max()
        
        # Считаем "торговые возможности" (бары с волатильностью >2%)
        tradeable_bars = (recent['volatility'] > 2.0).sum()
        tradeable_pct = (tradeable_bars / len(recent)) * 100
        
        results.append({
            'timeframe': name,
            'avg_vol': avg_vol,
            'avg_change': avg_change,
            'max_vol': max_vol,
            'tradeable_pct': tradeable_pct,
            'bars': len(df)
        })
        
        print(f"\n{name} (последние 100 баров):")
        print(f"  Всего баров: {len(df)}")
        print(f"  Средняя волатильность: {avg_vol:.2f}%")
        print(f"  Среднее изменение: {avg_change:.2f}%")
        print(f"  Макс. волатильность: {max_vol:.2f}%")
        print(f"  Торговых возможностей (>2% vol): {tradeable_pct:.1f}%")
        
    except Exception as e:
        print(f"\n{name}: ❌ Ошибка загрузки - {e}")

print('\n' + '='*80)
print('\n💡 РЕКОМЕНДАЦИЯ:')

if results:
    best = max(results, key=lambda x: x['tradeable_pct'])
    print(f"  🏆 Лучший таймфрейм: {best['timeframe']}")
    print(f"     • {best['tradeable_pct']:.1f}% баров с хорошей волатильностью")
    print(f"     • Средняя волатильность: {best['avg_vol']:.2f}%")
    print(f"     • Достаточно данных: {best['bars']} баров")
    
    print(f"\n  📈 Прогноз обучения:")
    if best['tradeable_pct'] > 40:
        print(f"     ✅ ОТЛИЧНО - Агент должен найти много торговых возможностей")
    elif best['tradeable_pct'] > 25:
        print(f"     ⚠️  СРЕДНЕ - Агент может научиться торговать")
    else:
        print(f"     ❌ ПЛОХО - Мало торговых возможностей, агент будет HOLD")
