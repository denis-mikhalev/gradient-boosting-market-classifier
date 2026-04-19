"""Анализ рыночных условий BTC"""
import sys
sys.path.insert(0, 'rl_system')
from data_loader import DataLoader
import pandas as pd

loader = DataLoader('data/cache')
df = loader.load_data('BTCUSDT', '1d', 1460)

df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
recent = df.tail(15)

recent['daily_change'] = ((recent['close'] - recent['open']) / recent['open'] * 100).round(2)
recent['volatility'] = ((recent['high'] - recent['low']) / recent['open'] * 100).round(2)

print('\n📊 ПОСЛЕДНИЕ 15 ДНЕЙ BTCUSDT:')
print('='*80)
print(f"{'Date':<12} {'Open':>10} {'Close':>10} {'Change':>10} {'Volatility':>12}")
print('-'*80)

for _, row in recent.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    change_icon = '🟢' if row['daily_change'] > 0 else '🔴' if row['daily_change'] < 0 else '⚪'
    vol_icon = '🔥' if row['volatility'] > 3 else '📊' if row['volatility'] > 2 else '💤'
    
    print(f"{date_str:<12} {row['open']:>10.0f} {row['close']:>10.0f} {change_icon} {row['daily_change']:>+6.2f}% {vol_icon} {row['volatility']:>7.2f}%")

print('='*80)

# Статистика по периодам
before_17 = df[df['date'] < '2025-11-17']
after_17 = df[df['date'] >= '2025-11-17']

before_17['daily_change'] = ((before_17['close'] - before_17['open']) / before_17['open'] * 100)
before_17['volatility'] = ((before_17['high'] - before_17['low']) / before_17['open'] * 100)

after_17['daily_change'] = ((after_17['close'] - after_17['open']) / after_17['open'] * 100)
after_17['volatility'] = ((after_17['high'] - after_17['low']) / after_17['open'] * 100)

print(f"\n📈 ДО 17 НОЯБРЯ (обучение старой модели):")
print(f"  Средняя волатильность: {before_17['volatility'].mean():.2f}%")
print(f"  Средн. изменение: {before_17['daily_change'].abs().mean():.2f}%")
print(f"  Макс. волатильность: {before_17['volatility'].max():.2f}%")

print(f"\n📉 ПОСЛЕ 17 НОЯБРЯ (новые данные):")
print(f"  Средняя волатильность: {after_17['volatility'].mean():.2f}%")
print(f"  Средн. изменение: {after_17['daily_change'].abs().mean():.2f}%")
print(f"  Макс. волатильность: {after_17['volatility'].max():.2f}%")

vol_drop = ((after_17['volatility'].mean() - before_17['volatility'].mean()) / before_17['volatility'].mean() * 100)
print(f"\n💡 Падение волатильности: {vol_drop:+.1f}%")

if vol_drop < -20:
    print("\n❌ ВЕРДИКТ: Рынок стал ЗНАЧИТЕЛЬНО менее волатильным")
    print("   Агент правильно учится НЕ торговать в таких условиях!")
elif vol_drop < -10:
    print("\n⚠️  ВЕРДИКТ: Рынок стал менее волатильным")
    print("   Торговля становится менее прибыльной")
else:
    print("\n✅ ВЕРДИКТ: Волатильность примерно одинаковая")
    print("   Проблема не в данных, а в чём-то другом")
