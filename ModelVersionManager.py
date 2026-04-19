#!/usr/bin/env python3
"""
Менеджер версий моделей v3
Работает только с новым форматом метаданных
"""

import os
import json
import argparse


class ModelVersionManager:
    def __init__(self, models_dir="models_v2", recommendations_file="cleanup_recommendations.json"):
        self.models_dir = models_dir
        self.recommendations_file = recommendations_file
        
    def load_recommendations(self):
        """Загружает рекомендации от AutoLauncher"""
        if not os.path.exists(self.recommendations_file):
            print(f"❌ Файл рекомендаций {self.recommendations_file} не найден!")
            print("   Сначала запустите AutoLauncher.py для генерации рекомендаций")
            return None
        
        try:
            with open(self.recommendations_file, 'r', encoding='utf-8') as f:
                recommendations = json.load(f)
            
            print(f"✅ Загружены рекомендации от {recommendations.get('timestamp', 'неизвестно')}")
            return recommendations
            
        except Exception as e:
            print(f"❌ Ошибка загрузки рекомендаций: {e}")
            return None
    
    def get_model_files(self, model_info):
        """Получает все файлы модели для удаления (включая meta и trades файлы)"""
        files_to_delete = []
        
        print(f"🔍 Анализируем файлы для модели: {model_info.get('symbol', 'unknown')}_{model_info.get('period', 'unknown')}")
        
        # Основные файлы модели
        model_file = model_info.get("model_file", "")
        meta_file = model_info.get("meta_file", "")
        
        print(f"   📄 model_file: {model_file}")
        print(f"   📄 meta_file: {meta_file}")
        
        if model_file and os.path.exists(model_file):
            files_to_delete.append(model_file)
            print(f"   ✅ Найден model_file: {os.path.basename(model_file)}")
        elif model_file:
            print(f"   ❌ model_file не существует: {model_file}")
            
        if meta_file and os.path.exists(meta_file):
            files_to_delete.append(meta_file)
            print(f"   ✅ Найден meta_file: {os.path.basename(meta_file)}")
        elif meta_file:
            print(f"   ❌ meta_file не существует: {meta_file}")
        
        # Извлекаем базовое имя из meta_file для поиска связанных файлов
        if meta_file:
            # Пример: meta_ETHUSDT_30m_500d_20250907_231407.json -> ETHUSDT_30m_500d_20250907_231407
            base_name = os.path.basename(meta_file).replace("meta_", "").replace(".json", "")
            print(f"   🔤 base_name: {base_name}")
            
            # Ищем связанные файлы
            model_dir = os.path.dirname(meta_file)
            print(f"   📁 model_dir: {model_dir}")
            
            # trades_*.csv файл
            trades_file = os.path.join(model_dir, f"trades_{base_name}.csv")
            print(f"   🔍 Ищем trades_file: {trades_file}")
            if os.path.exists(trades_file):
                files_to_delete.append(trades_file)
                print(f"   ✅ Найден trades_file: {os.path.basename(trades_file)}")
            else:
                print(f"   ℹ️  trades_file не найден (это нормально)")
        
        print(f"   📊 Всего файлов к удалению: {len(files_to_delete)}")
        return files_to_delete
    
    def cleanup_models(self, dry_run=True):
        """Выполняет очистку моделей согласно рекомендациям AutoLauncher"""
        print("🧹 ОЧИСТКА МОДЕЛЕЙ ПО РЕКОМЕНДАЦИЯМ AUTOLAUNCHER")
        print("=" * 60)
        
        recommendations = self.load_recommendations()
        if not recommendations:
            return
        
        models_to_remove = recommendations.get("models_to_remove", [])
        models_to_keep = recommendations.get("models_to_keep", [])
        
        print(f"📁 Папка моделей: {self.models_dir}")
        print(f"🔧 Режим: {'Симуляция' if dry_run else 'Реальное удаление'}")
        print(f"📊 Рекомендовано к удалению: {len(models_to_remove)} моделей")
        print(f"💾 Рекомендовано сохранить: {len(models_to_keep)} моделей")
        
        if not models_to_remove:
            print("\n✅ Нет моделей для удаления!")
            return
        
        # Группируем модели для удаления по символам
        removal_by_symbol = {}
        for model in models_to_remove:
            symbol = model["symbol"]
            if symbol not in removal_by_symbol:
                removal_by_symbol[symbol] = []
            removal_by_symbol[symbol].append(model)
        
        print("\n🔍 АНАЛИЗ УДАЛЕНИЯ ПО СИМВОЛАМ:")
        print("=" * 60)
        
        for symbol, symbol_models in removal_by_symbol.items():
            print(f"\n📊 {symbol}: удаляем {len(symbol_models)} моделей")
            
            for model in symbol_models:
                period = model["period"]
                version_info = ""
                if "version_timestamp" in model:
                    version_info = f"_{model['version_timestamp']}"
                
                cv = model.get("cv_score", 0) * 100
                wr = model.get("win_rate", 0) * 100
                pf = model.get("profit_factor", 0)
                score = model.get("quality_score", 0)
                
                print(f"   🔸 {period}d{version_info}: CV={cv:.1f}%, WR={wr:.1f}%, PF={pf:.2f}, Score={score:.3f}")
                
                # Определяем причину удаления
                reason = "Не входит в топ-3 по качеству"
                if wr == 0:
                    reason = f"Нулевой Win Rate ({wr:.1f}%)"
                elif score < 0.1:
                    reason = f"Очень низкое качество (Score={score:.3f})"
                elif cv < 35:
                    reason = f"Низкий CV ({cv:.1f}% < 35%)"
                
                print(f"      ❌ Причина: {reason}")
        
        print(f"\n{'🗑️ СИМУЛЯЦИЯ УДАЛЕНИЯ:' if dry_run else '🗑️ ВЫПОЛНЕНИЕ УДАЛЕНИЯ:'}")
        print("=" * 60)
        
        deleted_models = 0
        
        for model in models_to_remove:
            period = model["period"]
            version_info = ""
            if "version_timestamp" in model:
                version_info = f"_{model['version_timestamp']}"
            
            symbol = model["symbol"]
            model_desc = f"{symbol}_{period}d{version_info}"
            
            # Получаем файлы для удаления
            files_to_delete = self.get_model_files(model)
            
            if not files_to_delete:
                print(f"⚠️  {model_desc}: файлы не найдены")
                continue
            
            print(f"\n🔸 Удаляем модель: {model_desc}")
            
            # Определяем причину удаления
            reason = "Не входит в топ-3 по качеству"
            wr = model.get("win_rate", 0) * 100
            score = model.get("quality_score", 0)
            cv = model.get("cv_score", 0) * 100
            
            if wr == 0:
                reason = f"Нулевой Win Rate ({wr:.1f}%)"
            elif score < 0.1:
                reason = f"Очень низкое качество (Score={score:.3f})"
            elif cv < 35:
                reason = f"Низкий CV ({cv:.1f}% < 35%)"
            
            print(f"   Причина: {reason}")
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    if dry_run:
                        print(f"   📋 Будет удален: {os.path.basename(file_path)}")
                    else:
                        try:
                            os.remove(file_path)
                            print(f"   ✅ Удален: {os.path.basename(file_path)}")
                        except Exception as e:
                            print(f"   ❌ Ошибка удаления {os.path.basename(file_path)}: {e}")
            
            deleted_models += 1
        
        print(f"\n📊 ИТОГО:")
        print(f"   {'Будет удалено' if dry_run else 'Удалено'} моделей: {deleted_models}")
        
        if dry_run:
            print(f"\n⚠️  Это была симуляция. Запустите с --execute для реального удаления.")
        else:
            print(f"\n✅ Очистка завершена успешно!")


def main():
    parser = argparse.ArgumentParser(description="Менеджер версий моделей v3 (только новый формат)")
    parser.add_argument("--execute", action="store_true", 
                       help="Выполнить реальное удаление (по умолчанию - симуляция)")
    parser.add_argument("--models-dir", default="models_v2", 
                       help="Папка с моделями")
    parser.add_argument("--recommendations", default="cleanup_recommendations.json",
                       help="Файл с рекомендациями от AutoLauncher")
    
    args = parser.parse_args()
    
    manager = ModelVersionManager(
        models_dir=args.models_dir,
        recommendations_file=args.recommendations
    )
    
    # Выполняем очистку по рекомендациям AutoLauncher
    manager.cleanup_models(dry_run=not args.execute)


if __name__ == "__main__":
    main()
