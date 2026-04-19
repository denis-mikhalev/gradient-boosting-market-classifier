"""
Централизованный менеджер heartbeat для торговой системы
Отправляет единый heartbeat со всеми активными моделями раз в час
Отслеживает реально запущенные процессы Predict-Advanced.py
"""
import os
import json
import time
import glob
import psutil
from datetime import datetime, timedelta
from telegram_sender import send_heartbeat


class HeartbeatManager:
    def __init__(self):
        self.heartbeat_interval = 1800  # 30 минут в секундах
        self.last_heartbeat = datetime.now()
        self.start_time = datetime.now()
        self.status_dir = "status"
        
        # Создаем папку для статуса, если её нет
        if not os.path.exists(self.status_dir):
            os.makedirs(self.status_dir)
    
    def get_active_models_from_processes(self):
        """Находит все активные процессы Predict-Advanced.py и извлекает их параметры"""
        active_models = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    info = proc.info
                    if info['name'] and 'python' in info['name'].lower():
                        cmdline = info['cmdline']
                        if cmdline:
                            # Проверяем, есть ли Predict-Advanced.py в командной строке
                            cmdline_str = ' '.join(cmdline)
                            if 'Predict-Advanced.py' in cmdline_str:
                                symbol = None
                                
                                # Извлекаем только символ для простого мониторинга
                                for i, arg in enumerate(cmdline):
                                    if arg == '--symbol' and i + 1 < len(cmdline):
                                        symbol = cmdline[i + 1]
                                        break  # Нашли символ - больше ничего не нужно
                                
                                if symbol:
                                    # Простое отображение только символа
                                    active_models.append(symbol)
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError) as e:
                    continue
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка при сканировании процессов: {e}")
        
        # Убираем дубликаты
        unique_models = list(set(active_models))
        return unique_models
    

    
    def get_active_models(self):
        """Получает список активных моделей только из реальных процессов"""
        # Получаем только реально запущенные процессы
        process_models = self.get_active_models_from_processes()
        
        if process_models:
            return process_models
        else:
            print("⚠️ Реальные процессы не найдены - система не запущена")
            return []
    
    def should_send_heartbeat(self):
        """Проверяет, нужно ли отправлять heartbeat"""
        current_time = datetime.now()
        return (current_time - self.last_heartbeat).total_seconds() >= self.heartbeat_interval
    
    def send_unified_heartbeat(self):
        """Отправляет объединенный heartbeat со всеми активными моделями"""
        try:
            active_models = self.get_active_models()
            
            if not active_models:
                print("⚠️ Активные модели не найдены (система не запущена), heartbeat не отправляется")
                return False
            
            current_time = datetime.now()
            uptime = str(current_time - self.start_time).split('.')[0]
            print(f"Отправка heartbeat для {len(active_models)} моделей: {', '.join(active_models)}")
            success = send_heartbeat(active_models, uptime)
            
            if success:
                self.last_heartbeat = current_time
                print(f"Объединенный heartbeat отправлен для {len(active_models)} моделей: {', '.join(active_models)}")
            else:
                print("❌ Ошибка отправки объединенного heartbeat")
                
            return success
            
        except Exception as e:
            print(f"❌ Ошибка heartbeat: {e}")
            return False
    
    def run(self):
        """Основной цикл менеджера heartbeat"""
        print("HeartbeatManager запущен")
        print(f"Интервал heartbeat: {self.heartbeat_interval} секунд (1 час)")
        print("Мониторинг реально запущенных процессов Predict-Advanced.py")
        
        # Отправляем начальный heartbeat через 10 секунд
        print("Ожидание 10 секунд перед первым heartbeat...")
        time.sleep(10)
        self.send_unified_heartbeat()
        
        while True:
            try:
                # Проверяем активные модели каждую минуту для отладки
                active_models = self.get_active_models()
                current_time = datetime.now()
                
                print(f"{current_time.strftime('%H:%M:%S')} | HeartbeatManager активен | Найдено процессов: {len(active_models)}")
                if active_models:
                    print(f"    Активные процессы: {', '.join(active_models)}")
                else:
                    print("   ⚠️ Торговая система не запущена - процессы Predict-Advanced.py не найдены")
                
                if self.should_send_heartbeat():
                    self.send_unified_heartbeat()
                
                # Проверяем каждые 60 секунд
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n🛑 HeartbeatManager остановлен пользователем")
                break
            except Exception as e:
                print(f"❌ Ошибка в HeartbeatManager: {e}")
                time.sleep(60)


if __name__ == "__main__":
    manager = HeartbeatManager()
    manager.run()
