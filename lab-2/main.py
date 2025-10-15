

import os
import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def run_baseline_evaluation():
    """
    Етап 1: Оцінка продуктивності базової, попередньо навченої моделі YOLOv8m.
    """
    print("\n" + "="*50)
    print("Етап 1: Оцінка продуктивності базової моделі YOLOv8m...")
    print("="*50)

    try:
        # Завантаження офіційної попередньо навченої моделі YOLOv8m
        baseline_model = YOLO('yolov8m.pt')

        # Запуск валідації на вашому валідаційному наборі даних
        baseline_metrics = baseline_model.val(
            data='dataset/data.yaml',
            split='val',
            name='baseline_validation'
        )

        # Виведення ключових метрик
        print("\nРезультати базової моделі:")
        print(f"  mAP50-95: {baseline_metrics.box.map:.4f}")
        print(f"  mAP50:    {baseline_metrics.box.map50:.4f}")
        print(f"  mAP75:    {baseline_metrics.box.map75:.4f}")
        
    except Exception as e:
        print(f"Помилка на етапі оцінки базової моделі: {e}")
        print("Перевірте, чи правильно вказано шлях до 'data.yaml' та чи існує валідаційна вибірка.")

def run_training():
    """
    Етап 2: Доналаштування моделі на кастомному датасеті з аугментацією.
    """
    print("\n" + "="*50)
    print("Етап 2: Доналаштування моделі на кастомному датасеті...")
    print("="*50)

    try:
        # Завантаження моделі YOLOv8m для доналаштування
        model_to_train = YOLO('yolov8m.pt')

        # Запуск тренування з розширеними параметрами
        results = model_to_train.train(
            data='dataset/data.yaml',
            epochs=1,
            imgsz=640,
            batch=8,
            name='buses_train_augmented',
            project='training_runs',
            # Параметри аугментації
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mosaic=1.0,
            # Додаткові параметри
            patience=10,
            exist_ok=True
        )

        print("\nТренування завершено.")
        best_model_path = results.save_dir + '/weights/best.pt'
        print(f"Найкращу модель збережено у: {best_model_path}")
        return best_model_path

    except Exception as e:
        print(f"Помилка на етапі тренування: {e}")
        print("Перевірте конфігурацію 'data.yaml' та наявність тренувальних даних.")
        return None

def run_inference(path_to_best_weights):
    """
    Етап 3: Тестування доналаштованої моделі на нових зображеннях.
    """
    print("\n" + "="*50)
    print("Етап 3: Тестування фінальної моделі...")
    print("="*50)

    if path_to_best_weights is None or not os.path.exists(path_to_best_weights):
        print(f"Помилка: Файл з вагами не знайдено за шляхом: {path_to_best_weights}. Етап тестування пропущено.")
        return

    # Завантаження вашої доналаштованої моделі
    final_model = YOLO(path_to_best_weights)

    # Шлях до папки з тестовими зображеннями
    test_images_path = 'dataset/images/test/'
    
    if not os.path.isdir(test_images_path):
        print(f"Попередження: Папка для тестування '{test_images_path}' не знайдена. Етап тестування пропущено.")
        return

    # Отримання списку зображень
    image_files = glob.glob(os.path.join(test_images_path, '*.jpg')) + \
                  glob.glob(os.path.join(test_images_path, '*.png'))

    if not image_files:
        print(f"Попередження: Тестові зображення не знайдено у папці {test_images_path}")
        return

    print(f"Знайдено {len(image_files)} зображень для тестування.")
    
    # Запуск передбачень
    results = final_model.predict(source=test_images_path, save=True, conf=0.5)

    # Візуалізація результатів
    print("\nВізуалізація результатів...")
    for r in results:
        im_array = r.plot()  
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(im_rgb)
        plt.title(f"Результати для: {os.path.basename(r.path)}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    # Запуск етапу 1
    run_baseline_evaluation()
    
    # Запуск етапу 2
    best_model_path = run_training()
    
    # Запуск етапу 3
    run_inference(best_model_path)
    
    print("\n" + "="*50)
    print("Весь експеримент завершено.")
    print("="*50)