import os
import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def run_training():
    try:
        model_to_train = YOLO('yolov8m.pt')
        results = model_to_train.train(
            data='dataset/data.yaml',
            epochs=10,
            imgsz=640,
            batch=8,
            name='buses_train_augmented',
            project='training_runs',
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mosaic=1.0,
            patience=10,
            exist_ok=True
        )
        trained_metrics = model_to_train.val(data='dataset/data.yaml')
        
        if hasattr(results, 'save_dir') and results.save_dir:
            best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        else:
            best_model_path = 'training_runs/buses_train_augmented/weights/best.pt'
        
        if os.path.exists(best_model_path):
            return best_model_path
        else:
            return None
    except Exception as e:
        return None


def run_inference(path_to_best_weights):
    if path_to_best_weights is None or not os.path.exists(path_to_best_weights):
        return

    try:
        final_model = YOLO(path_to_best_weights)
        test_images_path = 'dataset/images/test/'
        if not os.path.isdir(test_images_path):
            return

        image_files = glob.glob(os.path.join(test_images_path, '*.jpg')) + \
                      glob.glob(os.path.join(test_images_path, '*.png'))
        if not image_files:
            return

        results = final_model.predict(source=test_images_path, save=True, conf=0.5)

        for r in results:
            im_array = r.plot()
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 12))
            plt.imshow(im_rgb)
            plt.title(f"Результати для: {os.path.basename(r.path)}")
            plt.axis('off')
            plt.show()

        final_model.export(format='torchscript')
    except Exception as e:
        pass


if __name__ == '__main__':
    best_model_path = run_training()
    run_inference(best_model_path)
