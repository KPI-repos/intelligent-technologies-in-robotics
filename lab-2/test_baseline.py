import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_baseline_model():
    model = YOLO('yolov8m.pt')
    test_images_path = 'dataset/images/test/'
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend([f for f in os.listdir(test_images_path) if f.lower().endswith(ext.replace('*', ''))])
    for i, image_file in enumerate(image_files):
        results = model.predict(
            source=os.path.join(test_images_path, image_file),
            conf=0.25,
            save=False
        )
        im_array = results[0].plot()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.title(f"Базова модель YOLOv8m: {image_file}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    test_baseline_model()
