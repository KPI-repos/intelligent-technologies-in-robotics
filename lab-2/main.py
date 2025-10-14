from ultralytics import YOLO

# Завантаження моделі YOLOv8m
model = YOLO('yolov8m.pt')

# Тренування на локальному датасеті
results = model.train(
    data='dataset/data.yaml',
    epochs=1,
    imgsz=640,
    batch=8,
    name='buses_train'
)

print("end")