from ultralytics import YOLO

#loading classification model
model = YOLO('yolo11n-cls.pt')

#train the model
results = model.train(data='/content/Rock-Paper-Scissors-1', epochs=20, imgsz=640)