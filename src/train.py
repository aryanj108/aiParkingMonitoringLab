from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # use small model for now

model.train(
   data="../data/parking_detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    name="parking_train"
)



