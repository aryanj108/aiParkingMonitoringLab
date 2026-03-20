import cv2
from ultralytics import YOLO
from occupancy import classify_spaces

model = YOLO("models/best.pt")

image_path = "data/images/sample.jpg"
result = model.predict(image_path, conf=0.3)[0]

car_boxes, space_boxes = [], []

for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
    if int(cls) == 0:
        car_boxes.append(box.tolist())
    elif int(cls) == 1:
        space_boxes.append(box.tolist())

occupied, vacant = classify_spaces(space_boxes, car_boxes)

print(f"Occupied: {len(occupied)} | Vacant: {len(vacant)}")
