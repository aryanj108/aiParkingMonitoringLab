import cv2
from ultralytics import YOLO
from occupancy import classify_spaces

model = YOLO("models/best.pt")

cap = cv2.VideoCapture("data/videos/parking_lot.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "data/outputs/parking_output.mp4",
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (int(cap.get(3)), int(cap.get(4)))
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.3, verbose=False)[0]

    car_boxes, space_boxes = [], []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 0:
            car_boxes.append(box.tolist())
        elif int(cls) == 1:
            space_boxes.append(box.tolist())

    occupied, vacant = classify_spaces(space_boxes, car_boxes)

    for x1, y1, x2, y2 in occupied:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    for x1, y1, x2, y2 in vacant:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Occupied: {len(occupied)} | Vacant: {len(vacant)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out.write(frame)

cap.release()
out.release()
print("✅ Video processing complete")
