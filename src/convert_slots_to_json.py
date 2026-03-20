import os
import json
import cv2

# Paths
FRAMES_DIR = "../data/frames"
OUTPUT_JSON = "../data/ground_truth/slots_gt.json"

slots = {}
slot_id = 1

for file in os.listdir(FRAMES_DIR):
    if not file.endswith(".txt"):
        continue

    img_file = file.replace(".txt", ".jpg")
    img_path = os.path.join(FRAMES_DIR, img_file)
    label_path = os.path.join(FRAMES_DIR, file)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            # Convert YOLO → pixel coords
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            slots[f"slot_{slot_id}"] = {
                "bbox": [x1, y1, x2, y2],
                "occupied": int(cls)  # 1 = occupied, 0 = empty
            }
            slot_id += 1

os.makedirs("../data/ground_truth", exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(slots, f, indent=2)

print(f"✅ Saved {len(slots)} parking slots to {OUTPUT_JSON}")
