import cv2
import os

VIDEO_PATH = "../data/videos/large_parking_lot.mp4"
OUTPUT_DIR = "../data/frames"
FRAME_SKIP = 10

print("Video path:", os.path.abspath(VIDEO_PATH))
print("Exists:", os.path.exists(VIDEO_PATH))

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("OpenCV could not open the video.")
    exit(1)

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_SKIP == 0:
        out_path = f"{OUTPUT_DIR}/frame_{frame_id}.jpg"
        cv2.imwrite(out_path, frame)
        saved += 1

    frame_id += 1

cap.release()
print(f"Saved {saved} frames")
