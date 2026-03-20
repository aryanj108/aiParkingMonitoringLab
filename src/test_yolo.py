from ultralytics import YOLO
import os
from datetime import datetime

# Load the trained YOLO model
model = YOLO("runs/detect/parking_train3/weights/best.pt")

# Paths for input
frames_path = "../data/frames"
video_path = "../data/videos/large_parking_lot.mp4"

# Create timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output folders with timestamp
frames_output = f"runs/detect/predict_frames/run_{timestamp}"
video_output = f"runs/detect/predict_video/run_{timestamp}"

# Make sure input exists
if not os.path.exists(frames_path) or len(os.listdir(frames_path)) == 0:
    print(f"No images found in {frames_path}")
else:
    # Predict on frames silently
    model.predict(
        source=frames_path,
        conf=0.25,
        save=True,
        show=False,  # <-- prevents opening images
        project=frames_output,
        name="results"
    )
    print(f"Frames predictions saved in: {frames_output}/results")

if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
else:
    # Predict on video silently
    model.predict(
        source=video_path,
        conf=0.25,
        save=True,
        show=False,  # <-- prevents opening video window
        project=video_output,
        name="results"
    )
    print(f"Video predictions saved in: {video_output}/results")
