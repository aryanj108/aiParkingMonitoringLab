from ultralytics import YOLO
import os
from datetime import datetime
import shutil
import json

# Load the trained YOLO model
model = YOLO("runs/detect/parking_train3/weights/best.pt")

# Paths for input
frames_path = "../data/frames"
video_path = "../data/videos/large_parking_lot.mp4"

# Timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output folders
frames_output = f"runs/detect/predict_frames/run_{timestamp}"
video_output = f"runs/detect/predict_video/run_{timestamp}"

# --------------------- PROCESS FRAMES IN BATCHES ---------------------
if not os.path.exists(frames_path) or len(os.listdir(frames_path)) == 0:
    print(f"No images found in {frames_path}")
else:
    # List all images
    image_files = [f for f in os.listdir(frames_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()  # optional: process in order
    batch_size = 50  # adjust depending on your RAM/GPU

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_paths = [os.path.join(frames_path, f) for f in batch_files]
        print(f"Processing batch {i // batch_size + 1}: {len(batch_paths)} images")

        model.predict(
            source=batch_paths,
            conf=0.25,
            save=True,
            show=False,
            project=frames_output,
            name="results"
        )

    print(f"Frames predictions saved in: {frames_output}/results")

# Video Processing
if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
else:
    model.predict(
        source=video_path,
        conf=0.25,
        save=True,
        show=False,
        project=video_output,
        name="results"
    )

    # Move and rename the processed video
    results_dir = os.path.join(video_output, "results")
    processed_videos = [f for f in os.listdir(results_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if processed_videos:
        src_video = os.path.join(results_dir, processed_videos[0])
        final_video_folder = "runs/detect/videos_processed"
        os.makedirs(final_video_folder, exist_ok=True)
        dst_video = os.path.join(final_video_folder, f"processed_{timestamp}.mp4")
        shutil.move(src_video, dst_video)
        print(f"Processed video saved as: {dst_video}")
    else:
        print("No processed video found in the results folder.")

# Save Predictions
occupancy_predictions = {}
slot_id = 1

for batch_result in model.predict(
    source=frames_path,
    conf=0.25,
    save=False,
    show=False
):
    if batch_result.boxes is None:
        continue

    # If any car detected (occupied)
    occupied = 1 if len(batch_result.boxes) > 0 else 0

    occupancy_predictions[f"slot_{slot_id}"] = {
        "occupied": occupied
    }
    slot_id += 1

json_output_path = os.path.join(frames_output, "occupancy_predictions.json")
with open(json_output_path, "w") as f:
    json.dump(occupancy_predictions, f, indent=2)

print(f"✅ Occupancy predictions saved to: {json_output_path}")

