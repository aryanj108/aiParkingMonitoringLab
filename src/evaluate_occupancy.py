import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

GT_PATH = "../data/ground_truth/slots_gt.json"
PRED_PATH = "../runs/detect/predict_frames/occupancy_predictions.json"

with open(GT_PATH) as f:
    gt = json.load(f)

with open(PRED_PATH) as f:
    preds = json.load(f)

y_true = []
y_pred = []

for slot_id in gt:
    if slot_id in preds:
        y_true.append(gt[slot_id]["occupied"])
        y_pred.append(preds[slot_id]["occupied"])

# Metrics
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nModel Performance")
print("---------------------")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# Plot confusion matrix
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks([0, 1], ["Empty", "Occupied"])
plt.yticks([0, 1], ["Empty", "Occupied"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()
