import torch
import numpy as np
from fusion_model import FusionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load tabular data
tabular = np.load("../data/processed_tabular_data.npz")
X = tabular["X"]
y = tabular["y"]

# Train-test split
X_train_tab, X_test_tab, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load image data (adjust this to match how you sliced earlier)
X_img = np.load("../data/processed_images.npy")
X_train_img, X_test_img = train_test_split(X_img, test_size=0.2, random_state=42)

# Convert to torch tensors
X_test_tab = torch.tensor(X_test_tab, dtype=torch.float32)
X_test_img = torch.tensor(X_test_img, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Load trained model
model = FusionModel()
model.load_state_dict(torch.load("../models/fusion_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred = model(X_test_img, X_test_tab)
    y_pred = (y_pred > 0.5).int().squeeze().numpy()
    y_true = y_test.numpy()

# Metrics
print("[INFO] Accuracy:  ", accuracy_score(y_true, y_pred))
print("[INFO] Precision: ", precision_score(y_true, y_pred))
print("[INFO] Recall:    ", recall_score(y_true, y_pred))
print("[INFO] F1 Score:  ", f1_score(y_true, y_pred))