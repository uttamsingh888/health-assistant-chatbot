import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = "../data/processed_tabular_data.npz"
MODEL_SAVE_PATH = "../models/tabular_model.pkl"

def train_tabular_model():
    # Load data
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[INFO] Accuracy:  {acc:.4f}")
    print(f"[INFO] Precision: {prec:.4f}")
    print(f"[INFO] Recall:    {rec:.4f}")
    print(f"[INFO] F1 Score:  {f1:.4f}")

    # Save model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"[SUCCESS] Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_tabular_model()