import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# Paths
DATA_PATH = "../data/patient_records.csv"
SAVE_PROCESSED = "../data/processed_tabular_data.npz"

def preprocess_tabular():
    # Load data
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Original shape:", df.shape)

    # Strip column names in case of trailing spaces
    df.columns = df.columns.str.strip()

    # Encode 'YES'/'NO' in target column
    df['LUNG_CANCER'] = df['LUNG_CANCER'].str.strip().map({'YES': 1, 'NO': 0})

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("[WARNING] Missing values found! Filling with median...")
        df.fillna(df.median(numeric_only=True), inplace=True)

    # Separate features and target
    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"].values

    # Encode categorical fields
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Processed feature shape:", X_scaled.shape)
    print("[INFO] Processed label shape:", y.shape)

    # Save processed data
    np.savez(SAVE_PROCESSED, X=X_scaled, y=y)
    print(f"[SUCCESS] Saved to {SAVE_PROCESSED}")

if __name__ == "__main__":
    preprocess_tabular()