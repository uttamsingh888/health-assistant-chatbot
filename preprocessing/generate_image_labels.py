import pandas as pd
import numpy as np
import os

IMAGE_DIR = "../data/xray_images/"
LABEL_CSV = "../data/Data_Entry_2017.csv"
OUTPUT_PATH = "../data/image_labels.npy"

# Diseases considered positive
DISEASES = [
    "Pneumonia", "Cardiomegaly", "Infiltration", "Atelectasis",
    "Effusion", "Mass", "Nodule", "Emphysema", "Fibrosis",
    "Edema", "Consolidation", "Tuberculosis"
]

def generate_labels():
    # Get available image filenames
    available_images = set(os.listdir(IMAGE_DIR))

    # Read metadata CSV
    df = pd.read_csv(LABEL_CSV)

    # Clean column names
    df.columns = df.columns.str.strip()
    df["Image Index"] = df["Image Index"].str.strip()

    # Filter for only the images you downloaded
    df = df[df["Image Index"].isin(available_images)]

    print(f"[INFO] Matched images: {len(df)} / {len(available_images)}")

    # Assign binary label: 1 if any disease, 0 if 'No Finding'
    def encode_label(label_str):
        labels = [l.strip() for l in label_str.split('|')]
        for l in labels:
            if l in DISEASES:
                return 1
        return 0

    df["Label"] = df["Finding Labels"].apply(encode_label)

    labels = df[["Image Index", "Label"]].set_index("Image Index").to_dict()["Label"]

    # Order labels in the same order as processed_images.npy
    image_filenames = sorted(list(available_images))
    y = np.array([labels.get(fname, 0) for fname in image_filenames])

    print("[INFO] Positive cases:", np.sum(y), "/", len(y))
    np.save(OUTPUT_PATH, y)
    print(f"[SUCCESS] Saved binary labels to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_labels()