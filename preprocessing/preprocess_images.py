import os
import cv2
import numpy as np
from tqdm import tqdm

# Parameters
IMAGE_DIR = "../data/xray_images/"
OUTPUT_PATH = "../data/processed_images.npy"
IMAGE_SIZE = (224, 224)

def load_and_preprocess_images():
    images = []
    filenames = []

    print("[INFO] Reading images from:", IMAGE_DIR)
    for file in tqdm(os.listdir(IMAGE_DIR)):
        if file.endswith(".png"):
            img_path = os.path.join(IMAGE_DIR, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 1-channel
            image = cv2.resize(image, IMAGE_SIZE)
            image = image.astype(np.float32) / 255.0             # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)               # Add channel dim: (1, 224, 224)
            images.append(image)
            filenames.append(file)

    images = np.stack(images)
    print(f"[INFO] Total images processed: {images.shape[0]}")

    np.save(OUTPUT_PATH, images)
    print(f"[SUCCESS] Processed images saved to {OUTPUT_PATH}")

    return filenames

if __name__ == "__main__":
    load_and_preprocess_images()