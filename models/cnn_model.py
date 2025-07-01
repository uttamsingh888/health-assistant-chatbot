import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

# Load data
images = np.load("../data/processed_images.npy")  # shape: (N, 1, 224, 224)
labels = np.load("../data/image_labels.npy")      # shape: (N,)

# Convert to torch format
class LungXRayDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
train_dataset = LungXRayDataset(X_train, y_train)
test_dataset = LungXRayDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define model
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1-channel
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)

    def forward(self, x):
        return self.base_model(x)

# Training function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):  # You can increase to 10–20 later
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "../models/cnn_classifier.pth")
    print("[SUCCESS] Model saved to ../models/cnn_classifier.pth")

if __name__ == "__main__":
    train_model()