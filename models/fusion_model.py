import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Load data
image_data = np.load("../data/processed_images.npy")  # (N, 1, 224, 224)
image_labels = np.load("../data/image_labels.npy")    # (N,)
tabular_data = np.load("../data/processed_tabular_data.npz")
tabular_features = tabular_data["X"]                  # (N2, 15)
tabular_labels = tabular_data["y"]                    # (N2,)

# Use same length subset (for simplicity)
N = min(len(image_data), len(tabular_features))
image_data = image_data[:N]
image_labels = image_labels[:N]
tabular_features = tabular_features[:N]
tabular_labels = tabular_labels[:N]

# Dataset
class FusionDataset(Dataset):
    def __init__(self, images, tabular, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.tabular[idx], self.labels[idx]

dataset = FusionDataset(image_data, tabular_features, image_labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC

    def forward(self, x):
        x = self.features(x)  # Output: (B, 512, 1, 1)
        return x.view(x.size(0), -1)  # Flatten to (B, 512)

# Fusion Model
class FusionModel(nn.Module):
    def __init__(self, tabular_input_size=15):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.tabular_fc = nn.Linear(tabular_input_size, 128)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = torch.relu(self.tabular_fc(tabular))
        combined = torch.cat((img_feat, tab_feat), dim=1)
        return self.classifier(combined)

# Training Loop
def train_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, tabular, labels in loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "../models/fusion_model.pth")
    print("[SUCCESS] Fusion model saved to ../models/fusion_model.pth")

if __name__ == "__main__":
    train_fusion()