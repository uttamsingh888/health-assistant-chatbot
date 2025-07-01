from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import torch
import io
import json
import cv2
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

# Fusion Model Imports
from torchvision import models
import torch.nn as nn

app = FastAPI()

# -----------------------------
# Load Models
# -----------------------------
# Load tokenizer & BERT model
tokenizer = BertTokenizer.from_pretrained("../models/bert_intent")
bert_model = BertForSequenceClassification.from_pretrained("../models/bert_intent")
bert_model.eval()

# Load Tabular Model
tabular_model = joblib.load("../models/tabular_model.pkl")

# Load Fusion Model
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class FusionModel(nn.Module):
    def __init__(self, tabular_input_size=15):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.tabular_fc = nn.Linear(tabular_input_size, 128)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = torch.relu(self.tabular_fc(tabular))
        combined = torch.cat((img_feat, tab_feat), dim=1)
        return self.classifier(combined)

fusion_model = FusionModel()
fusion_model.load_state_dict(torch.load("../models/fusion_model.pth", map_location=torch.device('cpu')))
fusion_model.eval()

# Label map
intent_map = {0: "greeting", 1: "lung_symptom", 2: "general_health", 3: "emergency"}

# -----------------------------
# Endpoints
# -----------------------------

class TabularInput(BaseModel):
    features: list  # 15 tabular input features

@app.post("/chat")
def chat_response(message: str = Form(...)):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        intent = intent_map[predicted_class]
    return {"intent": intent}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...), features: str = Form(...)):
    features = json.loads(features)  # convert string "[1,2,...]" ➝ list
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 224, 224)

    tab = np.array(features, dtype=np.float32).reshape(1, -1)    
    tab_tensor = torch.tensor(tab)

    with torch.no_grad():
        output = fusion_model(img, tab_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": "Lung Disease Detected" if prediction == 1 else "No Disease"}