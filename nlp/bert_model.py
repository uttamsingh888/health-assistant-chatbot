import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

# Sample data
data = [
    # GREETING (20)
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("Hey", "greeting"),
    ("Good morning", "greeting"),
    ("Good evening", "greeting"),
    ("Greetings", "greeting"),
    ("Hi there", "greeting"),
    ("Hello, bot", "greeting"),
    ("Nice to meet you", "greeting"),
    ("How are you?", "greeting"),
    ("What's your name?", "greeting"),
    ("Are you a chatbot?", "greeting"),
    ("Can you help me?", "greeting"),
    ("Is anyone there?", "greeting"),
    ("Yo!", "greeting"),
    ("Hey assistant", "greeting"),
    ("What's up?", "greeting"),
    ("Howdy!", "greeting"),
    ("Namaste", "greeting"),
    ("Hi, I need some help", "greeting"),

    # LUNG SYMPTOM (20)
    ("I have a cough and chest pain", "lung_symptom"),
    ("I'm short of breath", "lung_symptom"),
    ("My chest hurts when I breathe", "lung_symptom"),
    ("I have wheezing and feel tired", "lung_symptom"),
    ("I can't take deep breaths", "lung_symptom"),
    ("Difficulty breathing", "lung_symptom"),
    ("Sharp pain in my lungs", "lung_symptom"),
    ("My breathing is noisy", "lung_symptom"),
    ("I'm coughing all night", "lung_symptom"),
    ("My chest feels tight", "lung_symptom"),
    ("Persistent dry cough", "lung_symptom"),
    ("Wheezing sound in lungs", "lung_symptom"),
    ("Tightness in chest when lying down", "lung_symptom"),
    ("I feel heaviness in my lungs", "lung_symptom"),
    ("I struggle to breathe deeply", "lung_symptom"),
    ("My nose and chest feel blocked", "lung_symptom"),
    ("Severe dry coughing for a week", "lung_symptom"),
    ("I keep gasping for air", "lung_symptom"),
    ("Uncomfortable chest pressure", "lung_symptom"),
    ("Pain increases when I inhale", "lung_symptom"),

    # GENERAL HEALTH (20)
    ("What can this chatbot do?", "general_health"),
    ("Tell me about this app", "general_health"),
    ("What diseases can you check?", "general_health"),
    ("Explain how this bot works", "general_health"),
    ("What symptoms do you recognize?", "general_health"),
    ("Are you accurate?", "general_health"),
    ("How do you detect disease?", "general_health"),
    ("Can you scan reports?", "general_health"),
    ("Do I need to upload an image?", "general_health"),
    ("What’s your source?", "general_health"),
    ("How do you predict disease?", "general_health"),
    ("Are you using AI?", "general_health"),
    ("How do I interact with you?", "general_health"),
    ("Can I trust the results?", "general_health"),
    ("What data do you need?", "general_health"),
    ("Who developed you?", "general_health"),
    ("Can you analyze symptoms?", "general_health"),
    ("Do you use machine learning?", "general_health"),
    ("Do I need an account?", "general_health"),
    ("What kind of input do you take?", "general_health"),

    # EMERGENCY (20)
    ("I'm coughing blood!", "emergency"),
    ("I can't breathe!", "emergency"),
    ("My condition is worsening!", "emergency"),
    ("Help me right now!", "emergency"),
    ("Emergency! Chest pain!", "emergency"),
    ("This is serious. I need help!", "emergency"),
    ("My chest is collapsing!", "emergency"),
    ("I’m choking!", "emergency"),
    ("My lungs feel like shutting down", "emergency"),
    ("I feel like fainting", "emergency"),
    ("Critical chest issue", "emergency"),
    ("I can't move and my chest hurts", "emergency"),
    ("Shortness of breath worsening", "emergency"),
    ("Pain spreading to my arm", "emergency"),
    ("Ambulance needed immediately", "emergency"),
    ("I can't talk properly", "emergency"),
    ("My heart is racing too fast", "emergency"),
    ("I'm blacking out", "emergency"),
    ("Severe chest tightness", "emergency"),
    ("My lungs are giving out", "emergency"),
]

df = pd.DataFrame(data, columns=["text", "label"])
label_map = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label_id"] = df["label"].map(label_map)

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ChatDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }

# Prepare dataset
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label_id"], test_size=0.2, random_state=42)
train_dataset = ChatDataset(X_train.tolist(), y_train.tolist())
test_dataset = ChatDataset(X_test.tolist(), y_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(6):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Save model
model.save_pretrained("../models/bert_intent")
tokenizer.save_pretrained("../models/bert_intent")
print("[SUCCESS] BERT intent model saved to ../models/bert_intent")