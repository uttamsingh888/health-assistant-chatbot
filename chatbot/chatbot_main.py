import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

MODEL_DIR = "../models/bert_intent"

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Same label map used during training
label_map = {
    0: "greeting",
    1: "lung_symptom",
    2: "general_health",
    3: "emergency"
}

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return label_map[predicted_class]

# CLI loop for testing
if __name__ == "__main__":
    print("🧠 Health Assistant Chatbot (Text Interface)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("🗣️  You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        intent = predict_intent(user_input)
        print(f"🤖 Detected Intent: {intent}\n")