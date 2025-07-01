# 🩺 Health Assistant AI Chatbot for Lung Disease Detection

An intelligent multi-modal health assistant chatbot that predicts **lung disease** using:
- 📊 Structured patient records
- 🖼️ Chest X-ray images
- 💬 Symptom-based chatbot interface (BERT)

Built using:
- **CNN (ResNet18)** for image classification
- **ML (RandomForest)** for tabular patient data
- **BERT** for understanding symptoms
- **Fusion model** combining all three
- **FastAPI** for backend API
- **Streamlit** for interactive UI

---

## 🚀 Features

- 🔍 Predict lung disease from X-ray + symptoms + patient data
- 🧠 Intelligent intent recognition using BERT
- 🗂️ Multi-modal model fusion
- 📈 Performance evaluation with Accuracy, Precision, Recall, F1 Score
- 🖥️ User-friendly Streamlit interface
- ⚡ FastAPI backend for model serving

---

## 🧾 Project Structure

health-assistant-chatbot/
├── api/ # FastAPI backend
├── data/ # Raw & processed data (ignored in .gitignore)
├── models/ # Model training scripts & saved weights
├── nlp/ # BERT intent classification
├── preprocessing/ # Data preprocessing scripts
├── ui/ # Streamlit UI
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

yaml
Copy
Edit

---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/health-assistant-chatbot.git
   cd health-assistant-chatbot
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Preprocess data:

bash
Copy
Edit
cd preprocessing
python preprocess_tabular.py
python preprocess_images.py
python generate_image_labels.py
Train models:

bash
Copy
Edit
python models/cnn_model.py
python models/tabular_model.py
python nlp/bert_model.py
python models/fusion_model.py
🧪 Evaluation
Run this to get Accuracy, Precision, Recall, F1 Score for your fusion model:

bash
Copy
Edit
python models/evaluate_fusion_model.py
🧠 Run the Application
✅ Start FastAPI server:
bash
Copy
Edit
cd api
python -m uvicorn app:app --reload
Check docs at: http://127.0.0.1:8000/docs

✅ Start Streamlit chatbot UI:
bash
Copy
Edit
cd ui
streamlit run streamlit_ui.py
📊 Sample Input for API (Swagger)
Upload X-ray file

Use features as:

json
Copy
Edit
[1,65,2,1,1,2,1,2,1,1,2,2,2,1,1]
📁 Datasets Used
🧾 ICS_Synthetic_20000.csv (synthetic patient data)

🖼️ NIH Chest X-ray Dataset (subset of PNG images)

🧠 Symptom examples for BERT (custom JSON intent list)

💡 You can scale this to real-world datasets with proper cleaning.

🧠 Models Used
Input Type	Model
Tabular	RandomForestClassifier (sklearn)
Image	ResNet18 (CNN via PyTorch)
NLP Input	BERT (via HuggingFace)
Fusion	Custom NN combining image + tabular

# (Optional) Get Metrics
python models/evaluate_fusion_model.py

📌 Author
Uttam Singh
M.Tech Cloud Computing @ IIT Patna