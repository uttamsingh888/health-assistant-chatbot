import streamlit as st
import requests
import json
import io

st.set_page_config(page_title="Health Assistant AI", layout="centered")

st.title("🩺 Lung Health Assistant Chatbot")

# ---------------------
# 1. Chat Interface
# ---------------------
st.subheader("🧠 Chat Interface")
user_input = st.text_input("Type your symptoms here (e.g. I have chest pain)")

if st.button("Detect Intent"):
    res = requests.post("http://127.0.0.1:8000/chat", data={"message": user_input})
    intent = res.json()["intent"]
    st.success(f"🔍 Detected Intent: **{intent}**")

# ---------------------
# 2. Prediction Panel
# ---------------------
st.subheader("📷 Predict Lung Disease")

# Upload Image
image_file = st.file_uploader("Upload Chest X-ray Image (.png)", type=["png"])

# Tabular Input
st.markdown("#### Fill Patient Details")

features = []
fields = [
    "GENDER (0=F, 1=M)", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

default_vals = [1, 55, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1]

cols = st.columns(3)
for i, label in enumerate(fields):
    val = cols[i % 3].number_input(label, value=int(default_vals[i]))
    features.append(val)

if st.button("🔬 Predict Disease"):
    if image_file is None:
        st.warning("Please upload an X-ray image.")
    else:
        files = {"file": (image_file.name, image_file, "image/png")}
        data = {"features": str(features)}  # Send raw string: "[1,65,...]"
        res = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
        result = res.json()["prediction"]
        st.success(f"🩻 Prediction: **{result}**")