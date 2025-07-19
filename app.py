import os
import io
import threading
import pickle
import numpy as np
from PIL import Image
import requests
import streamlit as st
from keras.models import load_model
from keras.utils import img_to_array, load_img
from flask import Flask, request, jsonify

# 📦 Configuration
MODEL_PATH = "model/greenpaalnaModel.h5"
LABEL_PATH = "model/class_labels.pkl"
IMG_SIZE = (224, 224)
FLASK_PORT = 5001

# 🧠 Load model and class labels
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)

# Support both dict and list label formats
class_labels = (
    [label for label, idx in sorted(labels.items(), key=lambda x: x[1])]
    if isinstance(labels, dict)
    else labels
)

# 🚀 Flask API
flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return "✅ Green Palna Flask API is running."

@flask_app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_stream = io.BytesIO(file.read())
        img = load_img(file_stream, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask in background
def run_flask():
    flask_app.run(host="0.0.0.0", port=FLASK_PORT)

threading.Thread(target=run_flask, daemon=True).start()

# 🌿 Streamlit UI
st.set_page_config(page_title="Green Palna Plant Classifier", layout="centered")
st.title("🌱 Green Palna Plant Classifier")
st.markdown("Upload or capture a plant image to classify it using our trained AI model.")

# 🚦 Image input mode toggle
st.subheader("📷 Choose Input Method")
input_method = st.radio("Select image source:", ["Upload from File", "Capture from Camera"])

image_source = None

if input_method == "Capture from Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image_source = camera_image
elif input_method == "Upload from File":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_source = uploaded_file

# ⏬ Display and predict
if image_source:
    st.image(image_source, caption="Selected Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            try:
                img_bytes = image_source.read()
                file_stream = io.BytesIO(img_bytes)
                content_type = getattr(image_source, "type", "image/jpeg")
                files = {"image": ("image.jpg", file_stream, content_type)}

                response = requests.post(f"http://localhost:{FLASK_PORT}/predict", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Prediction: **{result['predicted_class']}**")
                    st.info(f"📊 Confidence: `{result['confidence'] * 100:.2f}%`")
                else:
                    st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"⚠️ Could not connect to backend:\n\n{str(e)}")
