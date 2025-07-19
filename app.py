import io
import pickle
import numpy as np
from PIL import Image
import streamlit as st
from keras.models import load_model
from keras.utils import img_to_array, load_img

# 📦 Configuration
MODEL_PATH = "model/greenpaalnaModel.h5"
LABEL_PATH = "model/class_labels.pkl"
IMG_SIZE = (224, 224)

# 🧠 Load model and class labels
model = load_model(MODEL_PATH)

with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)

# Handle both dict or list label formats
class_labels = (
    [label for label, idx in sorted(labels.items(), key=lambda x: x[1])]
    if isinstance(labels, dict)
    else labels
)

# 🌿 Streamlit UI
st.set_page_config(page_title="Green Paalna Plant Classifier", layout="centered")
st.title("🌱 Green Paalna Plant Classifier")
st.markdown("Upload or capture a plant image to classify it using our trained AI model.")

# 🚦 Image input mode toggle
st.subheader("📷 Choose Input Method")
input_method = st.radio("Select image source:", ["Upload from File", "Capture from Camera"])

image_source = None
if input_method == "Capture from Camera":
    image_source = st.camera_input("Take a picture")
elif input_method == "Upload from File":
    image_source = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ⏬ Predict
if image_source:
    st.image(image_source, caption="Selected Image", use_container_width=True)

    if st.button("🔍 Predict"):
        try:
            img_bytes = image_source.read()
            img = load_img(io.BytesIO(img_bytes), target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_index]
            confidence = float(np.max(predictions))

            st.success(f"✅ Prediction: **{predicted_class}**")
            st.info(f"📊 Confidence: `{confidence * 100:.2f}%`")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
