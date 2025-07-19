# 🌿GreenPlaanaAIModel - Plant Classifier with  Voice (pyttsx3)

> A smart plant classifier built with Flask + Streamlit. Upload or capture a plant image, get the prediction using a deep learning model, and hear the result spoken in **English** using `pyttsx3`.

---

## 📸 Features

- ✅ Upload or capture image from camera
- ✅ Predict plant species (Mango, Guava, Moringo, Aawla, Papaya using MobileNetV2
- ✅ Speak prediction result using `pyttsx3`
- ✅ Streamlit-powered web frontend
- ✅ Flask-powered REST API backend
- ✅ Lightweight and easy to deploy locally

---

## 🧠 Model Info

- **Architecture:** `MobileNetV2`
- **Framework:** TensorFlow / Keras
- **Input size:** `(224, 224)`
- **Output:** Plant class label with confidence

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/GreenPlaanaAIModel.git
cd GreenPlaanaAIModel

# Optional: setup virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

```


## 📁 Project Structure

```bash

GreenPlaanaAIModel/
│
├── model/
│ ├── plant_classifier.h5
│ └── class_labels.pkl
├── app.py
├── requirements.txt
└── README.md

```

## ▶️ Run the App

```bash

streamlit run app.py

```

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Made for learning Image processing and Image classification

[**Deo Prakash**](https://www.linkedin.com/in/deo-prakash-152265225/)
