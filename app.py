import streamlit as st
import os
st.write("Working dir:", os.getcwd())
st.write("Files here:", os.listdir())
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="EcoBin AI", layout="centered")
st.title("♻️ EcoBin – Live Waste Detection")

st.write("🚀 App started")

# ---------------- LOAD MODEL SAFELY ----------------
@st.cache_resource(show_spinner=False)
def load_my_model():
    try:
        model = tf.keras.models.load_model("model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_my_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully")

# ---------------- PREDICTION FUNCTION ----------------
def predict(image):
    try:
        img = image.convert("RGB")  # ensure 3 channels
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)[0][0]
        return pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------------- INPUT OPTIONS ----------------
st.subheader("📷 Capture or Upload Image")

camera_image = st.camera_input("Take a photo")
upload_image = st.file_uploader("Or upload image", type=["jpg", "png", "jpeg"])

image = None

if camera_image is not None:
    image = Image.open(camera_image)

elif upload_image is not None:
    image = Image.open(upload_image)

# ---------------- PROCESS IMAGE ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("🤖 AI analyzing..."):
        pred = predict(image)

    if pred is not None:
        if pred < 0.5:
            st.success(f"Biodegradable ✅ ({round((1-pred)*100,2)}%)")
            st.info("➡️ Sent to Biogas Unit ♻️")
        else:
            st.error(f"Non-Biodegradable ❌ ({round(pred*100,2)}%)")
            st.warning("➡️ Sent to Compactor 🏭")

# ---------------- DASHBOARD ----------------
st.subheader("📊 EcoBin Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("Bin Level", "67%")
    st.metric("Waste Today", "12 kg")

with col2:
    st.metric("Biogas Generated", "3.5 L")
    st.metric("Next Collection", "3 days")

# ---------------- AUTO DUMP ----------------
bin_level = 67

if bin_level > 80:
    st.error("⚠️ Auto Dump Activated")
else:
    st.success("✅ System Normal")