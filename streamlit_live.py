import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="EcoBin Live Detection", layout="centered")
st.title("♻️ EcoBin – Live Waste Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model_safe():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model_safe()
st.success("✅ Model loaded successfully")

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]  # Sigmoid output
    return pred

# ---------------- WEBCAM INPUT ----------------
st.subheader("📷 Live Webcam Feed")
camera_image = st.camera_input("Take a photo")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Input Image", use_column_width=True)

    pred = predict_image(image)

    if pred > 0.5:
        st.success(f"Biodegradable ✅ ({pred*100:.2f}%)")
        st.info("➡️ Sent to Biogas Unit ♻️")
    else:
        st.error(f"Non-Biodegradable ❌ ({(1-pred)*100:.2f}%)")
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