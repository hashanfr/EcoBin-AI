import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="EcoBin", layout="centered")
st.title("♻️ EcoBin – Live Waste Detection")

# ---------------- LOAD MODEL ----------------
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
        img = image.convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0  # match training normalization
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)[0][0]  # sigmoid output
        return pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------------- INPUT ----------------
st.subheader("📷 Capture or Upload Image")
camera_image = st.camera_input("Take a photo")
upload_image = st.file_uploader("Or upload image", type=["jpg","png","jpeg"])

image = None
if camera_image is not None:
    image = Image.open(camera_image)
elif upload_image is not None:
    image = Image.open(upload_image)

# ---------------- THRESHOLD ADJUST ----------------
st.subheader("⚖️ Adjust Threshold (0-1)")
threshold = st.slider("Biodegradable Threshold", 0.0, 1.0, 0.5, 0.01)

# ---------------- PREDICTION & DISPLAY ----------------
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("🤖 Analyzing..."):
        pred = predict(image)

    if pred is not None:
        st.write(f"Raw model probability: {pred:.4f}")

        if pred <= threshold:
            st.success(f"Biodegradable ✅ ({(1-pred)*100:.2f}%)")
        else:
            st.error(f"Non-Biodegradable ❌ ({pred*100:.2f}%)")