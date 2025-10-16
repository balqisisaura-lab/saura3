import streamlit as st
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import tensorflow as tf

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Load YOLO (model .pt)
    yolo_model = YOLO("model/object.pt")

    # Load CNN (model .h5 atau .keras)
    keras_model = tf.keras.models.load_model("model/Balqis_isaura_Laporan2.keras")

    return yolo_model, keras_model

yolo_model, keras_model = load_models()

# ==========================
# UI Streamlit
# ==========================
st.title("🖼️ Web Deteksi Gambar CNN & YOLO")
st.write("Upload gambar untuk dideteksi menggunakan model deep learning!")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # ==========================
    # Prediksi dengan CNN (Keras)
    # ==========================
    st.subheader("🔹 Hasil Klasifikasi CNN (Keras)")

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = keras_model.predict(img_array)
    class_names = ["Kelas 0", "Kelas 1", "Kelas 2"]  # ubah sesuai label kamu
    pred_idx = np.argmax(pred)
    confidence = np.max(pred) * 100

    st.write(f"**Prediksi:** {class_names[pred_idx]} ({confidence:.2f}%)")

    # ==========================
    # Prediksi dengan YOLO (PyTorch)
    # ==========================
    st.subheader("🔹 Deteksi Objek (YOLO .pt)")
    results = yolo_model(image)
    results_img = results[0].plot()  # hasil gambar dengan bounding box
    st.image(results_img, caption="Hasil Deteksi YOLO", use_container_width=True)
