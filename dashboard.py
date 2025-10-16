import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Load model YOLO (deteksi objek)
    yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")

    # Load model klasifikasi (Keras .h5)
    classifier = tf.keras.models.load_model("model/Balqis_isaura_Laporan2.keras")
    return yolo_model, classifier


# Panggil kedua model
yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE YOLO
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        st.write("### 🔍 Proses Deteksi Objek...")
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    # ==========================
    # MODE KLASIFIKASI
    # ==========================
    elif menu == "Klasifikasi Gambar":
        st.write("### 🧩 Proses Klasifikasi...")

        # ---- Preprocessing ----
        img_resized = img.resize((224, 224))  # sesuaikan dengan ukuran input model kamu
        img_array = np.array(img_resized)  # pastikan bentuknya array numpy
        img_array = img_array.astype("float32") / 255.0  # normalisasi piksel
        img_array = np.expand_dims(img_array, axis=0)  # tambah dimensi batch

        # Debug info (opsional)
        st.write("Tipe data img_array:", type(img_array))
        st.write("Bentuk img_array:", img_array.shape)
        st.write("Tipe elemen:", img_array.dtype)

        # ---- Prediksi ----
        try:
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.success(f"### ✅ Hasil Prediksi: {class_index}")
            st.info(f"Probabilitas: {confidence:.4f}")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
