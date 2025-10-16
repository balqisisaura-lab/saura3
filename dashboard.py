import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Model CNN
# ==========================
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("model/Balqis_isaura_Laporan2.keras")  # ganti sesuai nama file kamu
    return model

model = load_cnn_model()

# Tampilkan info model
st.title("🧠 Web Klasifikasi Gambar dengan CNN (Keras)")
st.write("Model input shape:", model.input_shape)
st.write("Model output shape:", model.output_shape)

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📤 Upload Gambar untuk Diklasifikasi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # ==========================
    # Preprocessing Gambar
    # ==========================
    st.write("🔄 Memproses gambar...")

    img_resized = img.resize((224, 224))  # ganti sesuai input model kamu
    img_array = np.array(img_resized) / 255.0  # normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # jadi (1, 224, 224, 3)

    st.write(f"✅ Bentuk array sebelum prediksi: {img_array.shape}")

    # ==========================
    # Prediksi
    # ==========================
    st.write("🔍 Melakukan prediksi...")
    prediction = model.predict(img_array)

    st.write("📊 Hasil prediksi mentah:")
    st.write(prediction)

    # ==========================
    # Interpretasi Hasil
    # ==========================
    # Ganti label sesuai jumlah kelas kamu
    class_labels = ["Kelas 1", "Kelas 2", "Kelas 3"]  

    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = np.max(prediction) * 100

    st.success(f"✅ Prediksi: **{predicted_label}** ({confidence:.2f}%)")
