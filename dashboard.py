import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load Model CNN
# ==========================
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("model/Balqis_isaura_Laporan2.keras")  # pastikan path benar
    return model

model = load_cnn_model()

st.title("🧠 Web Klasifikasi Gambar CNN (Keras)")
st.write("✅ Model loaded successfully!")
st.write("Input shape:", model.input_shape)
st.write("Output shape:", model.output_shape)

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📤 Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    # ==========================
    # Preprocessing
    # ==========================
    st.write("🔄 Memproses gambar...")

    # Ubah ukuran sesuai input model
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Pastikan bentuk (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    st.write("Bentuk array sebelum prediksi:", img_array.shape)

    # ==========================
    # Prediksi
    # ==========================
    try:
        prediction = model.predict(img_array)
        st.write("📊 Hasil prediksi mentah:", prediction)

        # Tentukan label kelas
        class_labels = ["Kelas 1", "Kelas 2", "Kelas 3"]  # ganti sesuai label kamu
        predicted_index = int(np.argmax(prediction))
        predicted_label = class_labels[predicted_index]
        confidence = float(np.max(prediction)) * 100

        st.success(f"✅ Prediksi: **{predicted_label}** ({confidence:.2f}%)")

    except Exception as e:
        st.error(f"❌ Terjadi error saat prediksi: {e}")
