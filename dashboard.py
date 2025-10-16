import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load Model CNN
# ==========================
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("model/Balqis_isaura_Laporan2.keras")
    return model

model = load_cnn_model()

# ==========================
# Label kelas (ubah sesuai model kamu)
# ==========================
CLASS_NAMES = ['Kelas1', 'Kelas2', 'Kelas3']  # ganti sesuai kelas aslimu

# ==========================
# Fungsi Prediksi
# ==========================
def predict_image(image):
    img = image.resize((224, 224))  # sesuaikan dengan ukuran input model kamu
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return CLASS_NAMES[class_idx], confidence

# ==========================
# Streamlit UI
# ==========================
st.title("🧠 Image Classification Web App")
st.write("Upload gambar untuk diklasifikasikan menggunakan model CNN kamu!")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Prediksi Gambar"):
        with st.spinner("Sedang memproses..."):
            label, confidence = predict_image(image)
        st.success(f"Hasil Prediksi: **{label}** ({confidence*100:.2f}% yakin)")
