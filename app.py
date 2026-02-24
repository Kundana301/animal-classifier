import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("animal_classifier.h5")

# Class labels (must match training)
class_names = [
    'butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'
]

st.set_page_config(page_title="Animal Classifier 🐾")
st.title("🐾 Animal Image Classifier")
st.write("Upload an image of an animal")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: **{class_names[index]}**")
    st.info(f"Confidence: {confidence:.2f}")