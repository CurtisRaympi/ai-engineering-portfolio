import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("🖼 AI Image Classifier")

# Load the trained model (make sure you have 'image_classifier_model.h5' saved in the folder)
@st.cache(allow_output_mutation=True)
def load_trained_model():
    return load_model("image_classifier_model.h5")

model = load_trained_model()

# Class names (adjust according to your dataset)
class_names = ['class1', 'class2', 'class3', '...']  # Replace with your actual classes

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(64, 64))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]

    st.write(f"Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
