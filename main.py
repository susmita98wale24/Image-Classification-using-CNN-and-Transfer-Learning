import streamlit as st
from PIL import Image
import numpy as np
from src.predict import predict_rice_type

st.title("Rice Grain Classifier")
file = st.file_uploader("Upload Rice Image", type=["jpg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict_rice_type(image)
    st.write(f"Prediction: **{prediction}**")