import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# Load trained model
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "cnn_model.keras")
    model_path = os.path.abspath(model_path)  
    return tf.keras.models.load_model(model_path)
model = load_model()

# Steamlit UI
st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9). Model trained on MNIST using CNN.")

canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Process and predict
if canvas.image_data is not None:
    img = canvas.image_data[:, :, 0]
    img = Image.fromarray(img.astype("uint8"))
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img, verbose=0)[0]
    digit = np.argmax(pred)
    st.subheader(f"Predicted Digit: **{digit}**")
    st.write("Confidence:")
    st.bar_chart(pred)