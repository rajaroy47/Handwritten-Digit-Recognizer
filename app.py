
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn.h5')

model = load_model()

st.title("🖐️ Handwritten Digit Recognizer")
st.write("Upload a 28x28 grayscale image of a handwritten digit (white digit on black background)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Show uploaded image
    st.image(image, caption='Uploaded Image', width=150)

    # Preprocess image: invert colors (white digit on black bg), resize to 28x28
    image = ImageOps.invert(image)
    image = image.resize((28,28))

    # Convert to numpy array and normalize
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: **{digit}**")
