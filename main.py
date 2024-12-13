import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import os
import numpy as np
from smash_img import smash_n_reconstruct
from filters import apply_all_filters
import matplotlib.pyplot as plt # type: ignore
from io import BytesIO

from test_model import featureExtractionLayer

# Paths to different models
model_paths = {
    "Mixed Dataset": "d:/Project/model_checkpoint_all_purpose.keras",
    "Faces": "d:/Project/model_checkpoint_face.keras",
    "Artistic Images": "d:/Project/model_checkpoint_artistic_final.keras"
}

# Preprocessing function
def preprocess_image(path):
    try:
        rt, pt = smash_n_reconstruct(path)

        # Plot rich and poor textures
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(rt)
        ax[0].set_title('Rich Texture')
        ax[1].imshow(pt)
        ax[1].set_title('Poor Texture')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64_1 = buffer.getvalue()
        plt.close(fig)

        rt = apply_all_filters(rt)
        pt = apply_all_filters(pt)

        # Plot filtered textures
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(rt, cmap='gray')
        ax[0].set_title('Filtered Rich Texture')
        ax[1].imshow(pt, cmap='gray')
        ax[1].set_title('Filtered Poor Texture')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64_2 = buffer.getvalue()
        plt.close(fig)

        frt = tf.cast(tf.expand_dims(rt, axis=-1), dtype=tf.float64)
        fpt = tf.cast(tf.expand_dims(pt, axis=-1), dtype=tf.float64)
        frt = tf.ensure_shape(frt, [256, 256, 1])
        fpt = tf.ensure_shape(fpt, [256, 256, 1])
        frt = tf.expand_dims(frt, axis=0)
        fpt = tf.expand_dims(fpt, axis=0)
        
        return frt, fpt, image_base64_1, image_base64_2
    except Exception as e:
        print(f"Error processing {path}: {e}")
        dummy = tf.zeros([1, 256, 256, 1], dtype=tf.float32)
        return dummy, dummy, None, None

# Streamlit UI
st.set_page_config(page_title="Real vs Fake Image Classifier", page_icon="🎨", layout="wide", initial_sidebar_state="expanded")
st.title("Real vs Fake Image Classifier")
st.write("Select the type of dataset and upload an image to classify whether it's Real or Fake.")

# Model selection
model_choice = st.selectbox("Select Model:", list(model_paths.keys()))
model = load_model(model_paths[model_choice], custom_objects={'featureExtractionLayer': featureExtractionLayer})

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    with open("temp_image", "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open("temp_image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image and display
    test_frt, test_fpt, image_base64_1, image_base64_2 = preprocess_image("temp_image")
    if image_base64_1 and image_base64_2:
        st.image(image_base64_1, caption="Rich and Poor Textures")
        st.image(image_base64_2, caption="Filtered Rich and Poor Textures")

    # Predict
    predictions = model.predict([test_frt, test_fpt])
    result = "Fake" if predictions > 0.5 else "Real"
    st.write(f"Prediction: {result}")
