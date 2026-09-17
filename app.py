import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Enable lambda layer deserialization for Keras 3
tf.keras.config.enable_unsafe_deserialization()

# Dynamic path resolution to find the model wherever Streamlit is launched from
POSSIBLE_PATHS = [
    'best_intel_mobilenetv2.keras',
    'Task4_folder/best_intel_mobilenetv2.keras',
    '../best_intel_mobilenetv2.keras'
]

MODEL_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    MODEL_PATH = 'best_intel_mobilenetv2.keras'

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Cache resource to prevent reloading model on every interaction
@st.cache_resource
def load_classification_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file non-existent at path `{MODEL_PATH}` from directory `{os.getcwd()}`.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

model = load_classification_model()

# Streamlit User Interface
st.title("🌲 Intel Image Classification Web App")
st.write("Upload a landscape image to classify its environment.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess image to match MobileNetV2 target input shape (150x150)
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    if st.button('Classify Image'):
        with st.spinner('Classifying image...'):
            preds = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(preds[0])
            confidence = preds[0][pred_idx] * 100
            
            st.success(f"**Prediction:** {CLASS_NAMES[pred_idx]}")
            st.info(f"**Confidence:** {confidence:.2f}%")