import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="AI Medical Diagnostics",
                   page_icon="🏥", layout="wide")

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🏥 AI Medical Diagnostics Suite")
st.markdown("---")

# Display model files found
st.sidebar.header("📁 Model Files")
for file in ['best_pneumonia_model.h5', 'best_pathmnist_model.h5', 'best_retinamnist_model.h5', 'best_brain_tumor_fixed.h5']:
    file_path = os.path.join(BASE_DIR, file)
    if os.path.exists(file_path):
        st.sidebar.success(f"✅ {file}")
    else:
        st.sidebar.error(f"❌ {file} - not found")

# Custom loader that ignores quantization_config


class CustomDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        # Remove quantization_config if present
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)


# Register custom objects
custom_objects = {
    'Dense': CustomDense,
    'Conv2D': tf.keras.layers.Conv2D,
    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
    'Dropout': tf.keras.layers.Dropout,
    'Flatten': tf.keras.layers.Flatten,
    'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
    'BatchNormalization': tf.keras.layers.BatchNormalization,
    'Sequential': tf.keras.Sequential,
    'InputLayer': tf.keras.layers.InputLayer
}


@st.cache_resource
def load_model_compatible(model_path):
    """Load model with custom objects to handle quantization_config"""
    if not os.path.exists(model_path):
        return None

    try:
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        return model
    except Exception as e:
        st.error(
            f"Error loading {os.path.basename(model_path)}: {str(e)[:100]}")
        return None


# Load all models
with st.spinner("Loading AI models..."):
    pneumonia_model = load_model_compatible(
        os.path.join(BASE_DIR, 'best_pneumonia_model.h5'))
    breast_model = load_model_compatible(
        os.path.join(BASE_DIR, 'best_pathmnist_model.h5'))
    retinopathy_model = load_model_compatible(
        os.path.join(BASE_DIR, 'best_retinamnist_model.h5'))
    brain_model = load_model_compatible(
        os.path.join(BASE_DIR, 'best_brain_tumor_fixed.h5'))

# Show loaded status
st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Status")
st.sidebar.markdown(
    f"Pneumonia: {'🟢 Active' if pneumonia_model else '🔴 Inactive'}")
st.sidebar.markdown(
    f"Breast Cancer: {'🟢 Active' if breast_model else '🔴 Inactive'}")
st.sidebar.markdown(
    f"Diabetic Retinopathy: {'🟢 Active' if retinopathy_model else '🔴 Inactive'}")
st.sidebar.markdown(
    f"Brain Tumor: {'🟡 Beta' if brain_model else '🔴 Inactive'}")

# Model selection
model_option = st.selectbox(
    "🔬 Select Diagnostic Test",
    [
        "🫁 Pneumonia Detection (Chest X-Ray)",
        "🎗️ Breast Cancer Detection (Pathology)",
        "👁️ Diabetic Retinopathy (Retinal Scan)",
        "🧠 Brain Tumor Detection (MRI) - Beta"
    ]
)

st.markdown("---")

# File upload
uploaded_file = st.file_uploader("📤 Upload Medical Image", type=[
                                 'png', 'jpg', 'jpeg', 'bmp'])


def preprocess_image_pneumonia(image):
    """Preprocess for pneumonia model (28x28 grayscale)"""
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    return img_array


def preprocess_image_breast(image):
    """Preprocess for breast cancer model (28x28 RGB)"""
    img = image.convert('RGB').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 3) / 255.0
    return img_array


def preprocess_image_retinopathy(image):
    """Preprocess for retinopathy model (28x28 RGB)"""
    img = image.convert('RGB').resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 3) / 255.0
    return img_array


def preprocess_image_brain(image):
    """Preprocess for brain tumor model (224x224 RGB)"""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img).reshape(1, 224, 224, 3) / 255.0
    return img_array


if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Analyze button
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("🧠 AI is analyzing the image..."):

            # Pneumonia Detection
            if model_option == "🫁 Pneumonia Detection (Chest X-Ray)":
                if pneumonia_model:
                    try:
                        processed = preprocess_image_pneumonia(image)
                        prediction = pneumonia_model.predict(
                            processed, verbose=0)[0][0]

                        with col2:
                            if prediction > 0.5:
                                st.error(f"### ⚠️ PNEUMONIA DETECTED")
                                st.metric("Confidence", f"{prediction:.1%}")
                                st.warning(
                                    "Recommendation: Consult a physician")
                            else:
                                st.success(f"### ✅ NORMAL CHEST X-RAY")
                                st.metric("Confidence",
                                          f"{(1-prediction):.1%}")
                                st.info("No signs of pneumonia detected")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)[:100]}")
                else:
                    st.error("Pneumonia model not available")

            # Breast Cancer Detection
            elif model_option == "🎗️ Breast Cancer Detection (Pathology)":
                if breast_model:
                    try:
                        processed = preprocess_image_breast(image)
                        prediction = breast_model.predict(
                            processed, verbose=0)[0][0]

                        with col2:
                            if prediction > 0.5:
                                st.error(f"### 🔴 MALIGNANT (Cancerous)")
                                st.metric("Confidence", f"{prediction:.1%}")
                                st.warning(
                                    "Recommendation: Consult an oncologist")
                            else:
                                st.success(f"### 🟢 BENIGN (Non-Cancerous)")
                                st.metric("Confidence",
                                          f"{(1-prediction):.1%}")
                                st.info("No malignancy detected")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)[:100]}")
                else:
                    st.error("Breast cancer model not available")

            # Diabetic Retinopathy
            elif model_option == "👁️ Diabetic Retinopathy (Retinal Scan)":
                if retinopathy_model:
                    try:
                        processed = preprocess_image_retinopathy(image)
                        prediction = retinopathy_model.predict(
                            processed, verbose=0)[0][0]

                        with col2:
                            if prediction > 0.5:
                                st.error(
                                    f"### ⚠️ DIABETIC RETINOPATHY DETECTED")
                                st.metric("Confidence", f"{prediction:.1%}")
                                st.warning(
                                    "Recommendation: Consult an ophthalmologist")
                            else:
                                st.success(f"### ✅ NO DIABETIC RETINOPATHY")
                                st.metric("Confidence",
                                          f"{(1-prediction):.1%}")
                                st.info("Regular eye exams recommended")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)[:100]}")
                else:
                    st.error("Retinopathy model not available")

            # Brain Tumor (Beta)
            elif model_option == "🧠 Brain Tumor Detection (MRI) - Beta":
                if brain_model:
                    try:
                        processed = preprocess_image_brain(image)
                        prediction = brain_model.predict(
                            processed, verbose=0)[0][0]

                        with col2:
                            st.info(f"### 🧪 BETA RESULT")
                            st.metric("Tumor Probability", f"{prediction:.1%}")
                            st.warning(
                                "⚠️ Beta model - Limited accuracy. Consult a radiologist.")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)[:100]}")
                else:
                    st.error("Brain tumor model not available")

# Footer
st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This is a demonstration AI tool. Always consult healthcare professionals for medical decisions.")
