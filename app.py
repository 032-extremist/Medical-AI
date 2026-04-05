# ===================================================================
# COMPLETE AI MEDICAL DIAGNOSTICS WEB APP
# Features: Pneumonia, Breast Cancer, Diabetic Retinopathy, Brain Tumor
# ===================================================================

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st
import os
import tensorflow as tf

print("="*50)
print("DIAGNOSTIC INFORMATION")
print("="*50)
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
print("="*50)

# Try to load each model and print detailed errors
print("\nAttempting to load models...")

models_to_test = [
    ('best_pneumonia_model.h5', 'Pneumonia'),
    ('best_pathmnist_model.h5', 'Breast Cancer'),
    ('best_retinamnist_model.h5', 'Retinopathy'),
    ('best_brain_tumor_fixed.h5', 'Brain Tumor')
]

for model_file, model_name in models_to_test:
    if os.path.exists(model_file):
        print(f"✅ {model_name} file found: {model_file}")
        try:
            test_model = tf.keras.models.load_model(model_file)
            print(f"   ✅ {model_name} loaded successfully!")
        except Exception as e:
            print(f"   ❌ {model_name} failed to load: {e}")
    else:
        print(f"❌ {model_name} file NOT found: {model_file}")
print("="*50)


# Set page config
st.set_page_config(
    page_title="AI Medical Diagnostics Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .main-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    .model-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .model-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .result-box {
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        transition: all 0.3s ease;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        box-shadow: 0 2px 8px rgba(244,67,54,0.2);
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 8px rgba(76,175,80,0.2);
    }
    .beta-badge {
        background-color: #ff9800;
        color: white;
        font-size: 0.7em;
        padding: 3px 8px;
        border-radius: 20px;
        margin-left: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 15px;
        font-size: 0.8em;
        color: #666;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS (Cached for performance)
# -------------------------------


@st.cache_resource
def load_pneumonia_model():
    try:
        model = tf.keras.models.load_model('best_pneumonia_model.h5')
        return model
    except:
        return None


@st.cache_resource
def load_breast_cancer_model():
    try:
        model = tf.keras.models.load_model('best_pathmnist_model.h5')
        return model
    except:
        return None


@st.cache_resource
def load_retinopathy_model():
    try:
        model = tf.keras.models.load_model('best_retinamnist_model.h5')
        return model
    except:
        return None


@st.cache_resource
def load_brain_tumor_model():
    try:
        model = tf.keras.models.load_model('best_brain_tumor_fixed.h5')
        return model
    except:
        return None

# -------------------------------
# PREDICTION FUNCTIONS
# -------------------------------


def preprocess_image_pneumonia(img):
    """Preprocess for pneumonia model (28x28 grayscale)"""
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def preprocess_image_breast(img):
    """Preprocess for breast cancer model (28x28 RGB)"""
    img = img.convert('RGB')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 3)
    return img_array


def preprocess_image_retinopathy(img):
    """Preprocess for retinopathy model (28x28 RGB)"""
    img = img.convert('RGB')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 3)
    return img_array


def preprocess_image_brain(img):
    """Preprocess for brain tumor model (224x224 RGB)"""
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array


# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Medical Diagnostics Suite</h1>
    <p>Deep Learning Models for Multi-Disease Detection from Medical Images</p>
    <p style="font-size: 0.9em;">⚡ Real-time AI Analysis | 📊 Evidence-based Predictions | 🔬 Clinically Inspired</p>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("🔄 Loading AI Models... Please wait..."):
    pneumonia_model = load_pneumonia_model()
    breast_model = load_breast_cancer_model()
    retinopathy_model = load_retinopathy_model()
    brain_model = load_brain_tumor_model()

# Display model status in sidebar
st.sidebar.markdown("## 📊 Model Status")
st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)
with col1:
    if pneumonia_model:
        st.markdown("✅ **Pneumonia**")
    else:
        st.markdown("❌ **Pneumonia**")

    if breast_model:
        st.markdown("✅ **Breast Cancer**")
    else:
        st.markdown("❌ **Breast Cancer**")
with col2:
    if retinopathy_model:
        st.markdown("✅ **Retinopathy**")
    else:
        st.markdown("❌ **Retinopathy**")

    if brain_model:
        st.markdown("⚠️ **Brain Tumor** (Beta)")
    else:
        st.markdown("❌ **Brain Tumor**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.markdown("""
| Model | AUC |
|-------|-----|
| 🫁 Pneumonia | **0.968** |
| 🎗️ Breast Cancer | **0.961** |
| 👁️ Retinopathy | **0.787** |
| 🧠 Brain Tumor | 0.532* |
""")
st.sidebar.markdown("*Beta - Under development")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("""
1. Select a diagnostic test
2. Upload a medical image
3. Click 'Analyze'
4. View AI prediction
""")

# -------------------------------
# MAIN CONTENT - MODEL SELECTION
# -------------------------------
st.markdown("## 🔬 Select Diagnostic Test")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🫁 **Pneumonia**\nChest X-Ray", use_container_width=True):
        st.session_state.selected_model = "pneumonia"
with col2:
    if st.button("🎗️ **Breast Cancer**\nPathology Image", use_container_width=True):
        st.session_state.selected_model = "breast"
with col3:
    if st.button("👁️ **Retinopathy**\nRetinal Scan", use_container_width=True):
        st.session_state.selected_model = "retinopathy"
with col4:
    if st.button("🧠 **Brain Tumor**\nMRI Scan (Beta)", use_container_width=True):
        st.session_state.selected_model = "brain"

# Default selection
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "pneumonia"

st.markdown("---")

# -------------------------------
# PNEUMONIA DETECTION
# -------------------------------
if st.session_state.selected_model == "pneumonia":
    st.markdown("""
    <div class="model-card">
        <div class="model-title">
            🫁 Pneumonia Detection from Chest X-Ray
            <span style="font-size:0.8em; background:#4caf50; color:white; padding:2px 10px; border-radius:20px;">PRODUCTION</span>
        </div>
        <p>Upload a chest X-ray image for pneumonia detection. The model analyzes the image and provides a probability score.</p>
        <p><strong>📊 Performance:</strong> AUC = 0.968 | Sensitivity = 96.9% | Specificity = 77.4%</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray Image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="pneumonia_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded X-Ray Image",
                     use_container_width=True)

        if st.button("🔍 Analyze Image", key="pneumonia_analyze"):
            if pneumonia_model:
                with st.spinner("🧠 Analyzing image with AI..."):
                    processed = preprocess_image_pneumonia(image)
                    prediction = pneumonia_model.predict(
                        processed, verbose=0)[0][0]

                with col2:
                    if prediction > 0.5:
                        st.markdown(f"""
                        <div class="result-box positive">
                            <h3>⚠️ PNEUMONIA DETECTED</h3>
                            <p><strong>Confidence:</strong> {prediction:.2%}</p>
                            <p><strong>Recommendation:</strong> Immediate medical consultation recommended.</p>
                            <hr>
                            <p style="font-size:0.9em;">Model confidence: {prediction:.1%} | Threshold: 50%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box negative">
                            <h3>✅ NORMAL CHEST X-RAY</h3>
                            <p><strong>Confidence:</strong> {(1-prediction):.2%}</p>
                            <p>No signs of pneumonia detected.</p>
                            <hr>
                            <p style="font-size:0.9em;">Model confidence: {(1-prediction):.1%} | Threshold: 50%</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(
                    "❌ Model not loaded. Please ensure best_pneumonia_model.h5 exists.")

# -------------------------------
# BREAST CANCER DETECTION
# -------------------------------
elif st.session_state.selected_model == "breast":
    st.markdown("""
    <div class="model-card">
        <div class="model-title">
            🎗️ Breast Cancer Detection from Pathology Image
            <span style="font-size:0.8em; background:#4caf50; color:white; padding:2px 10px; border-radius:20px;">PRODUCTION</span>
        </div>
        <p>Upload a breast pathology image to classify as Benign (non-cancerous) or Malignant (cancerous).</p>
        <p><strong>📊 Performance:</strong> AUC = 0.961 | Sensitivity = 77.1% | Specificity = 97.6%</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Pathology Image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="breast_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Pathology Image",
                     use_container_width=True)

        if st.button("🔍 Analyze Image", key="breast_analyze"):
            if breast_model:
                with st.spinner("🧠 Analyzing image with AI..."):
                    processed = preprocess_image_breast(image)
                    prediction = breast_model.predict(
                        processed, verbose=0)[0][0]

                with col2:
                    if prediction > 0.5:
                        st.markdown(f"""
                        <div class="result-box positive">
                            <h3>🔴 MALIGNANT (Cancerous)</h3>
                            <p><strong>Confidence:</strong> {prediction:.2%}</p>
                            <p><strong>Recommendation:</strong> Immediate consultation with an oncologist recommended.</p>
                            <hr>
                            <p style="font-size:0.9em;">Note: This is an AI-assisted screening tool. Confirm with biopsy.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box negative">
                            <h3>🟢 BENIGN (Non-Cancerous)</h3>
                            <p><strong>Confidence:</strong> {(1-prediction):.2%}</p>
                            <p>No malignancy detected. Regular screening still recommended.</p>
                            <hr>
                            <p style="font-size:0.9em;">Note: Continue regular check-ups as recommended.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(
                    "❌ Model not loaded. Please ensure best_pathmnist_model.h5 exists.")

# -------------------------------
# DIABETIC RETINOPATHY DETECTION
# -------------------------------
elif st.session_state.selected_model == "retinopathy":
    st.markdown("""
    <div class="model-card">
        <div class="model-title">
            👁️ Diabetic Retinopathy Detection from Retinal Scan
            <span style="font-size:0.8em; background:#4caf50; color:white; padding:2px 10px; border-radius:20px;">PRODUCTION</span>
        </div>
        <p>Upload a retinal fundus image to detect signs of Diabetic Retinopathy.</p>
        <p><strong>📊 Performance:</strong> AUC = 0.787 | Sensitivity = 82.7% | Specificity = 69.0%</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Retinal Scan Image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="retinopathy_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Retinal Scan",
                     use_container_width=True)

        if st.button("🔍 Analyze Image", key="retinopathy_analyze"):
            if retinopathy_model:
                with st.spinner("🧠 Analyzing image with AI..."):
                    processed = preprocess_image_retinopathy(image)
                    prediction = retinopathy_model.predict(
                        processed, verbose=0)[0][0]

                with col2:
                    if prediction > 0.5:
                        st.markdown(f"""
                        <div class="result-box positive">
                            <h3>⚠️ DIABETIC RETINOPATHY DETECTED</h3>
                            <p><strong>Confidence:</strong> {prediction:.2%}</p>
                            <p><strong>Recommendation:</strong> Consult an ophthalmologist for comprehensive eye examination.</p>
                            <hr>
                            <p style="font-size:0.9em;">Early detection can prevent vision loss.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box negative">
                            <h3>✅ NO DIABETIC RETINOPATHY</h3>
                            <p><strong>Confidence:</strong> {(1-prediction):.2%}</p>
                            <p>No signs of diabetic retinopathy detected.</p>
                            <hr>
                            <p style="font-size:0.9em;">Regular eye exams still recommended for diabetic patients.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(
                    "❌ Model not loaded. Please ensure best_retinamnist_model.h5 exists.")

# -------------------------------
# BRAIN TUMOR DETECTION (BETA)
# -------------------------------
elif st.session_state.selected_model == "brain":
    st.markdown("""
    <div class="model-card">
        <div class="model-title">
            🧠 Brain Tumor Detection from MRI Scan
            <span class="beta-badge">BETA</span>
        </div>
        <p>Upload a brain MRI image for tumor detection. <strong>⚠️ Beta version - Under development.</strong></p>
        <p><strong>📊 Performance:</strong> AUC = 0.532 | Under improvement</p>
        <p><em>Note: This model is currently in beta testing. Results should be interpreted with caution.</em></p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Brain MRI Image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="brain_upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded MRI Scan",
                     use_container_width=True)

        if st.button("🔍 Analyze Image (Beta)", key="brain_analyze"):
            if brain_model:
                with st.spinner("🧠 Analyzing image with AI (Beta)..."):
                    processed = preprocess_image_brain(image)
                    prediction = brain_model.predict(
                        processed, verbose=0)[0][0]

                with col2:
                    st.markdown(f"""
                    <div class="result-box" style="background-color:#fff3e0; border-left-color:#ff9800;">
                        <h3>🧪 BETA RESULT</h3>
                        <p><strong>Tumor Probability:</strong> {prediction:.2%}</p>
                        <p><strong>⚠️ Beta Disclaimer:</strong> This model is experimental with limited accuracy.</p>
                        <hr>
                        <p style="font-size:0.9em;">Please consult a radiologist for definitive diagnosis.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add a note about model limitations
                    st.info("ℹ️ **Beta Model Note:** This model is currently under development. Performance is limited due to small training dataset. Do not use for clinical decisions.")
            else:
                st.error(
                    "❌ Model not loaded. Please ensure best_brain_tumor_fixed.h5 exists.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
    <p>⚠️ <strong>Medical Disclaimer:</strong> This is a demonstration AI tool for educational purposes. 
    All predictions should be verified by qualified healthcare professionals. 
    Do not make medical decisions based solely on AI outputs.</p>
    <p>📊 Models trained on MedMNIST and public datasets | © 2025 AI Medical Diagnostics Suite</p>
    <p>🏥 For research and demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)
