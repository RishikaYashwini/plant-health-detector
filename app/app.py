import streamlit as st
import sys
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2
import seaborn as sns

sys.path.append(os.path.abspath("src"))
from predict import predict_image

st.set_page_config(page_title="AI Plant Doctor", layout="wide")

st.title("🌿 AI Plant Doctor — Intelligent Diagnosis")

# ---------- Sidebar Smart Controls ----------

st.sidebar.title("🧠 AI Control Panel")

# Model Status
st.sidebar.header("📊 Model Status")

try:
    with open("model/metrics.json") as f:
        metrics = json.load(f)
        st.sidebar.metric("AI Reliability Score", f"{metrics.get('val_acc',0)*100:.2f}%")
except:
    st.sidebar.info("Metrics not available")

# User Mode
st.sidebar.header("🌾 User Mode")
farmer_mode = st.sidebar.checkbox("Farmer Friendly Language", value=True)

# Explainability Controls
st.sidebar.header("🔍 Explain AI Decisions")
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)
opacity = st.sidebar.slider("Heatmap Opacity", 0.1, 1.0, 0.4)

# Evaluation Panel
st.sidebar.header("📈 Evaluation")
show_confusion = st.sidebar.checkbox("Show Confusion Matrix", value=True)

# Load model
model = tf.keras.models.load_model("model/plant_model.h5", compile=False)

with open("model/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v:k for k,v in class_indices.items()}

# ---------- GradCAM ----------
def gradcam(img_path):

    img = image.load_img(img_path, target_size=(160,160))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap

# ---------- Treatment DB ----------
treatment_db = {
    "blight": ("Remove infected leaves and apply fungicide.","Keep leaves dry and ensure airflow."),
    "rust": ("Apply fungicide.","Monitor regularly and avoid wet leaves."),
    "spot": ("Use copper spray.","Clean tools and remove debris."),
    "mildew": ("Apply neem oil or sulfur spray.","Reduce humidity."),
    "virus": ("Remove infected plants.","Control insects and pests.")
}

uploaded = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded:

    with open("temp.jpg","wb") as f:
        f.write(uploaded.read())

    col1,col2 = st.columns([1,1])

    with col1:
        st.image("temp.jpg", use_container_width=True)

    with col2:
        label, confidence = predict_image("temp.jpg")

        st.subheader("Diagnosis")
        st.success(label)
        st.write(f"Confidence: {confidence*100:.2f}%")

        if "healthy" in label.lower():
            if farmer_mode:
                st.success("🌱 Your plant looks healthy — no disease found.")
            else:
                st.success("Plant appears healthy.")
        else:
            st.error("Disease detected")

            for key in treatment_db:
                if key in label.lower():
                    treat, prevent = treatment_db[key]

                    st.subheader("Suggested Treatment")
                    st.write(treat)

                    st.subheader("Preventive Measures")
                    st.write(prevent)

                    if farmer_mode:
                        st.info("👉 Treat early to avoid crop loss.")

# ---------- Interactive Heatmap ----------
st.markdown("---")
st.subheader("🔥 Disease Heatmap")

if uploaded and show_heatmap:

    heatmap = gradcam("temp.jpg")

    original = cv2.imread("temp.jpg")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)

    overlay = heatmap_colored * opacity + original

    st.image(overlay.astype(np.uint8), caption="AI Focus Area")

# ---------- Confusion Matrix ----------
st.markdown("---")
st.header("🧾 How Well The AI Is Performing")

st.write("""
🟢 Green areas = AI predicts correctly  
🔴 Dark spots away from green = AI sometimes gets confused  
The brighter the square, the more samples.
""")

if show_confusion:
    try:
        cm = np.load("model/confusion_matrix.npy")

        class_names = list(class_indices.keys())

        # Calculate overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)

        st.subheader(f"✅ Overall Accuracy: {accuracy*100:.2f}%")

        fig, ax = plt.subplots(figsize=(10,8))

        sns.heatmap(
            cm,
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("AI Prediction")
        ax.set_ylabel("Actual Disease")

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        st.pyplot(fig)

        st.info("""
    📘 What this means:

    If most color is along the diagonal line, the AI is performing well.
    Small off-diagonal spots show rare mistakes.
    """)

        # Find most confused pairs
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)

        idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)

        if off_diag[idx] > 0:
            confused_pair = (
                class_names[idx[0]],
                class_names[idx[1]]
            )

            st.warning(f"""
    ⚠️ The AI sometimes confuses:

    • {confused_pair[0]}  
    with  
    • {confused_pair[1]}

    This happens because symptoms look similar.
    """)

    except:
        st.error("Confusion matrix not available — run evaluation script.")