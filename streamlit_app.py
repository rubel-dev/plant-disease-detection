import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/plant_disease_model.keras"
CLASS_PATH = "models/class_names.txt"

st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# ----------------------------
# UI CSS
# ----------------------------
st.markdown("""
<style>
    .main-title { font-size: 38px; font-weight: 800; margin-bottom: 0px; }
    .sub-title { font-size: 16px; color: #6b7280; margin-top: 0px; }
    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.03);
        text-align: center;
    }
    .metric-label { font-size: 14px; color: #6b7280; margin-bottom: 6px; }
    .metric-value { font-size: 22px; font-weight: 800; color: #111827; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(img, size=(224, 224)):
    img_resized = img.resize(size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def plot_confidence_gauge(conf):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")

    ax.barh(0, 1, height=0.35)
    ax.barh(0, conf, height=0.35)

    ax.text(0, 0.45, "0%", fontsize=12)
    ax.text(0.95, 0.45, "100%", fontsize=12)
    ax.text(conf, -0.2, f"{conf*100:.1f}%", fontsize=14, fontweight="bold")

    ax.set_xlim(0, 1)
    return fig


@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        st.error(f" Model file not found: `{MODEL_PATH}`")
        st.stop()

    if not os.path.exists(CLASS_PATH):
        st.error(f" Class names file not found: `{CLASS_PATH}`")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names


#  Treatment Suggestions
TREATMENT_GUIDE = {
    "healthy": " Leaf looks healthy. Keep regular watering and avoid over-fertilizing.",
    "blight": " Remove infected leaves, avoid overhead watering, and apply fungicide if needed.",
    "rust": " Remove infected leaves and apply sulfur-based fungicide. Improve airflow.",
    "mold": " Reduce humidity, improve airflow, apply antifungal spray.",
    "spot": " Remove infected leaves, avoid wet leaves, use copper-based fungicide.",
    "scab": " Remove infected leaves, improve soil drainage, apply fungicide.",
}

def get_treatment(pred_class):
    pred_lower = pred_class.lower()
    for key in TREATMENT_GUIDE.keys():
        if key in pred_lower:
            return TREATMENT_GUIDE[key]
    return " No treatment suggestion available. Consult local agricultural experts."


# ----------------------------
# Load Model
# ----------------------------
model, class_names = load_model_and_classes()

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="main-title"> Plant Disease Detection Web App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload leaf images → detect disease → confidence + treatment suggestions</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header(" Settings")
show_prob_table = st.sidebar.checkbox("Show Probability Table", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader(" Model Info")
st.sidebar.write(f" Total Classes: **{len(class_names)}**")
st.sidebar.write(" Model: **MobileNetV2 Transfer Learning**")

# ----------------------------
# Upload Multiple Images
# ----------------------------
uploaded_files = st.file_uploader(
    " Upload Leaf Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    all_reports = []

    for uploaded_file in uploaded_files:
        st.markdown(f"##  Image Result — {uploaded_file.name}")

        img = Image.open(uploaded_file).convert("RGB")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Predict
        img_array = preprocess_image(img, size=(224, 224))
        predictions = model.predict(img_array, verbose=0)[0]

        top_3_idx = predictions.argsort()[-3:][::-1]
        final_class = class_names[top_3_idx[0]]
        final_conf = float(predictions[top_3_idx[0]])
        treatment = get_treatment(final_class)

        with col_right:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction</div>
                    <div class="metric-value">{final_class}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{final_conf:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.success(f" Predicted Disease: **{final_class}**")
            st.info(f" Treatment Suggestion: {treatment}")

            st.subheader(" Confidence Gauge")
            st.pyplot(plot_confidence_gauge(final_conf))

        # Top 3 Predictions
        st.subheader(" Top 3 Predictions")
        results = [{"Class": class_names[i], "Confidence (%)": round(float(predictions[i]) * 100, 2)} for i in top_3_idx]
        st.dataframe(pd.DataFrame(results), width="stretch")

        # Full Probability Table
        if show_prob_table:
            st.subheader(" Full Probability Breakdown (Top 10)")
            full_df = pd.DataFrame({"Class": class_names, "Probability": predictions}).sort_values("Probability", ascending=False)
            st.dataframe(full_df.head(10), width="stretch")

        # Save report row
        all_reports.append({
            "File": uploaded_file.name,
            "Prediction": final_class,
            "Confidence": round(final_conf, 4),
            "Top_3": ", ".join([class_names[i] for i in top_3_idx]),
            "Treatment": treatment
        })

        st.markdown("-----")

    # Download All Reports CSV
    st.subheader(" Download All Predictions Report")
    report_df = pd.DataFrame(all_reports)
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=" Download Full CSV Report",
        data=csv,
        file_name="plant_disease_predictions_report.csv",
        mime="text/csv"
    )

else:
    st.info(" Upload one or multiple leaf images to get predictions.")
