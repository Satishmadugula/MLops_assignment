import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Input Dashboard")
st.markdown("Use the sliders below to set wine features:")

# ---------------------------
# Sliders (live reruns)
# ---------------------------
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0, step=0.1)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, step=0.01)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5, step=0.1)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05, step=0.001)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15, step=1)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46, step=1)
density = st.slider("Density", 0.9900, 1.0040, 0.9960, step=0.0001, format="%.4f")
pH = st.slider("pH", 2.5, 4.5, 3.2, step=0.01)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6, step=0.01)
alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0, step=0.1)

FEATURE_ORDER = [
    "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",
]

raw_inputs = {
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
}

input_df = pd.DataFrame([raw_inputs], columns=FEATURE_ORDER)

st.subheader("🔎 Input Summary")
st.dataframe(input_df, use_container_width=True)

# ---------------------------
# Load model + scaler (cached)
# ---------------------------
MODEL_PATH = "Wine_quality_Random_forest_classifier.pkl"
SCALER_PATH = "Wine_quality_scaler.pkl"

@st.cache_resource
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model_obj = load_pickle(MODEL_PATH) if Path(MODEL_PATH).exists() else None
scaler_obj = load_pickle(SCALER_PATH) if Path(SCALER_PATH).exists() else None

if model_obj is None:
    st.warning(f"Model file not found at: {MODEL_PATH}")
if scaler_obj is None:
    st.warning(f"Scaler file not found at: {SCALER_PATH}")
if model_obj is None or scaler_obj is None:
    st.stop()

# ---------------------------
# Predict helpers
# ---------------------------
def predict_with_model(model_like, X_df: pd.DataFrame):
    label_encoder = None
    model = model_like
    if isinstance(model_like, dict):
        model = model_like.get("model", model_like)
        label_encoder = model_like.get("label_encoder")

    pred_enc = model.predict(X_df)

    proba_df, classes_display = None, None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_df)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            for step in reversed(model.named_steps.values()):
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break
        classes_display = classes
        if classes is not None and label_encoder is not None:
            try:
                classes_display = label_encoder.inverse_transform(classes.astype(int))
            except Exception:
                pass
        proba_df = pd.DataFrame(probs, columns=classes_display if classes_display is not None else None)

    pred_labels = pred_enc
    if label_encoder is not None:
        try:
            pred_labels = label_encoder.inverse_transform(pred_enc.astype(int))
        except Exception:
            pass

    return np.array(pred_labels), proba_df, classes_display

# ---------------------------
# Live prediction (argmax label + smooth expected value)
# ---------------------------
ordered = input_df.reindex(columns=FEATURE_ORDER)
scaled = scaler_obj.transform(ordered.values)
X_scaled_df = pd.DataFrame(scaled, columns=FEATURE_ORDER)

pred, proba_df, classes_display = predict_with_model(model_obj, X_scaled_df)

# Always show the class label that won this round
pred_label = str(pred[0])
st.subheader("🔮 Prediction")
st.metric(label="Predicted Class", value=pred_label)

# If probabilities are available, compute an EXPECTED quality (smooth signal)
if proba_df is not None and proba_df.shape[1] > 0:
    # Try to parse class labels as numbers; fallback to index order if not numeric
    try:
        class_vals = np.array([float(c) for c in proba_df.columns])
    except Exception:
        class_vals = np.arange(proba_df.shape[1], dtype=float)

    probs = proba_df.iloc[0].values.astype(float)
    expected_quality = float((class_vals * probs).sum())

    # st.metric(label="Expected Quality (prob-weighted)", value=f"{expected_quality:.2f}")

    # Show top-3 classes by probability so you can see movement
    top3 = proba_df.iloc[0].sort_values(ascending=False).head(3)
    st.markdown("**Top-3 class probabilities**")
    st.dataframe(top3.to_frame(name="prob").style.format("{:.3f}"), use_container_width=True)

    # Optional: show full probability table
    with st.expander("See all class probabilities"):
        st.dataframe(proba_df.style.format("{:.3f}"), use_container_width=True)

st.caption(
    "Why the label may look 'stuck': the arg-max class only changes when probabilities cross over. "
    "The expected value and probabilities above update smoothly on every slider move."
)
