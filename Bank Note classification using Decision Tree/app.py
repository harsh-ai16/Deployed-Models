from pathlib import Path

import joblib
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "banknote_model.joblib"

st.set_page_config(page_title="Banknote authenticity", page_icon="💵")
st.title("Banknote authenticity prediction")
st.write("Enter the four measurements used by the trained model.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure banknote_model.joblib is in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.exception(exc)
    st.stop()

variance = st.number_input("Variance", value=0.0, format="%.6f")
skewness = st.number_input("Skewness", value=0.0, format="%.6f")
curtosis = st.number_input("Curtosis", value=0.0, format="%.6f")
entropy = st.number_input("Entropy", value=0.0, format="%.6f")

if st.button("Predict", type="primary"):
    features = np.array([[variance, skewness, curtosis, entropy]], dtype=float)
    prediction = int(model.predict(features)[0])

    if prediction == 0:
        st.success("Prediction: genuine banknote")
    elif prediction == 1:
        st.warning("Prediction: counterfeit banknote")
    else:
        st.info(f"Model returned class {prediction}.")
