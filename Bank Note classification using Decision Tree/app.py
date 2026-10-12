import joblib
import streamlit as st
import numpy as np

model = joblib.load("banknote_model.joblib")

st.title("Banknote authenticity prediction")
st.write("Enter the four measurements, then select Predict.")

variance = st.number_input("Variance", value=0.0)
skewness = st.number_input("Skewness", value=0.0)
curtosis = st.number_input("Curtosis", value=0.0)
entropy = st.number_input("Entropy", value=0.0)

if st.button("Predict"):
    measurements = np.array([[variance, skewness, curtosis, entropy]])
    prediction = model.predict(measurements)[0]

    if prediction == 0:
        st.success("Prediction: genuine banknote")
    else:
        st.warning("Prediction: counterfeit banknote")