import streamlit as st
import joblib
import numpy as np

st.title("Neural Latency Prediction System")
st.subheader("System Overview")

st.markdown("""
This system predicts neural response latency based on signal characteristics.
Reducing latency improves prosthetic response speed and overall system efficiency.
""")

st.write("Predict neural response latency for prosthetic control optimization.")

model = joblib.load("latency_model.pkl")

amplitude = st.slider("Amplitude", 0.0, 10.0, 5.0)
frequency = st.slider("Frequency", 0.0, 100.0, 50.0)
noise_level = st.slider("Noise Level", 0.0, 1.0, 0.5)
electrode_distance = st.slider("Electrode Distance", 0.0, 2.0, 1.0)
stimulus_intensity = st.slider("Stimulus Intensity", 0.0, 20.0, 10.0)

if st.button("Predict Latency"):
    input_data = np.array([[amplitude, frequency, noise_level, electrode_distance, stimulus_intensity]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Latency: {prediction[0]:.3f}")
st.subheader("Key Insight")

st.write("Noise level and electrode distance are major contributors to neural latency.")