import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.title("🧠 Pandemica - Disease Outbreak Estimator (No AI)")
st.subheader("Simple statistical estimation for disease trend detection")

st.sidebar.header("Simulate or Upload Data")
data_mode = st.sidebar.radio("Choose Input Mode:", ("Simulate Synthetic Data", "Upload CSV"))

symptoms = [
    'fever_cases',
    'rash_cases',
    'platelet_alerts',
    'malnutrition_cases',
    'conjunctivitis_cases',
    'jaundice_cases'
]

if data_mode == "Simulate Synthetic Data":
    days = np.arange(30)
    data = pd.DataFrame({
        'day': days,
        'fever_cases': np.random.poisson(50, 30),
        'rash_cases': np.random.poisson(5, 30),
        'platelet_alerts': np.random.poisson(2, 30),
        'malnutrition_cases': np.random.poisson(4, 30),
        'conjunctivitis_cases': np.random.poisson(3, 30),
        'jaundice_cases': np.random.poisson(1, 30),
    })
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if not all(symptom in data.columns for symptom in symptoms):
            st.error(f"CSV must include: {', '.join(symptoms)}")
            st.stop()
    else:
        st.warning("Upload CSV file with all symptom columns.")
        st.stop()

st.write("### Clinical Data Preview:")
st.dataframe(data.head())

# Simple heuristic: average of last 5 days + 10% increase
last_days = data[symptoms].iloc[-5:]
mean_last = last_days.mean()
predicted_values = mean_last * 1.1

# Outbreak alert if 35% jump
overall_mean = data[symptoms].mean()
alert_triggered = any(
    (predicted_values[i] - overall_mean[i]) > overall_mean[i] * 0.35
    for i in range(len(symptoms))
)

if alert_triggered:
    st.error("🚨 A disease outbreak may occur!")
else:
    st.success("👍 No major outbreak trend detected.")

# Plotting
st.write("### Symptom Trends and Forecasts")
for symptom in symptoms:
    fig, ax = plt.subplots()
    ax.plot(data['day'], data[symptom], label='Actual', marker='o')
    ax.plot([data['day'].iloc[-1] + 1], [predicted_values[symptom]],
            label='Forecast', marker='X', color='red', markersize=10)
    ax.set_xlabel("Day")
    ax.set_ylabel(symptom.replace('_', ' ').capitalize())
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.caption("💡 Pandemica - Surveillance Without AI")
