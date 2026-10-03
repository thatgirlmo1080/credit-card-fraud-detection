import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection")

st.write(
    "This app uses a Logistic Regression model to predict "
    "whether a credit card transaction is fraudulent."
)

legitimate_sample = {
    "Time": 160760.0,
    "V1": -0.674466,
    "V2": 1.408105,
    "V3": -1.110622,
    "V4": -1.328366,
    "V5": 1.388996,
    "V6": -1.308439,
    "V7": 1.885879,
    "V8": -0.614233,
    "V9": 0.311652,
    "V10": 0.650757,
    "V11": -0.857785,
    "V12": -0.229961,
    "V13": -0.199817,
    "V14": 0.266371,
    "V15": -0.046544,
    "V16": -0.741398,
    "V17": -0.605617,
    "V18": -0.392568,
    "V19": -0.162648,
    "V20": 0.394322,
    "V21": 0.080084,
    "V22": 0.810034,
    "V23": -0.224327,
    "V24": 0.707899,
    "V25": -0.135837,
    "V26": 0.045102,
    "V27": 0.533837,
    "V28": 0.291319,
    "Amount": 23.0
}

fraud_sample = {
    "Time": 406.0,
    "V1": -2.312227,
    "V2": 1.951992,
    "V3": -1.609851,
    "V4": 3.997906,
    "V5": -0.522188,
    "V6": -1.426545,
    "V7": -2.537387,
    "V8": 1.391657,
    "V9": -2.770089,
    "V10": -2.772272,
    "V11": 3.202033,
    "V12": -2.899907,
    "V13": -0.595222,
    "V14": -4.289254,
    "V15": 0.389724,
    "V16": -1.140747,
    "V17": -2.830056,
    "V18": -0.016822,
    "V19": 0.416956,
    "V20": 0.126911,
    "V21": 0.517232,
    "V22": -0.035049,
    "V23": -0.465211,
    "V24": 0.320198,
    "V25": 0.044519,
    "V26": 0.177840,
    "V27": 0.261145,
    "V28": -0.143276,
    "Amount": 0.0
}

st.subheader("Test a Transaction")

choice = st.selectbox(
    "Choose a sample transaction",
    ["Legitimate Transaction", "Fraudulent Transaction"]
)

if st.button("Predict Transaction"):

    if choice == "Legitimate Transaction":
        transaction = legitimate_sample
    else:
        transaction = fraud_sample

    data = pd.DataFrame([transaction])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    if prediction == 1:
        st.error("Potential Fraud Detected")
    else:
        st.success("Transaction Appears Legitimate")

    st.write(f"Fraud probability: {probability:.2%}")

st.write("---")

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "97.55%")
col2.metric("Fraud Recall", "92.00%")
col3.metric("ROC-AUC", "97.21%")
col4.metric("PR-AUC", "71.90%")

st.subheader("About")

st.write(
    "The model was trained using Logistic Regression. "
    "Class weights were used because fraudulent transactions "
    "are much less common than legitimate transactions."
)