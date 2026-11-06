import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("Credit Card Fraud Detection Dashboard")
st.write("Upload transaction data to detect potentially fraudulent transactions.")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Load trained machine learning model
try:
    with open("fraud_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please keep fraud_model.pkl in the project folder.")
    st.stop()

st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

st.info(
    "Your CSV must contain 30 input columns: Time, V1 to V28, and Amount. "
    "The Class column is not required."
)

if uploaded_file is not None:
    transactions = pd.read_csv(uploaded_file)

    missing_columns = [
        column for column in FEATURES
        if column not in transactions.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")

    else:
        input_data = transactions[FEATURES]

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        results = transactions.copy()

        results["Prediction"] = [
            "Fraud" if prediction == 1 else "Genuine"
            for prediction in predictions
        ]

        results["Fraud Probability (%)"] = (
            probabilities * 100
        ).round(2)

        fraud_count = int((predictions == 1).sum())
        genuine_count = int((predictions == 0).sum())

        st.subheader("Detection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(results))
        col2.metric("Fraud Detected", fraud_count)
        col3.metric("Genuine Transactions", genuine_count)

        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

st.divider()

st.caption(
    "Educational machine learning prototype using an anonymized historical "
    "credit card transaction dataset. Not intended for production banking use."
)