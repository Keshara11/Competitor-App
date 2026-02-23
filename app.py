import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

st.title("📊 Industry & Region Sentiment Predictor")

# Load dataset to get dropdown options
df = pd.read_csv("competitor_analysis_dataset.csv")

industry = st.selectbox(
    "Select Industry",
    sorted(df["industry"].unique())
)

region = st.selectbox(
    "Select Region",
    sorted(df["region"].unique())
)

if st.button("Predict Sentiment"):

    input_data = pd.DataFrame({
        "industry": [industry],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)

    st.success(f"Predicted Sentiment: {prediction}")

    st.write("### Prediction Probabilities")
    st.write(dict(zip(model.classes_, probability[0])))