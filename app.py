import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ---------------------- LOAD MODEL ----------------------
model = joblib.load("onion_price_xgboost.pkl")
R2_SCORE = 0.9772

# ---------------------- CONSTANTS ----------------------
MIN_DATE = date(2002, 4, 1)
MAX_DATE = date(2025, 12, 6)
default_date = date.today() if date.today() <= MAX_DATE else MAX_DATE

all_markets = [
    "Chandvad","Chandvad APMC","Devala","Devala APMC","Dindori","Dindori(Vani)",
    "Dindori(Vani) APMC","Ghoti","Kalvan","Kalvan APMC","Lasalgaon","Lasalgaon APMC",
    "Lasalgaon(Niphad)","Lasalgaon(Niphad) APMC","Lasalgaon(Vinchur)",
    "Lasalgaon(Vinchur) APMC","Malegaon","Malharshree Farmers Producer Co Ltd",
    "Mankamneshwar Farmar Producer CoLtd Sanchalit Mank","Manmad","Manmad APMC",
    "Nampur","Nampur APMC","Nandgaon","Nandgaon APMC","Nashik(Devlali)","Nasik",
    "Nasik APMC","Pimpalgaon","Pimpalgaon APMC","Pimpalgaon Baswant(Saykheda)",
    "Pimpalgaon Baswant(Saykheda) APMC","Premium Krushi Utpanna Bazar","Satana",
    "Satana APMC","Shivsiddha Govind Producer Company Limited Sanchal",
    "Shivsiddha Govind Producer Company Limited Sanchal APMC",
    "Shree Rameshwar Krushi Market","Sinner","Sinner APMC","Suragana","Umrane",
    "Umrane APMC","Yeola","Yeola APMC"
]

varieties = ["Other", "Pole", "Red", "1st Sort", "White", "Dry F.A.Q.", "Onion", "Local variety"]
grades = ["FAQ", "Local"]

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Onion Price Predictor", layout="wide")

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("Onion Price Prediction App")
st.sidebar.info("Single page app with Prediction + Model Performance")

page = "Prediction"  # Only one page now

# ============================================================
# PAGE — PREDICTION + MODEL PERFORMANCE BELOW
# ============================================================

st.title("🧅 Onion Price Prediction")

st.write("Fill the inputs below to predict the modal price:")

# ---------------------- USER INPUTS ----------------------
market = st.selectbox("Market", all_markets)
variety = st.selectbox("Variety", varieties)
grade = st.selectbox("Grade", grades)

min_price = st.number_input("Minimum Price (₹)", min_value=0, max_value=50000, value=375)
max_price = st.number_input("Maximum Price (₹)", min_value=0, max_value=50000, value=1224)

arrival_date = st.date_input(
    "Expected Date", 
    value=default_date,
    min_value=MIN_DATE,
    max_value=MAX_DATE
)

if min_price > max_price:
    st.error("Min price cannot be greater than max price.")

input_df = pd.DataFrame({
    "market": [market],
    "variety": [variety],
    "grade": [grade],
    "min_price": [min_price],
    "max_price": [max_price],
    "day": [arrival_date.day],
    "month": [arrival_date.month],
    "year": [arrival_date.year]
})

# ---------------------- PREDICT BUTTON ----------------------
if st.button("Predict Modal Price"):
    price = model.predict(input_df)[0]
    st.success(f"Predicted Modal Price: ₹ {price:.2f}")


    
st.markdown("---")
st.subheader("📊 Model Performance Visualizations")


st.write("### Model Comparison Table")

comparison_df = pd.DataFrame({
    "Model": ["XGBoost", "Random Forest", "Gradient Boosting", "Decision Tree", "KNN Regressor", "Linear Regression"],
    "MAE": [59.47, 68.01, 69.12, 76.54, 85.26, 88.72],
    "RMSE": [91.46, 103.99, 104.87, 120.84, 123.23, 132.53],
    "R² Score": [0.9772, 0.97055, 0.97005, 0.96023, 0.95864, 0.95217]
})

st.dataframe(
    comparison_df.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R² Score": "{:.4f}"
    })
)


st.write("### Actual vs Predicted")
st.image("actual_predicted_plot.png")

st.write("### Residual Plot")
st.image("residual_plot.png")

