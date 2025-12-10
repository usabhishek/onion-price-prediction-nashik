import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

model = joblib.load("onion_price_xgboost.pkl")

R2_SCORE = 0.98

top_markets = [
    "Shivsiddha Govind Producer Company Limited Sanchal",
    "Shree Rameshwar Krushi Market",
    "Malharshree Farmers Producer Co Ltd",
    "Premium Krushi Utpanna Bazar",
    "Dindori(Vani) APMC",
    "Shivsiddha Govind Producer Company Limited Sanchal APMC",
    "Pimpalgaon APMC",
    "Lasalgaon APMC",
    "Lasalgaon(Vinchur)",
    "Umrane",
    "Mankamneshwar Farmar Producer CoLtd Sanchalit Mank",
    "Umrane APMC",
    "Chandvad APMC",
    "Nandgaon APMC",
    "Nampur"
]
varieties = [
    "Other",
    "Pole",
    "Red",
    "1st Sort",
    "White",
    "Dry F.A.Q.",
    "Onion",
    "Local"
]

grades = ['FAQ', 'Local']

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Onion Price Prediction",
    layout="wide",
    page_icon="🧅"
)

# ---------- SIDEBAR NAVIGATION ----------
page = st.sidebar.radio(
    "Navigation",
    ["1️. Prediction", "2️. Model Performance", "3️. Insights"],
)

# PAGE 1 : PREDICTION

if page == "1️. Prediction":
    st.title("🧅 Onion Price Prediction")
    st.markdown("### Predict modal price based on market conditions")
    
    st.write("### Enter Input Details:")

    col1, col2 = st.columns(2)

    with col1:
        market = st.selectbox("Select Market", top_markets)
        variety = st.selectbox("Select Variety", varieties)
        grade = st.selectbox("Select Grade", grades)

    with col2:
        arrival_date = st.date_input("Select Date", value=date(2024, 1, 1))
        min_price = st.number_input("Min Price (₹)", min_value=0, max_value=50000, value=375)
        max_price = st.number_input("Max Price (₹)", min_value=0, max_value=50000, value=1224)

    day = arrival_date.day
    month = arrival_date.month
    year = arrival_date.year

    input_df = pd.DataFrame({
        "market": [market],
        "variety": [variety],
        "grade": [grade],
        "min_price": [min_price],
        "max_price": [max_price],
        "day": [day],
        "month": [month],
        "year": [year],
    })

    if st.button("Predict Modal Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"### Predicted Modal Price: **₹ {prediction:.2f}**")
        st.info(f"📈 Model R² Score: **{R2_SCORE:.2f}** (Explains {R2_SCORE*100:.2f}% of price variation)")


# PAGE 2 : MODEL PERFORMANCE
elif page == "2️. Model Performance":
    st.title("📊 Model Comparison & Performance Analysis")

    st.markdown("""
    ### Models Used:
    - Linear Regression   
    - Decision Tree  
    - Random Forest  
    - Gradient Boosting  
    - XGBoost  
    - KNN Regressor   
    """)

    st.markdown("---")

    st.subheader("Actual vs Predicted Curve")
    st.image("actual_predicted_plot.png")

    st.subheader("Residual Plot")
    st.image("residual_plot.png")

    st.subheader("Model Comparison Table")
    st.image("model_table.png")

# PAGE 3 : INSIGHTS & SUMMARY
elif page == "3️. Insights":
    st.title("🌍 Onion Market Insights & Key Understanding")

    st.markdown("""
    ### 🧅 Importance of Onion in Indian Economy
    - India is one of the world’s largest onion producers.
    - Maharashtra (especially Nashik) supplies **40%+** of India’s onion demand.
    - Nashik’s Lasalgaon Mandi is Asia’s largest onion market.

    ### 🚢 Export & Import Dynamics
    - Onion prices fluctuate sharply due to export bans, weather patterns, and storage losses.
    - Price spikes often occur when crops in **Kharif or Rabi seasons** are affected.

    ### 🌦 Seasonal Impact on Prices
    - **Summer–Monsoon**: Prices rise due to storage loss.
    - **Winter–Rabi Harvest**: Prices stabilize due to high supply.
    - Festivals, rainfall, crop disease → cause sudden surges.

    ### 📉 Why Prediction is Important
    - Helps farmers choose storage vs. immediate sale.
    - Helps wholesalers and retailers plan inventory.
    - Helps policymakers predict inflation trends.

    ### 🧠 Recommended Model
    - **XGBoost** (best performing model)
    - Explains **97.72%** of price variation.
    - Fast, stable, suitable for real-time apps.

    ### 📘 Future Enhancements
    - Add weather data (rainfall, humidity).
    - Incorporate supply-chain delays.
    - Train separate models for each district.
    - Deploy online for farmers and traders.
    """)

    st.success("This page will strengthen your project explanation in viva or documentation.")
