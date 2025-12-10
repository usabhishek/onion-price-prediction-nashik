import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime

MODEL_PATH = "onion_price_xgboost.pkl"
R2_DISPLAY_PERCENT = "98%"
MIN_DATE = date(2002, 4, 1)
MAX_DATE = date(2025, 12, 6)

# Auto-correct arrival_date default if today > MAX_DATE
default_date = date.today() if date.today() <= MAX_DATE else MAX_DATE

all_markets = [
    "Chandvad", "Chandvad APMC", "Devala", "Devala APMC", "Dindori", "Dindori(Vani)",
    "Dindori(Vani) APMC", "Ghoti", "Kalvan", "Kalvan APMC", "Lasalgaon",
    "Lasalgaon APMC", "Lasalgaon(Niphad)", "Lasalgaon(Niphad) APMC",
    "Lasalgaon(Vinchur)", "Lasalgaon(Vinchur) APMC", "Malegaon",
    "Malharshree Farmers Producer Co Ltd",
    "Mankamneshwar Farmar Producer CoLtd Sanchalit Mank",
    "Manmad", "Manmad APMC", "Nampur", "Nampur APMC", "Nandgaon", "Nandgaon APMC",
    "Nashik(Devlali)", "Nasik", "Nasik APMC", "Pimpalgaon", "Pimpalgaon APMC",
    "Pimpalgaon Baswant(Saykheda)",
    "Pimpalgaon Baswant(Saykheda) APMC", "Premium Krushi Utpanna Bazar",
    "Satana", "Satana APMC", "Shivsiddha Govind Producer Company Limited Sanchal",
    "Shivsiddha Govind Producer Company Limited Sanchal APMC",
    "Shree Rameshwar Krushi Market", "Sinner", "Sinner APMC", "Suragana",
    "Umrane", "Umrane APMC", "Yeola", "Yeola APMC"
]

varieties = ["Other", "Pole", "Red", "1st Sort", "White", "Dry F.A.Q.", "Onion", "Local variety"]
grades = ["FAQ", "Local"]


# =====================================================================
# PAGE CONFIGURATION + CUSTOM CSS (Modern UI like 2nd image)
# =====================================================================
st.set_page_config(page_title="Onion Price Predictor", layout="wide", page_icon="🧅")

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main-container {
    max-width: 1100px;
    margin-left: auto;
    margin-right: auto;
}

.header-box {
    background: linear-gradient(90deg, #F6F9ED, #FFFFFF);
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 25px;
    text-align: center;
    border: 1px solid #E5E5E5;
}

.form-card {
    background: #FFFFFF;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.06);
    border: 1px solid #EFEFEF;
}

input, select, .stDateInput input {
    border-radius: 8px !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #7CB342, #558B2F);
    color: white;
    padding: 14px;
    border-radius: 8px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #8BC34A, #689F38);
    color: white;
}

.result-card {
    background: linear-gradient(135deg, #2E7D32, #1B5E20);
    color: white;
    padding: 25px;
    border-radius: 14px;
    margin-top: 20px;
}

.metric-title {
    font-size: 20px;
    font-weight: 600;
}

.metric-value {
    font-size: 42px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =====================================================================
# LOAD MODEL
# =====================================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except:
        st.error("Model could not be loaded.")
        return None

model = load_model(MODEL_PATH)


# =====================================================================
# PAGE LAYOUT
# =====================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Performance", "Insights"])
st.sidebar.markdown("---")
st.sidebar.write("Model: **XGBoost**")
st.sidebar.write(f"R² Score: **{R2_DISPLAY_PERCENT}**")


# =====================================================================
# PREDICTION PAGE (Main UI similar to 2nd image)
# =====================================================================
if page == "Prediction":
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="header-box">
            <h2>🧅 Onion Price Prediction</h2>
            <p>AI-Powered Agricultural Price Forecasting</p>
        </div>
    """, unsafe_allow_html=True)

    # Form
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        market = st.selectbox("Market", all_markets)
        variety = st.selectbox("Variety", varieties)
        grade = st.selectbox("Grade", grades)

    with c2:
        arrival_date = st.date_input("Arrival Date", value=default_date,
                                     min_value=MIN_DATE, max_value=MAX_DATE)
        min_price = st.number_input("Min Price (₹)", value=375, min_value=0)
        max_price = st.number_input("Max Price (₹)", value=1224, min_value=0)

    st.markdown('</div>', unsafe_allow_html=True)

    if min_price > max_price:
        st.error("Min price cannot be greater than max price.")

    # Predict Button
    if st.button("Predict Price"):
        if model is not None:

            df_input = pd.DataFrame({
                "market": [market],
                "variety": [variety],
                "grade": [grade],
                "min_price": [min_price],
                "max_price": [max_price],
                "day": [arrival_date.day],
                "month": [arrival_date.month],
                "year": [arrival_date.year]
            })

            pred = model.predict(df_input)[0]

            # Result UI
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-title">Predicted Modal Price</div>
                <div class="metric-value">₹ {pred:.2f}</div>
                <p>This model explains <b>{R2_DISPLAY_PERCENT}</b> of market price variation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(0.98)

    st.markdown('</div>', unsafe_allow_html=True)
