import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

MODEL_PATH = "onion_price_xgboost.pkl"
R2_DISPLAY_PERCENT = "98%"
MIN_DATE = date(2002, 4, 1)
MAX_DATE = date(2025, 12, 6)

all_markets = [
    "Chandvad", "Chandvad APMC", "Devala", "Devala APMC",
    "Dindori", "Dindori(Vani)", "Dindori(Vani) APMC", "Ghoti",
    "Kalvan", "Kalvan APMC", "Lasalgaon", "Lasalgaon APMC",
    "Lasalgaon(Niphad)", "Lasalgaon(Niphad) APMC",
    "Lasalgaon(Vinchur)", "Lasalgaon(Vinchur) APMC",
    "Malegaon", "Malharshree Farmers Producer Co Ltd",
    "Mankamneshwar Farmar Producer CoLtd Sanchalit Mank",
    "Manmad", "Manmad APMC", "Nampur", "Nampur APMC",
    "Nandgaon", "Nandgaon APMC", "Nashik(Devlali)", "Nasik",
    "Nasik APMC", "Pimpalgaon", "Pimpalgaon APMC",
    "Pimpalgaon Baswant(Saykheda)",
    "Pimpalgaon Baswant(Saykheda) APMC",
    "Premium Krushi Utpanna Bazar", "Satana", "Satana APMC",
    "Shivsiddha Govind Producer Company Limited Sanchal",
    "Shivsiddha Govind Producer Company Limited Sanchal APMC",
    "Shree Rameshwar Krushi Market",
    "Sinner", "Sinner APMC", "Suragana", "Umrane",
    "Umrane APMC", "Yeola", "Yeola APMC"
]

varieties = ["Other", "Pole", "Red", "1st Sort", "White", "Dry F.A.Q.", "Onion", "Local variety"]
grades = ["FAQ", "Local"]

st.set_page_config(page_title="Onion Price Predictor", layout="wide", page_icon="🧅")

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.card {
    background: #ffffff;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}
.header {
    background: linear-gradient(90deg, #e8f5e9, white);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.result-card {
    background: linear-gradient(135deg, #0f9d58, #1b5e20);
    color: white;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except:
        st.error("Model could not be loaded.")
        return None

model = load_model(MODEL_PATH)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Model Performance", "Insights"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model: XGBoost**")
st.sidebar.markdown(f"**R² Score: {R2_DISPLAY_PERCENT}**")

if page == "Prediction":

    st.markdown('<div class="header"><h2>🧅 Onion Price Prediction</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        market = st.selectbox("Market", all_markets)
        variety = st.selectbox("Variety", varieties)
        grade = st.selectbox("Grade", grades)

    with c2:
        arrival_date = st.date_input("Arrival Date", value=date.today(), min_value=MIN_DATE, max_value=MAX_DATE)
        min_price = st.number_input("Min Price (₹)", value=375, min_value=0, max_value=50000)
        max_price = st.number_input("Max Price (₹)", value=1224, min_value=0, max_value=50000)

    st.markdown('</div>', unsafe_allow_html=True)

    if min_price > max_price:
        st.error("Min price cannot be greater than max price.")

    if st.button("Predict Price"):
        if model is not None:
            day = arrival_date.day
            month = arrival_date.month
            year = arrival_date.year

            df_input = pd.DataFrame({
                "market": [market],
                "variety": [variety],
                "grade": [grade],
                "min_price": [min_price],
                "max_price": [max_price],
                "day": [day],
                "month": [month],
                "year": [year]
            })

            pred = model.predict(df_input)[0]

            st.markdown(f"""
            <div class="result-card">
                <h3>Predicted Modal Price</h3>
                <h1>₹ {pred:.2f}</h1>
                <p>This model explains <b>{R2_DISPLAY_PERCENT}</b> of price variation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(0.98)

elif page == "Model Performance":

    st.markdown('<div class="header"><h2>📊 Model Performance</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Gradient Boosting", "Decision Tree", "KNN Regressor", "Linear Regression"],
        "MAE": [59.47, 68.01, 69.12, 76.54, 85.26, 88.72],
        "RMSE": [91.46, 103.99, 104.87, 120.84, 123.23, 132.53],
        "R² Score": [0.9772, 0.97055, 0.97005, 0.96023, 0.95864, 0.95217]
    })

    st.dataframe(comparison_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R² Score": "{:.4f}"}), height=300)
    st.markdown("### Best Model: **XGBoost**")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Insights":

    st.markdown('<div class="header"><h2>🌍 Market Insights & Key Findings</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("1️⃣ Importance of Onion Market in India")
    st.markdown("""
    - India is among the world's top onion producers.  
    - **Nashik district** contributes over **40%** of India's onion supply.  
    - Lasalgaon APMC is Asia’s **largest onion trading market**.  
    """)

    st.subheader("2️⃣ Seasonal Price Variation")
    st.markdown("""
    - **Rabi Season (Jan–March):** Highest supply → stable prices  
    - **Monsoon (Jun–Sept):** Storage damage → higher prices  
    - **Festival periods:** Sudden spikes due to increased demand  
    - **Heavy rainfall / crop disease:** Rapid inflation  
    """)

    st.subheader("3️⃣ Why Price Prediction is Valuable")
    st.markdown("""
    - Guides farmers in **storage vs. immediate sale** decisions  
    - Helps traders optimize **inventory and logistics**  
    - Supports government policy for **export bans and inflation control**  
    """)

    st.subheader("4️⃣ Future Enhancements")
    st.markdown("""
    - Integrate **rainfall, temperature, soil moisture** data  
    - Add **transportation cost** and **supply-chain delay**  
    - Use LSTM/Time-Series models for **long-term forecasting**  
    - Develop a mobile app for **real-time decision support**  
    """)

    st.success("These insights strengthen project documentation and viva presentation.")
