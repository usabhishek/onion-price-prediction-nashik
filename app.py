import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ---------------------- MODEL ----------------------
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
st.set_page_config(page_title="Onion Price Predictor", layout="wide", page_icon="🧅")

# ---------------------- GLOBAL THEME ----------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #000000 !important;
}

.header-box {
    background: linear-gradient(90deg, #E8F5E9, #FFFFFF);
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 25px;
    text-align: center;
    border: 1px solid #DDDDDD;
    color: black
}

.card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.05);
    border: 1px solid #E5E5E5;
    margin-bottom: 20px;
    color: black;
}

input, select, textarea, .stDateInput input {
    border-radius: 8px !important;
    border: 1px solid #333 !important;
    color: #111111 !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7CB342, #558B2F);
    color: white !important;
    padding: 14px;
    border-radius: 8px;
    font-size: 18px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #8BC34A, #689F38);
}

.result-card {
    background: linear-gradient(135deg, #2E7D32, #1B5E20);
    color: white !important;
    padding: 25px;
    border-radius: 14px;
    margin-top: 20px;
    text-align: center;
}

.metric-value {
    font-size: 40px;
    font-weight: 800;
}

.justify {
    text-align: justify;
    font-size: 17px;
    line-height: 1.6;
}

.center {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- NAVIGATION ----------------------
page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Model Performance", "Insights"]
)

# ============================================================
# 1️⃣ PAGE — PREDICTION
# ============================================================
if page == "Prediction":

    st.markdown('<div class="header-box"><h2>🧅 Onion Price Prediction</h2>'
                '<p>AI-Based Agricultural Price Forecasting</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    market = st.selectbox("Market", all_markets)
    variety = st.selectbox("Variety", varieties)
    grade = st.selectbox("Grade", grades)
    min_price = st.number_input("Min Price (₹)", min_value=0, max_value=50000, value=375)
    max_price = st.number_input("Max Price (₹)", min_value=0, max_value=50000, value=1224)
    arrival_date = st.date_input("Expected Date", value=default_date,
                                 min_value=MIN_DATE, max_value=MAX_DATE)

    st.markdown('</div>', unsafe_allow_html=True)

    if min_price > max_price:
        st.error("Min price cannot be greater than max price.")

    if st.button("Predict Price"):
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

        st.markdown(f"""
        <div class="result-card">
            <h3>Predicted Modal Price</h3>
            <div class="metric-value">₹ {pred:.2f}</div>
            <p>Model explains <b>{R2_SCORE*100:.2f}%</b> variation in prices.</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(R2_SCORE)

# ============================================================
# 2️⃣ PAGE — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.markdown('<div class="header-box"><h2>📊 Model Performance Overview</h2></div>', unsafe_allow_html=True)

    # Center images
    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.image("actual_predicted_plot.png", width=800)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.image("residual_plot.png", width=800)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="center">', unsafe_allow_html=True)
    st.image("model_table.png", width=800)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 3️⃣ PAGE — INSIGHTS
# ============================================================
elif page == "Insights":

    st.markdown('<div class="header-box"><h2>🌍 Onion Market Insights & Story</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="card justify">', unsafe_allow_html=True)

    st.markdown("""
### 🧅 Importance of Onion in Indian Economy
Onions are one of India's most consumed vegetables and a critical part of the nation's food supply chain. 
Nashik—especially Lasalgaon—is responsible for nearly **40% of India’s onion distribution** and is Asia’s 
largest onion hub.

### 🌦 Seasonal Factors Affecting Market Prices
Prices rise and fall sharply due to weather conditions, storage losses, and supply interruptions.  
- **Rabi (Jan–March):** Fresh harvest → stable prices  
- **Monsoon (June–Sept):** Storage damage → high volatility  
- **Festivals or heavy rainfall:** Sudden spikes  

### 🚢 Export–Import Dynamics
Government export bans, international demand, and crop failure directly impact wholesale prices.
Any disruption in Nashik instantly affects prices across India.

### 📈 Why Prediction Matters
- Helps farmers decide whether to **store or sell immediately**  
- Helps traders plan **inventory and logistics**  
- Helps policymakers anticipate **inflation risk**  

### 🚀 Best Performing Model: XGBoost
The model used in this project captures **97.72% price variation**, making it 
highly reliable for short-term price prediction.

### 📘 Future Scope
- Add weather data (rainfall, temperature, humidity)  
- Add transportation + supply chain delays  
- Use LSTM for long-term forecasting  
- Mobile portal for farmers  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    st.success("These insights make your project strong for viva, documentation, and demonstrations.")

