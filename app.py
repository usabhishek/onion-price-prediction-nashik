import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

MODEL_PATH = "onion_price_xgboost.pkl"
R2_DISPLAY_PERCENT = "98%"
MIN_DATE = date(2002, 4, 1)
MAX_DATE = date(2025, 12, 6)

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
    "Pimpalgaon Baswant(Saykheda)", "Pimpalgaon Baswant(Saykheda) APMC",
    "Premium Krushi Utpanna Bazar", "Satana", "Satana APMC",
    "Shivsiddha Govind Producer Company Limited Sanchal",
    "Shivsiddha Govind Producer Company Limited Sanchal APMC",
    "Shree Rameshwar Krushi Market", "Sinner", "Sinner APMC", "Suragana",
    "Umrane", "Umrane APMC", "Yeola", "Yeola APMC"
]

varieties = ["Other", "Pole", "Red", "1st Sort", "White", "Dry F.A.Q.", "Onion", "Local variety"]
grades = ["FAQ", "Local"]

# ------------ PAGE CONFIG ------------
st.set_page_config(page_title="Onion Price Predictor", page_icon="🧅", layout="wide")

# ------------ UNIVERSAL CSS ------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: #000000 !important;
}

h1, h2, h3, h4, h5, h6, label, p, div, span {
    color: #000000 !important;
}

.main-container {
    max-width: 1150px;
    margin-left: auto;
    margin-right: auto;
}

.header-box {
    background: linear-gradient(90deg, #E8F5E9, #FFFFFF);
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 30px;
    text-align: center;
    border: 1px solid #DDDDDD;
}

.form-card, .card {
    background: #FFFFFF;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.06);
    border: 1px solid #E5E5E5;
    margin-bottom: 25px;
}

input, select, textarea, .stDateInput input {
    border-radius: 8px !important;
    border: 1px solid #CCCCCC !important;
    color: #000000 !important;
}

.stSelectbox > div > div {
    color: #000000 !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7CB342, #558B2F);
    color: white !important;
    padding: 14px;
    border-radius: 8px;
    font-size: 18px;
    border: none;
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
}

.metric-title {
    font-size: 20px;
    font-weight: 600;
    color: white !important;
}

.metric-value {
    font-size: 42px;
    font-weight: 800;
    color: white !important;
}

.dataframe caption, .dataframe th, .dataframe td {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

# ------------ MODEL LOADING ------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except:
        st.error("Model could not be loaded.")
        return None

model = load_model(MODEL_PATH)

# ------------ SIDEBAR ------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Prediction", "Model Performance", "Insights"])
st.sidebar.markdown("---")
st.sidebar.write("Model: **XGBoost**")
st.sidebar.write(f"R² Score: **{R2_DISPLAY_PERCENT}**")


# =====================================================================
# PAGE 1: PREDICTION
# =====================================================================
if page == "Prediction":

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown("""
        <div class="header-box">
            <h2>🧅 Onion Price Prediction</h2>
            <p>AI-powered forecasting engine for agricultural markets</p>
        </div>
    """, unsafe_allow_html=True)

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

            st.markdown(f"""
            <div class="result-card">
                <div class="metric-title">Predicted Modal Price</div>
                <div class="metric-value">₹ {pred:.2f}</div>
                <p>This model explains <b>{R2_DISPLAY_PERCENT}</b> of price variation.</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(0.98)

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE 2: MODEL PERFORMANCE
# =====================================================================
elif page == "Model Performance":

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown("""
        <div class="header-box">
            <h2>📊 Model Performance Overview</h2>
            <p>Regression accuracy & model comparison</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Gradient Boosting", "Decision Tree", "KNN Regressor", "Linear Regression"],
        "MAE": [59.47, 68.01, 69.12, 76.54, 85.26, 88.72],
        "RMSE": [91.46, 103.99, 104.87, 120.84, 123.23, 132.53],
        "R² Score": [0.9772, 0.97055, 0.97005, 0.96023, 0.95864, 0.95217]
    })

    st.dataframe(comparison_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R² Score": "{:.4f}"}), height=320)

    st.markdown("### ⭐ Best Model: **XGBoost**")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE 3: INSIGHTS
# =====================================================================
elif page == "Insights":

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown("""
        <div class="header-box">
            <h2>🌍 Market Insights & Key Findings</h2>
            <p>Understanding market behavior and price volatility</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("1️⃣ Importance of Onion Market in India")
    st.markdown("""
- India is among the world’s leading onion producers.  
- **Nashik district** contributes **40%+** of the country’s supply.  
- Lasalgaon APMC is Asia’s **largest onion trading hub**.  
""")

    st.subheader("2️⃣ Seasonal Price Variation")
    st.markdown("""
- **Rabi Season (Jan–March):** Stable and lower prices due to high supply.  
- **Monsoon Season (Jun–Sept):** Storage loss → price hikes.  
- **Festivals:** Sudden demand-driven spikes.  
- **Rainfall / crop disease:** Sharp inflation.  
""")

    st.subheader("3️⃣ Why Price Prediction Matters")
    st.markdown("""
- Helps farmers with **storage vs. immediate sale** decisions.  
- Helps traders plan **inventory, logistics, and transport**.  
- Supports government in **inflation control and export regulation**.  
""")

    st.subheader("4️⃣ Future Scope")
    st.markdown("""
- Add weather + soil parameters for increased accuracy.  
- Add transportation + supply-chain delay modeling.  
- Use deep learning (LSTM) for time-series prediction.  
- Build a mobile app for farmers and mandis.  
""")

    st.success("These insights strengthen project documentation, viva explanation, and overall presentation.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
