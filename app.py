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

# ---------------------- MODERN THEME ----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main background with gradient */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stRadio > label {
    font-size: 18px;
    font-weight: 600;
    color: white !important;
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 50px 40px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 40px;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    animation: fadeInDown 0.8s ease;
}

.hero-header h1 {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 10px;
    color: white !important;
}

.hero-header p {
    font-size: 20px;
    opacity: 0.95;
    color: white !important;
}

/* Glass-morphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(10px);
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin-bottom: 25px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

/* Input styling */
.stSelectbox > div > div, .stNumberInput > div > div > input, .stDateInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
    padding: 12px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}

.stSelectbox > div > div:focus-within, 
.stNumberInput > div > div > input:focus, 
.stDateInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Label styling */
label {
    font-weight: 600 !important;
    font-size: 15px !important;
    color: #2c3e50 !important;
    margin-bottom: 8px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 18px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Result card with animation */
.result-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white !important;
    padding: 40px;
    border-radius: 20px;
    margin-top: 30px;
    text-align: center;
    box-shadow: 0 15px 50px rgba(17, 153, 142, 0.4);
    animation: slideInUp 0.6s ease;
}

.result-card h3 {
    color: white !important;
    font-size: 24px;
    margin-bottom: 20px;
    opacity: 0.95;
}

.metric-value {
    font-size: 60px;
    font-weight: 800;
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    margin: 20px 0;
}

.result-card p {
    color: white !important;
    font-size: 16px;
    opacity: 0.9;
}

/* Info boxes */
.info-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 25px;
    border-radius: 15px;
    color: white;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
}

.info-box h3 {
    color: white !important;
    margin-bottom: 10px;
}

.info-box p {
    color: white !important;
    font-size: 15px;
    line-height: 1.6;
}

/* Stats cards */
.stat-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

.stat-number {
    font-size: 36px;
    font-weight: 700;
    color: #667eea;
    margin-bottom: 5px;
}

.stat-label {
    font-size: 14px;
    color: #7f8c8d;
    font-weight: 500;
}

/* Image containers */
.image-container {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    margin: 25px 0;
    transition: all 0.3s ease;
}

.image-container:hover {
    box-shadow: 0 15px 50px rgba(0,0,0,0.12);
}

/* Insights section */
.insights-section {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    margin: 20px 0;
}

.insights-section h3 {
    color: #667eea !important;
    font-size: 24px;
    margin-bottom: 15px;
    font-weight: 700;
}

.insights-section p, .insights-section ul {
    color: #2c3e50 !important;
    font-size: 16px;
    line-height: 1.8;
    text-align: justify;
}

.insights-section ul {
    margin-left: 20px;
}

.insights-section li {
    margin-bottom: 10px;
}

/* Animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

/* Success/Error messages */
.stSuccess, .stError {
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- NAVIGATION ----------------------
page = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Prediction", "📊 Model Performance", "💡 Insights"],
    index=0
)

# ============================================================
# 1️⃣ PAGE — PREDICTION
# ============================================================
if page == "🏠 Prediction":

    st.markdown('''
    <div class="hero-header">
        <h1>🧅 Onion Price Predictor</h1>
        <p>AI-Powered Agricultural Price Forecasting for Smart Trading</p>
    </div>
    ''', unsafe_allow_html=True)

    # Create two columns for better layout
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📍 Market & Product Details")
        
        market = st.selectbox("🏪 Select Market", all_markets, help="Choose the market location")
        
        col_a, col_b = st.columns(2)
        with col_a:
            variety = st.selectbox("🌱 Variety", varieties, help="Select onion variety")
        with col_b:
            grade = st.selectbox("⭐ Grade", grades, help="Quality grade")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💰 Price Range & Date")
        
        col_c, col_d = st.columns(2)
        with col_c:
            min_price = st.number_input("📉 Min Price (₹)", min_value=0, max_value=50000, value=375, help="Minimum expected price")
        with col_d:
            max_price = st.number_input("📈 Max Price (₹)", min_value=0, max_value=50000, value=1224, help="Maximum expected price")
        
        arrival_date = st.date_input("📅 Expected Arrival Date", value=default_date,
                                     min_value=MIN_DATE, max_value=MAX_DATE, help="Date for price prediction")
        
        st.markdown("</div>", unsafe_allow_html=True)

        if min_price > max_price:
            st.error("⚠️ Min price cannot be greater than max price.")

        if st.button("🚀 Predict Price"):
            with st.spinner("🔮 Analyzing market data..."):
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
                    <h3>🎯 Predicted Modal Price</h3>
                    <div class="metric-value">₹ {pred:.2f}</div>
                    <p>Model Accuracy: <b>{R2_SCORE*100:.2f}%</b> | Confidence: High</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(R2_SCORE)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Quick Guide")
        st.markdown("""
        **How to use:**
        1. Select your market location
        2. Choose variety and grade
        3. Enter price range
        4. Pick the date
        5. Click predict!
        
        **Model Features:**
        - 97.72% Accuracy
        - Real-time prediction
        - Based on historical data
        - XGBoost algorithm
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stats display
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">97.72%</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Model Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-number">{len(all_markets)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Markets Covered</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 2️⃣ PAGE — MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":

    st.markdown('''
    <div class="hero-header">
        <h1>📊 Model Performance Analytics</h1>
        <p>Comprehensive evaluation metrics and visualizations</p>
    </div>
    ''', unsafe_allow_html=True)

    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">97.72%</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">R² Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">XGBoost</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Algorithm</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">High</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Reliability</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Images with better containers
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Actual vs Predicted Prices")
    st.image("actual_predicted_plot.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("### 📉 Residual Analysis")
    st.image("residual_plot.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Model Comparison Table")
    st.image("model_table.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 3️⃣ PAGE — INSIGHTS
# ============================================================
elif page == "💡 Insights":

    st.markdown('''
    <div class="hero-header">
        <h1>💡 Market Insights & Analysis</h1>
        <p>Understanding onion market dynamics in India</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="insights-section">', unsafe_allow_html=True)
    st.markdown("""
    ### 🧅 Importance of Onion in Indian Economy
    
    Onions are one of India's most consumed vegetables and a critical part of the nation's food supply chain. 
    Nashik—especially Lasalgaon—is responsible for nearly **40% of India's onion distribution** and is Asia's 
    largest onion hub. The market dynamics here directly influence prices across the entire country.

    ### 🌦️ Seasonal Factors Affecting Market Prices
    
    Onion prices exhibit significant seasonal variation influenced by multiple factors:
    
    - **Rabi Season (January–March):** Fresh harvest leads to stable prices and good supply
    - **Monsoon Period (June–September):** Storage damage and high humidity cause price volatility
    - **Festival Seasons:** Increased demand during festivals leads to temporary price spikes
    - **Extreme Weather:** Heavy rainfall or drought conditions can disrupt supply chains
    
    ### 🚢 Export–Import Dynamics
    
    International trade plays a crucial role in domestic pricing. Government export bans, international demand fluctuations, 
    and crop failures in major producing regions directly impact wholesale prices. Any disruption in Nashik's supply chain 
    instantly affects prices across India, making it a critical barometer for national onion economics.

    ### 📈 Why Accurate Price Prediction Matters
    
    - **For Farmers:** Enables better decisions on whether to store or sell immediately, maximizing profits
    - **For Traders:** Helps plan inventory management and optimize logistics operations
    - **For Policymakers:** Provides early warning signals for potential inflation risks
    - **For Consumers:** Indirect benefits through more stable market conditions
    
    ### 🚀 XGBoost Model Advantages
    
    The machine learning model deployed in this application achieves **97.72% accuracy** in price prediction through:
    
    - Analysis of historical price patterns and trends
    - Consideration of seasonal variations and market dynamics
    - Integration of multiple market parameters
    - Real-time prediction capabilities for informed decision-making
    
    ### 📘 Future Enhancement Opportunities
    
    - **Weather Integration:** Incorporate rainfall, temperature, and humidity data
    - **Supply Chain Metrics:** Add transportation delays and logistics data
    - **Deep Learning:** Implement LSTM networks for long-term forecasting
    - **Mobile Application:** Develop farmer-friendly mobile interface
    - **Real-time Updates:** Connect to live market data feeds
    - **Multi-commodity Support:** Extend to other agricultural commodities
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.success("✅ This comprehensive analysis provides valuable insights for stakeholders across the onion supply chain.")
