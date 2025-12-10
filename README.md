# 🧅 Onion Price Prediction – Maharashtra (Nashik)

Predicting **onion modal prices** using Machine Learning with real market data from **Nashik APMC markets (2005–2025)**.  
This project includes **data cleaning**, **EDA**, **model comparison**, **XGBoost training**, and a **Streamlit web application**.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/XGBoost-Best%20Model-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge" />
</p>

---

## 📌 Project Overview

This project predicts **Modal Price of Onion** for various markets in **Nashik District**, Maharashtra.  
A complete ML pipeline was implemented:

- Data Cleaning & Date Parsing  
- Outlier Detection using **Per-Market IQR**  
- Visualization & Exploratory Data Analysis  
- Regression Model Training (5+ algorithms)  
- Model Evaluation (MAE, RMSE, R² Score)  
- Final Deployment using **Streamlit**  
- Best Performing Model → **XGBoost Regression (R² = 0.9772)**  

---

## 🧹 Data Preprocessing

The dataset contains:

| Column | Description |
|--------|-------------|
| `market` | APMC market name |
| `variety` | Onion variety |
| `grade` | Quality grade |
| `arrival_date` | Date of arrival |
| `min_price` | Minimum price |
| `max_price` | Maximum price |
| `modal_price` | Target variable |
| `commodity_code` | Internal identifier |

### Preprocessing Steps
- Column renaming  
- Converting arrival_date → day, month, year  
- Removing invalid prices (negative & above ₹50,000)  
- Outlier removal per market using **IQR method**  
- One-hot encoding for categorical features  
- Scaling numeric features  

---

## 📊 Exploratory Data Analysis (EDA)

Key visualizations included:

- Distribution of Modal Prices  
- Market-wise price comparison  
- Variety-wise price variation  
- Seasonality trends (month-wise prices)  
- Correlation heatmap  
- Outlier analysis  
- Actual vs Predicted plot  
- Residual plots  

---

## 🤖 Machine Learning Models Used

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Decision Tree Regressor | Max depth tuned |
| Random Forest Regressor | Bagging ensemble |
| Gradient Boosting Regressor | Boosting |
| **XGBoost Regressor** | ⭐ Best model |
| KNN Regressor | Baseline |

---

## 🏆 Model Performance Summary

| Model | MAE | RMSE | R² Score |
|-------|------|---------|----------|
| **XGBoost Regressor** | **59.47** | **91.46** | **0.9772** |
| Bagging Regressor | 59.46 | 95.51 | 0.9751 |
| Random Forest | 62.23 | 97.33 | 0.9742 |
| Gradient Boosting | 69.12 | 104.86 | 0.9700 |
| Polynomial Regression | 79.07 | 118.90 | 0.9615 |
| Decision Tree | 76.54 | 120.84 | 0.9602 |
| KNN Regressor | 85.26 | 123.23 | 0.9586 |
| Linear Regression | 88.73 | 132.53 | 0.9521 |

**Final Model Used:** ✔ **XGBoost Regression Pipeline**

---

## 🏗 Project Structure

