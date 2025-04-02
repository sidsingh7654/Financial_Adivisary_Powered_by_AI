import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler

def load_models():
    stage1_model = joblib.load("stage1_gbm.pkl")
    stage2_model = joblib.load("stage2_gbm.pkl")
    scaler = joblib.load("scaler.pkl")  # Load the same scaler used in training
    return stage1_model, stage2_model, scaler

stage1_model, stage2_model, scaler = load_models()

def get_stock_data():
    stock_list = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "BAJFINANCE.NS"]
    stock_data = []
    for ticker in stock_list:
        stock = yf.Ticker(ticker)
        info = stock.info
        stock_data.append({
            "Product_Name": info.get('longName', ticker),
            "Expected_Return (%)": round(info.get('forwardPE', 10),2),
            "Risk_Level": "High" if info.get('beta', 1) > 1 else "Medium",
            "Volatility_Level": "High" if info.get('beta', 1) > 1 else "Medium"
        })
    return pd.DataFrame(stock_data)

def get_mutual_fund_data():
    response = requests.get("https://www.amfiindia.com/spages/NAVAll.txt")
    data = response.text
    mf_data = []
    for line in data.split("\n"):
        tokens = line.strip().split(";")
        if len(tokens) > 5:
            mf_data.append({
                "Product_Name": tokens[3],
                "Expected_Return (%)": 8,  # Placeholder
                "Risk_Level": "Medium",
                "Volatility_Level": "Medium"
            })
    return pd.DataFrame(mf_data)

def recommend_products(df, risk_tolerance, top_n=3):
    df_filtered = df[df["Risk_Level"] == risk_tolerance]
    return df_filtered.sort_values(by='Expected_Return (%)', ascending=False).head(top_n)

st.title("Financial Advisory Powered by AI")
st.sidebar.header("User Financial Input")

income = st.sidebar.number_input("Monthly Household Income ($)", min_value=1000, step=100)
savings = st.sidebar.number_input("Monthly Savings ($)", min_value=100, step=50)
debt_ratio = st.sidebar.slider("Debt-to-Income Ratio (%)", 0, 100, 30)
investment_horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
investment_experience = st.sidebar.selectbox("Investment Experience", ["Beginner", "Intermediate", "Advanced"])

if st.sidebar.button("Generate Investment Plan"):
    user_data = {
        'Mthly_HH_Income': income,
        'Mthly_HH_Expense': income * 0.4,
        'Emi_or_Rent_Amt': income * 0.2,
        'No_of_Earning_Members': 2,
        'Savings_Amount': savings,
        'Investment_Horizon': {"Short": 1, "Medium": 2, "Long": 3}[investment_horizon],
        'Risk_Tolerance': {"Low": 1, "Medium": 2, "High": 3}[risk_tolerance],
        'Investment_Experience': {"Beginner": 1, "Intermediate": 2, "Advanced": 3}[investment_experience],
        'Market_Volatility_Tolerance': 3,
        'Short_Term_Goal': 1,
        'Mid_Term_Goal': 1,
        'Long_Term_Goal': 1,
        'Goal_Based_Investing': 1,
        'Preferred_Investment_Type': 2,
        'Adjusted_DTI': debt_ratio / 100,
        'Savings_Rate': savings / income,
        'Disposable_Income': income - (income * 0.4) - (income * 0.2),
        'Debt_to_Income_Ratio': debt_ratio / 100
    }

    expected_features = list(user_data.keys())
    X_input = pd.DataFrame([user_data], columns=expected_features)
    X_input_scaled = scaler.transform(X_input)

    invest_percentage = stage1_model.predict(X_input_scaled)[0]
    allocation = stage2_model.predict(X_input_scaled)[0]
    
    st.subheader("📌 Investment Allocation")
    st.write(f"Recommended Investment: {invest_percentage:.2f}% of Savings")
    st.write(f"Equity: {allocation[0]:.2f}%, Debt: {allocation[1]:.2f}%, Mutual Fund: {allocation[2]:.2f}%")
    
    st.subheader("📈 Recommended Stocks")
    stocks_df = get_stock_data()
    st.dataframe(recommend_products(stocks_df, risk_tolerance))
    
    st.subheader("📊 Recommended Mutual Funds")
    mf_df = get_mutual_fund_data()
    st.dataframe(recommend_products(mf_df, risk_tolerance))
