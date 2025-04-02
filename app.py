import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import requests
from bs4 import BeautifulSoup

def load_models():
    stage1_model = joblib.load("stage1_gbm.pkl")
    stage2_model = joblib.load("stage2_gbm.pkl")
    return stage1_model, stage2_model

stage1_model, stage2_model = load_models()

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

def recommend_products(df, allocation, risk_tolerance, top_n=3):
    df_filtered = df[df["Risk_Level"] == risk_tolerance]
    return df_filtered.sort_values(by='Expected_Return (%)', ascending=False).head(top_n)

st.title("Financial_Adivisary_Powered_by_AI")
st.sidebar.header("User Financial Input")

income = st.sidebar.number_input("Monthly Household Income ($)", min_value=1000, step=100)
savings = st.sidebar.number_input("Monthly Savings ($)", min_value=100, step=50)
debt_ratio = st.sidebar.slider("Debt-to-Income Ratio (%)", 0, 100, 30)
investment_horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
investment_experience = st.sidebar.selectbox("Investment Experience", ["Beginner", "Intermediate", "Advanced"])

if st.sidebar.button("Generate Investment Plan"):
    X_input = np.array([[income, savings, debt_ratio]])  # Placeholder
    print("Model expects:", stage1_model.n_features_in_)  # Prints the expected number of features
    print("X_input shape:", X_input.shape)  # Prints the actual input shape
    invest_percentage = stage1_model.predict(X_input)[0]
    allocation = stage2_model.predict(X_input)[0]
    
    st.subheader("📌 Investment Allocation")
    st.write(f"Recommended Investment: {invest_percentage:.2f}% of Savings")
    st.write(f"Equity: {allocation[0]:.2f}%, Debt: {allocation[1]:.2f}%, Mutual Fund: {allocation[2]:.2f}%")
    
    st.subheader("📈 Recommended Stocks")
    stocks_df = get_stock_data()
    st.dataframe(recommend_products(stocks_df, allocation[0], risk_tolerance))
    
    st.subheader("📊 Recommended Mutual Funds")
    mf_df = get_mutual_fund_data()
    st.dataframe(recommend_products(mf_df, allocation[2], risk_tolerance))
