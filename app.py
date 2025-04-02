import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
import os

def load_models():
    stage1_model = joblib.load("stage1_gbm.pkl")
    stage2_model = joblib.load("stage2_gbm.pkl")
    
    # Load the scaler if available
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        st.error("Missing scaler.pkl. Please retrain and save the MinMaxScaler.")
        scaler = None
    
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

def recommend_products(df, allocation, risk_tolerance, top_n=3):
    df_filtered = df[df["Risk_Level"] == risk_tolerance]
    return df_filtered.sort_values(by='Expected_Return (%)', ascending=False).head(top_n)

st.title("Financial Advisory Powered by AI")
st.sidebar.header("User Financial Input")

income = st.sidebar.number_input("Monthly Household Income (Rs.)", min_value=1000, step=100)
expense = st.sidebar.number_input("Monthly Expense (Rs.)", min_value=100, step=50)
EMI_or_Rent_Amt = st.sidebar.number_input("Monthly EMI/Rent Amount (Rs.)", min_value=0, step=50)    
debt_ratio = st.sidebar.slider("Debt-to-Income Ratio (%)", 0, 100, 30)
investment_horizon = st.sidebar.selectbox("Investment Horizon", ["Short", "Medium", "Long"])
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
investment_experience = st.sidebar.selectbox("Investment Experience", ["Beginner", "Intermediate", "Advanced"])
Preferred_Investment_Type = st.sidebar.selectbox("Preferred Investment Type", ["Equity", "Mutual Fund", "Debt", "Gold", "Real Estate"])

if st.sidebar.button("Generate Investment Plan"):
    expected_features = [
        'Mthly_HH_Income', 'Mthly_HH_Expense', 'Emi_or_Rent_Amt',
        'No_of_Earning_Members', 'Savings_Amount', 'Investment_Horizon',
        'Risk_Tolerance', 'Investment_Experience', 'Market_Volatility_Tolerance',
        'Short_Term_Goal', 'Mid_Term_Goal', 'Long_Term_Goal', 'Goal_Based_Investing',
        'Preferred_Investment_Type', 'Adjusted_DTI', 'Savings_Rate', 
        'Disposable_Income', 'Debt_to_Income_Ratio'
    ]
    
    user_data = {
        'Mthly_HH_Income': income,
        'Mthly_HH_Expense': expense,  # Example assumption
        'Emi_or_Rent_Amt': EMI_or_Rent_Amt,  # Example assumption
        'No_of_Earning_Members': 2,  # Placeholder
        'Savings_Amount': income - expense - EMI_or_Rent_Amt,  # Example assumption
        'Investment_Horizon': investment_horizon,  # Example assumption
        'Risk_Tolerance': risk_tolerance,  # Example assumption
        'Investment_Experience': investment_experience,  # Example assumption
        'Market_Volatility_Tolerance': 4,  # Example assumption
        'Short_Term_Goal': 1,
        'Mid_Term_Goal': 1,
        'Long_Term_Goal': 1,
        'Goal_Based_Investing': 1,
        'Preferred_Investment_Type': Preferred_Investment_Type,  # Example assumption
        'Adjusted_DTI': debt_ratio / 100,
        'Savings_Rate': (income - expense - EMI_or_Rent_Amt) / income if income > 0 else 0,
        'Disposable_Income': income - expense - EMI_or_Rent_Amt,
        'Debt_to_Income_Ratio': debt_ratio / 100
    }
    
    X_input = pd.DataFrame([user_data], columns=expected_features)
   
    if scaler is None:
        st.error("Scaler is missing. Please retrain and save it.")
    else:
        st.write("Scaler is loaded successfully.")

        st.write("Expected features from scaler:", scaler.feature_names_in_)
        st.write("Columns in X_input:", X_input.columns.tolist())
    if scaler:
        X_input_scaled = scaler.transform(X_input)
    else:
        st.error("Scaler is missing. Please retrain and save it.")
        X_input_scaled = X_input  # Fallback to raw data
    
    invest_percentage = stage1_model.predict(X_input_scaled)[0]
    allocation = stage2_model.predict(X_input_scaled)[0]
    
    st.subheader("📌 Investment Allocation")
    
    st.write(f"Recommended Investment: {invest_percentage:.2f}% of Savings")
    st.write(f"Equity: {allocation[0]:.2f}%, Debt: {allocation[1]:.2f}%, Mutual Fund: {allocation[2]:.2f}%")
    
    st.subheader("📈 Recommended Stocks")

    stocks_df = get_stock_data()

    mf_df = get_mutual_fund_data()
    st.write("Fetched Stock Data:", stocks_df)
    st.write("Fetched Mutual Fund Data:", mf_df)
    recommended_stocks = recommend_products(stocks_df, allocation[0], risk_tolerance)
    recommended_mf = recommend_products(mf_df, allocation[2], risk_tolerance)

    

    
    
