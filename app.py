import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
import os
import random
def load_models():
    stage1_model = joblib.load("stage1_gbm.pkl")
    stage2_model = joblib.load("stage2_gbm.pkl")
    scaler = joblib.load("scaler.pkl")
    # Check if the scaler has the correct feature names
    return stage1_model, stage2_model, scaler

stage1_model, stage2_model, scaler = load_models()

def get_random_stocks(n=5, risk_tolerance="Medium", investment_experience="Intermediate", investment_type="Equity"):
    # Sample stock universe with risk levels
    stock_universe = [
        {"ticker": "RELIANCE.NS", "risk": "High"},
        {"ticker": "INFY.NS", "risk": "Medium"},
        {"ticker": "TCS.NS", "risk": "Medium"},
        {"ticker": "HDFCBANK.NS", "risk": "Low"},
        {"ticker": "BAJFINANCE.NS", "risk": "High"},
        {"ticker": "TATAMOTORS.NS", "risk": "High"},
        {"ticker": "ITC.NS", "risk": "Low"},
        {"ticker": "MARUTI.NS", "risk": "Medium"},
        {"ticker": "WIPRO.NS", "risk": "Low"},
        {"ticker": "HINDUNILVR.NS", "risk": "Low"}
    ]

    # Filter stocks based on risk tolerance
    filtered_stocks = [s for s in stock_universe if s["risk"] == risk_tolerance]

    # If not enough stocks match the criteria, expand the selection
    if len(filtered_stocks) < n:
        filtered_stocks = stock_universe  # Use all stocks as fallback

    # Select random stocks
    selected_stocks = random.sample(filtered_stocks, min(n, len(filtered_stocks)))

    stock_data = []
    
    for stock_info in selected_stocks:
        ticker = stock_info["ticker"]
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if not hist.empty:
                last_close = hist["Close"].iloc[-1]
                pe_ratio = stock.info.get("forwardPE", None)
                beta = stock.info.get("beta", None)

                stock_data.append({
                    "Product_Name": stock.info.get("longName", ticker),
                    "Last_Close_Price (Rs.)": round(last_close, 2),
                    "Expected_Return (%)": round(pe_ratio, 2) if pe_ratio else "N/A",
                    "Risk_Level": stock_info["risk"],
                    "Volatility_Level": "High" if beta and beta > 1 else "Medium"
                })

        except Exception as e:
            print(f"⚠️ Error fetching data for {ticker}: {e}")

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
# Auto-calculate Savings Amount
savings = income - expense
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
        'Savings_Amount': savings,  # Example assumption
        'Investment_Horizon': 5,  # Example assumption
        'Risk_Tolerance': 3,  # Example assumption
        'Investment_Experience': 2,  # Example assumption
        'Market_Volatility_Tolerance': 4,  # Example assumption
        'Short_Term_Goal': 1,
        'Mid_Term_Goal': 1,
        'Long_Term_Goal': 1,
        'Goal_Based_Investing': 1,
        'Preferred_Investment_Type': Preferred_Investment_Type,  # Example assumption
        'Adjusted_DTI': debt_ratio / 100,
        'Savings_Rate': savings / income if income > 0 else 0,
        'Disposable_Income': income - expense - EMI_or_Rent_Amt,
        'Debt_to_Income_Ratio': debt_ratio / 100
    }
    
    X_input = pd.DataFrame([user_data], columns=expected_features)
   

    # Check expected vs actual features
    expected_features = list(scaler.feature_names_in_)
    actual_features = X_input.columns.tolist()

    
    # Identify mismatches
    missing_features = [feat for feat in expected_features if feat not in actual_features]
    extra_features = [feat for feat in actual_features if feat not in expected_features]

    if missing_features:
        st.error(f"❌ Missing Features in X_input: {missing_features}")
        st.stop()

    if extra_features:
        st.warning(f"⚠️ Extra Features in X_input: {extra_features}")
        X_input = X_input.drop(columns=extra_features)  # Remove extra features

    # Ensure columns are in correct order
    X_input = X_input[expected_features]

    # Check for NaN values
    if X_input.isnull().values.any():
        st.error(f"❌ Found NaN values in input data: {X_input.isnull().sum()}")
        st.stop()

    # Convert to NumPy and transform
    try:
        # Define encoding mapping (Ensure this matches training data!)
        investment_type_mapping = {
            "Equity": 0,
            "Mutual Fund": 1,
            "Debt": 2,
            "Gold": 3,
            "Real Estate": 4
        }

        # Convert categorical feature to numeric
        X_input["Preferred_Investment_Type"] = X_input["Preferred_Investment_Type"].map(investment_type_mapping)
        # Check if encoding is successful
        if X_input["Preferred_Investment_Type"].isnull().any():
            st.error("❌ Error: 'Preferred_Investment_Type' contains invalid values.")
            st.stop()

        # Now apply scaling
        X_input_scaled = scaler.transform(X_input)
        
    except Exception as e:
        st.error(f"❌ Error during scaling: {e}")
        st.stop()

    invest_percentage = stage1_model.predict(X_input_scaled)[0]
    allocation = stage2_model.predict(X_input_scaled)[0]
    
    st.subheader("📌 Investment Allocation")
    
    st.write(f"Recommended Investment: {invest_percentage:.2f}% of Savings")
    st.write(f"Equity: {allocation[0]:.2f}%, Debt: {allocation[1]:.2f}%, Mutual Fund: {allocation[2]:.2f}%")
    
    st.subheader("📈 Recommended Stocks")

    sstocks_df = get_random_stocks(n=5, risk_tolerance="High", investment_experience="Advanced", investment_type="Equity")


    mf_df = get_mutual_fund_data()
    st.write("Fetched Stock Data:", stocks_df)
    st.write("Fetched Mutual Fund Data:", mf_df)
    recommended_stocks = recommend_products(stocks_df, allocation[0], risk_tolerance)
    recommended_mf = recommend_products(mf_df, allocation[2], risk_tolerance)
    

    
    
