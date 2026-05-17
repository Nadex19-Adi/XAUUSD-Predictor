import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import os
import sys

# Add root directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.config import settings

st.set_page_config(page_title="XAUUSD Predictor Dashboard", layout="wide")

st.title("📈 XAUUSD Predictor Dashboard")

st.sidebar.header("System Status")
st.sidebar.success("API: Online")
st.sidebar.success("MT5: Disconnected")
st.sidebar.success("RAG Vector Store: Ready")

# API URL (assuming FastAPI runs on 8000)
API_URL = "http://localhost:8000/api/v1"

def get_prediction():
    try:
        response = requests.post(f"{API_URL}/predict")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
    return None

col1, col2, col3 = st.columns(3)

if st.button("Generate Live Prediction"):
    with st.spinner("Analyzing current market state & querying RAG memory..."):
        pred = get_prediction()
        if pred:
            col1.metric("Predicted Direction", pred["direction"].upper(), delta="Bullish" if pred["direction"] == "up" else "Bearish")
            col2.metric("Confidence Score", f"{pred['confidence'] * 100:.1f}%")
            col3.metric("Detected Regime", pred["regime"].replace("_", " ").title())

st.divider()

st.subheader("Live Chart")
# Fetch last day 5m data for visualization
@st.cache_data(ttl=300)
def load_data():
    df = yf.download(settings.SYMBOL, period="1d", interval="5m")
    return df

df = load_data()
if not df.empty:
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Failed to fetch live data from Yahoo Finance.")
