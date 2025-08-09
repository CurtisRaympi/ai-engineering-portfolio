import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL):", "AAPL")
period = st.selectbox("Select Period", ['1mo', '3mo', '6mo', '1y', '2y'])

if ticker:
    data = yf.download(ticker, period=period)
    st.line_chart(data['Close'])
    
    st.write("Latest Close Price:", data['Close'][-1])
    
    # Here you can add your LSTM prediction or placeholder
    st.info("LSTM prediction model coming soon!")
