import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor (LSTM-style with Prophet)")

# Sidebar for user inputs
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, TSLA):", "AAPL")
years = st.sidebar.slider("Years to Predict:", 1, 4, 2)

# Load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="max")
    data.reset_index(inplace=True)
    return data

try:
    data = load_data(ticker)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Show raw data
st.subheader("📊 Raw Data Preview")
st.write(data.tail())

# Prepare data for Prophet
df = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
df.dropna(inplace=True)  # Remove NaNs
df["ds"] = pd.to_datetime(df["ds"])  # Ensure datetime format
df["y"] = pd.to_numeric(df["y"], errors="coerce")  # Force numeric
df.dropna(inplace=True)  # Drop rows that couldn't convert

# Train Prophet model
m = Prophet(daily_seasonality=True)
m.fit(df)

# Make future dataframe
future = m.make_future_dataframe(periods=years * 365)
forecast = m.predict(future)

# Plot forecast using Plotly for better visuals
st.subheader("🔮 Stock Price Forecast")
fig1 = plot_plotly(m, forecast)
fig1.update_layout(
    title=f"{ticker} Price Prediction",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    font=dict(size=14)
)
st.plotly_chart(fig1)

# Add a custom chart showing real vs predicted
st.subheader("📉 Historical vs Predicted Prices")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual Price", line=dict(color="blue")))
fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicted Price", line=dict(color="orange")))
fig2.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig2)

st.caption("⚠️ This is a demo prediction model using Prophet. Do not use for financial decisions.")
