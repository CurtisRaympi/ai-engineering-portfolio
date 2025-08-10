import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor")

# Sidebar inputs
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, TSLA):", "AAPL")
years = st.sidebar.slider("Years to Predict:", 1, 4, 2)

# Load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="max")
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data

data = load_data(ticker)

if data is None or data.empty:
    st.error("❌ No data found for the given ticker. Please try another symbol.")
    st.stop()

# Show raw data preview
st.subheader("📊 Raw Data Preview")
st.dataframe(data.tail())

# Check if 'Close' exists
if "Close" not in data.columns:
    st.error("❌ 'Close' price data is missing. Unable to proceed.")
    st.stop()

# Prepare data for Prophet
df = data[["Date", "Close"]].copy()
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Ensure correct types
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df.dropna(subset=["ds", "y"], inplace=True)

if df.empty:
    st.error("❌ No valid data after cleaning. Try a different stock ticker.")
    st.stop()

# Train Prophet model
m = Prophet(daily_seasonality=True)
m.fit(df)

# Forecast
future = m.make_future_dataframe(periods=years * 365)
forecast = m.predict(future)

# Plot forecast
st.subheader("🔮 Stock Price Forecast")
fig1 = plot_plotly(m, forecast)
fig1.update_layout(
    title=f"{ticker} Price Prediction",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white"
)
st.plotly_chart(fig1)

# Custom chart: actual vs predicted
st.subheader("📉 Historical vs Predicted Prices")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Actual Price", line=dict(color="blue")))
fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicted Price", line=dict(color="orange")))
fig2.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig2)

st.caption("⚠️ This is a demo prediction model using Prophet. Do not use for financial decisions.")
