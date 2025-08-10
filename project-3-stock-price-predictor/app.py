# project-3-stock-price-predictor/app.py
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

st.set_page_config(page_title="📈 Stock Forecast (Prophet)", layout="wide")
st.title("📈 Stock Price Forecast (Prophet)")
st.markdown("Interactive forecast demo using Yahoo Finance + Prophet. Fast, interpretable, and deployable.")

# Sidebar controls
st.sidebar.header("Forecast Settings")
symbol = st.sidebar.text_input("Ticker symbol", value="AAPL")
period = st.sidebar.selectbox("Historical period to download", ["1y","2y","3y","5y","10y"], index=2)
horizon_days = st.sidebar.slider("Days to forecast", 7, 180, 30)

# Fetch data
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    df = yf.download(ticker, period=period)
    df = df.reset_index()
    df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    return df

data_load_state = st.info(f"Downloading {symbol} data...")
try:
    df = load_data(symbol, period)
    data_load_state.success("Data loaded ✅")
except Exception as e:
    data_load_state.error(f"Error downloading data: {e}")
    st.stop()

st.subheader(f"Historical Close Prices: {symbol.upper()}")
st.dataframe(df.tail(10))

# Plot historical close
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Close', line=dict(color='royalblue')))
fig_hist.update_layout(title=f"{symbol.upper()} - Historical Close", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig_hist, use_container_width=True)

# Prepare & fit Prophet (fast)
with st.spinner("Fitting Prophet model..."):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)

# Create future dataframe and predict
future = m.make_future_dataframe(periods=horizon_days)
forecast = m.predict(future)

# Show forecast table & plot
st.subheader("Forecast")
st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon_days))

fig_forecast = plot_plotly(m, forecast)
fig_forecast.update_layout(title=f"{symbol.upper()} - Forecast (next {horizon_days} days)")
st.plotly_chart(fig_forecast, use_container_width=True)

# Show components
st.subheader("Forecast Components (trend & seasonality)")
fig_comp = plot_components_plotly(m, forecast)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.markdown("Notes: Prophet is great for quick, interpretable forecasts. Replace with LSTM later if you need deep models.")
