import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="Project 3 - Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Price Prediction App")
st.markdown("""
Enter any **stock ticker symbol** (e.g., AAPL, MSFT, TSLA) to fetch recent data  
and predict future prices using **Linear Regression**.
---
""")

# ----------------------
# Example tickers (for user reference)
# ----------------------
st.sidebar.subheader("💡 Example Tickers")
st.sidebar.write("""
AAPL - Apple Inc.  
MSFT - Microsoft Corp.  
GOOGL - Alphabet Inc.  
AMZN - Amazon.com Inc.  
TSLA - Tesla Inc.  
META - Meta Platforms Inc.  
""")

# ----------------------
# User ticker input
# ----------------------
ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper().strip()

# Prediction period
period_days = st.slider("Prediction period (days):", 30, 365, 180)

# ----------------------
# Fetch and process stock data
# ----------------------
if ticker:
    try:
        stock_data = yf.download(ticker, period="1y", interval="1d")

        if stock_data.empty:
            st.error("No data found for this ticker. Please check the symbol.")
        else:
            stock_data.reset_index(inplace=True)
            stock_data = stock_data.rename(columns={"Date": "ds", "Close": "y"})
            df = stock_data[["ds", "y"]].dropna()

            # Prepare regression model
            df["day_number"] = (df["ds"] - df["ds"].min()).dt.days
            X = df[["day_number"]]
            y = df["y"]

            model = LinearRegression()
            model.fit(X, y)

            # Predict future prices
            last_day = df["day_number"].max()
            future_days = np.arange(last_day + 1, last_day + period_days + 1).reshape(-1, 1)
            future_dates = [df["ds"].max() + pd.Timedelta(days=i) for i in range(1, period_days + 1)]
            predictions = model.predict(future_days)

            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "y_pred": predictions
            })

            # ----------------------
            # Plot
            # ----------------------
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["y"],
                mode="lines+markers",
                name="Actual Prices",
                line=dict(color="blue")
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df["ds"], y=forecast_df["y_pred"],
                mode="lines+markers",
                name="Predicted Prices",
                line=dict(color="red", dash="dot")
            ))

            fig.update_layout(
                title=f"{ticker} Stock Price Prediction (Linear Regression)",
                title_x=0.5,
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            st.subheader("📄 Forecast Data")
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
