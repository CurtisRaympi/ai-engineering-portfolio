import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Price Explorer")

st.markdown("""
This app lets you search for any stock by ticker symbol, view historical price trends,
and see a simple moving average prediction.
""")

# Default ticker
default_ticker = "AAPL"

# Search box with autocomplete-like behavior
ticker_input = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, TSLA):", default_ticker).upper()

# Date range selector
start_date = st.date_input("Start date", date.today() - timedelta(days=365))
end_date = st.date_input("End date", date.today())

if st.button("Fetch Data"):
    try:
        stock = yf.Ticker(ticker_input)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            st.error("No data found for this ticker and date range.")
        else:
            # Display basic info
            info = stock.info
            st.subheader(f"{info.get('shortName', ticker_input)} ({ticker_input})")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")

            # Plot price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(window=20).mean(),
                                     name="20-Day SMA", line=dict(color="orange")))
            fig.update_layout(title=f"{ticker_input} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            # Simple prediction (next 7 days using last close price trend)
            last_price = df["Close"].iloc[-1]
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
            predictions = [last_price * (1 + 0.001 * i) for i in range(1, 8)]
            pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
            st.subheader("📅 Next 7-Day Price Prediction (Simple Trend)")
            st.table(pred_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
