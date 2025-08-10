import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
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
This app predicts future stock prices using **Facebook Prophet**.  
Upload a CSV with columns `ds` (date) and `y` (price).  
---
""")

# ----------------------
# File Upload
# ----------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if "ds" not in df.columns or "y" not in df.columns:
        st.error("CSV must contain 'ds' (date) and 'y' (price) columns.")
    else:
        # Convert 'ds' to datetime
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

        # Ensure 'y' is numeric
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        # Drop rows with invalid values
        df.dropna(subset=["ds", "y"], inplace=True)

        if df.empty:
            st.error("No valid data after cleaning. Please check your CSV.")
        else:
            # ----------------------
            # User settings
            # ----------------------
            period = st.slider("Prediction period (days):", 30, 365, 180)

            # ----------------------
            # Fit Model
            # ----------------------
            m = Prophet()
            m.fit(df)

            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # ----------------------
            # Forecast Plot
            # ----------------------
            st.subheader("📊 Forecast Plot")
            fig1 = plot_plotly(m, forecast)
            fig1.update_layout(
                title="Stock Price Forecast",
                title_x=0.5,
                title_font=dict(size=22),
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # ----------------------
            # Components Plot
            # ----------------------
            st.subheader("📈 Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

            # ----------------------
            # Data Preview
            # ----------------------
            st.subheader("📄 Forecast Data")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

