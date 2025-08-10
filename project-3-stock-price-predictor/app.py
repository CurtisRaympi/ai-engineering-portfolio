import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 🎯 Page Config
st.set_page_config(page_title="📈 Stock Price Predictor", page_icon="📊", layout="wide")

# 🎨 Custom CSS for better look
st.markdown("""
    <style>
    body {
        color: #000000;
        background-color: #F5F5F5;
    }
    .stTextInput>div>div>input {
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# 🏷️ App Title
st.title("📈 Stock Price Predictor")
st.write("Predict future stock prices using LSTM Neural Networks.")

# 📝 User input
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")
n_days = st.slider("Days to Predict:", min_value=5, max_value=60, value=30)

# 📥 Download Stock Data
st.subheader("Historical Stock Data")
try:
    data = yf.download(stock_symbol, start="2015-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if data.empty:
    st.error("No data found for this symbol. Please check and try again.")
    st.stop()

st.dataframe(data.tail())

# 📊 Plot Historical Data
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Close'], label='Closing Price', color='blue')
ax.set_title(f"{stock_symbol} Closing Price History", fontsize=16)
ax.set_xlabel("Date")
ax.set_ylabel("Price USD")
ax.legend()
st.pyplot(fig)

# 📦 Prepare Data for LSTM
df = data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 🧠 Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

with st.spinner("Training the LSTM model... ⏳"):
    model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

# 📈 Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# 📅 Predict Future
last_60_days = scaled_data[-60:]
future_input = last_60_days.reshape(1, -1)
temp_input = list(future_input[0])

future_predictions = []
for _ in range(n_days):
    x_input = np.array(temp_input[-60:]).reshape(1, 60, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    future_predictions.append(yhat[0][0])

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 📊 Plot Future Predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_days)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data['Close'], label='Historical Price', color='blue')
ax2.plot(future_dates, future_predictions, label='Predicted Price', color='red')
ax2.set_title(f"{stock_symbol} Price Prediction", fontsize=16)
ax2.set_xlabel("Date")
ax2.set_ylabel("Price USD")
ax2.legend()
st.pyplot(fig2)

# ✅ Show Predictions Table
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
st.subheader("📅 Predicted Prices")
st.dataframe(pred_df)
