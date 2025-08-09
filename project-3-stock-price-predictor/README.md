# Stock Price Predictor (Time Series Forecasting)

A machine learning model that predicts future stock prices using **LSTM (Long Short-Term Memory)** networks.

## Features
- Fetches real-time stock market data
- Trains an LSTM model for price prediction
- Visualizes predictions vs. actual prices

## Tech Stack
- Python
- TensorFlow / Keras
- Pandas / NumPy
- Matplotlib
- yfinance API

## Structure
# Stock Price Predictor

This project uses Long Short-Term Memory (LSTM) neural networks to predict future stock prices based on historical data.  
It downloads data from Yahoo Finance, preprocesses it with normalization and windowing, then trains a sequential deep learning model.

**Key Features:**
- Automatic stock data retrieval using yfinance API
- Data scaling and time-series sequence creation for model input
- Two-layer LSTM network with dense output layers
- Model training with mean squared error loss
- Saves trained model for future predictions or deployment

**Technical Highlights:**
- Uses TensorFlow/Keras for deep learning implementation
- Demonstrates advanced time series forecasting
- Can be adapted for different stock tickers and data ranges

**Usage Instructions:**
- Install dependencies listed in README
- Run `main.py` to download data, train, and save the model
- Modify ticker and dates as needed for other stocks
## How to Run
```bash
git clone https://github.com/CurtisRaympi/ai-engineering-portfolio.git
cd ai-engineering-portfolio/project-3-stock-price-predictor/
pip install -r requirements.txt
python predictor.py
