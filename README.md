# 📈 Market Forecasting using Machine Learning

## Overview

This project implements an end-to-end machine learning pipeline for stock market forecasting using historical Apple (AAPL) stock data from Yahoo Finance.

The project includes:

- Data Collection
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Price Forecasting
- Return Forecasting
- Classification of Market Direction
- Feature Importance Analysis
- Strategy Backtesting

---

## Project Structure

```
market-forecasting-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Dataset

Source:

- Yahoo Finance

Ticker:

- AAPL

Time Period:

- 2015–2025

---

## Feature Engineering

Technical indicators used:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD
- Bollinger Bands
- Daily Returns
- Rolling Mean
- Rolling Standard Deviation
- Volatility
- Momentum Features
- Lag Features

---

## Machine Learning Models

### Regression

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### Classification

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

---

## Results

### Price Forecasting

| Model | R² |
|------|------|
| Linear Regression | 0.9647 |

---

### Classification

| Model | Accuracy |
|------|----------|
| Logistic Regression | 56.20% |
| Random Forest | 44.28% |
| XGBoost | 44.77% |
| LightGBM | 44.28% |

---

### Backtesting

| Metric | Value |
|--------|-------|
| Strategy Return | -12.07% |
| Buy & Hold Return | 806.29% |
| Win Rate | 50.65% |

---

## Key Findings

- Linear Regression performed well for price prediction due to strong autocorrelation in stock prices.
- Predicting next-day returns and market direction using only technical indicators proved significantly more challenging.
- Technical indicators alone were insufficient for building a profitable trading strategy over the tested period.
- Future improvements could incorporate news sentiment, macroeconomic indicators, options data, and alternative data sources.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- yfinance
- TA Library

---

## Future Work

- Hyperparameter Optimization
- LSTM Models
- Transformer-based Time Series Models
- Sentiment Analysis
- Portfolio Optimization
- Reinforcement Learning for Trading

---

## Author

Pushyamithra

