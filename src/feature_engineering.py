import os
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def engineer_features(input_file, output_file):

    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"])

    # Moving Averages
    df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
    df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()

    # Exponential Moving Averages
    df["EMA_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()

    # RSI
    df["RSI"] = RSIIndicator(df["Close"]).rsi()

    # MACD
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()

    # Bollinger Bands
    bb = BollingerBands(df["Close"])

    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    # Daily Returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Rolling Statistics
    df["Rolling_Mean_5"] = df["Close"].rolling(window=5).mean()
    df["Rolling_Mean_10"] = df["Close"].rolling(window=10).mean()

    df["Rolling_STD_5"] = df["Close"].rolling(window=5).std()
    df["Rolling_STD_10"] = df["Close"].rolling(window=10).std()

    # Volatility
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    # Momentum Features
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_10"] = df["Close"].pct_change(10)
    df["Return_20"] = df["Close"].pct_change(20)

    # Lag Features
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)
    df["Lag_5"] = df["Close"].shift(5)
    df["Lag_10"] = df["Close"].shift(10)
    df["Lag_20"] = df["Close"].shift(20)

    # Classification Target
    # 1 = Tomorrow's return is positive
    # 0 = Tomorrow's return is zero or negative
    df["Target"] = (df["Daily_Return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(output_file, index=False)

    print(df.head())
    print("\nFeature Engineering Completed!")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    engineer_features(
        "data/raw/AAPL.csv",
        "data/processed/AAPL_features.csv"
    )
