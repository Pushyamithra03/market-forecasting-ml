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

    # Volatility
    df["Volatility"] = df["Daily_Return"].rolling(20).std()

    # Lag Features
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    # Target (next day's close)
    df["Target"] = df["Close"].shift(-1)

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
