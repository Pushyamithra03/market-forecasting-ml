import os
import pandas as pd
import matplotlib.pyplot as plt


def perform_eda(file_path):

    # Create reports directory
    os.makedirs("reports", exist_ok=True)

    # Load dataset
    df = pd.read_csv(file_path)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATASET INFO ==========")
    df.info()

    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Closing Price Plot
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Close"])
    plt.title("Apple Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/closing_price.png")
    plt.close()

    # Trading Volume Plot
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Volume"])
    plt.title("Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/volume.png")
    plt.close()

    # Daily Returns
    df["Daily Return"] = df["Close"].pct_change()

    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Daily Return"])
    plt.title("Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/daily_returns.png")
    plt.close()

    print("\nEDA completed successfully!")
    print("Plots saved in reports/")


if __name__ == "__main__":
    perform_eda("data/raw/AAPL.csv")
