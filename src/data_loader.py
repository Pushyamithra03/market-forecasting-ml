import os
import yfinance as yf


def download_stock_data(ticker, start_date, end_date, save_path):
    """
    Downloads historical stock data from Yahoo Finance
    and saves it as a CSV file.
    """

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_csv(save_path)

    print(f"Downloaded {len(data)} rows")
    print(f"Saved to {save_path}")
    print(data.head())

    return data


if __name__ == "__main__":
    download_stock_data(
        ticker="AAPL",
        start_date="2015-01-01",
        end_date="2025-01-01",
        save_path="data/raw/AAPL.csv"
    )
