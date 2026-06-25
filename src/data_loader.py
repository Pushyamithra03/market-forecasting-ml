import os
import yfinance as yf


def download_stock_data(ticker, start_date, end_date, save_path):

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    # Flatten columns if yfinance returns a MultiIndex
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_csv(save_path, index=False)

    print(data.head())


if __name__ == "__main__":
    download_stock_data(
        ticker="AAPL",
        start_date="2015-01-01",
        end_date="2025-01-01",
        save_path="data/raw/AAPL.csv"
    )
