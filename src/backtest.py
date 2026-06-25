import joblib
import pandas as pd


def main():

    # Load model
    model = joblib.load(
        "models/random_forest_classifier.pkl"
    )

    # Load data
    df = pd.read_csv(
        "data/processed/AAPL_features.csv"
    )

    X = df.drop(columns=["Date", "Target"])

    # Predict
    df["Prediction"] = model.predict(X)

    # Strategy:
    # Buy only if prediction = 1

    df["Strategy_Return"] = (
        df["Prediction"] *
        df["Daily_Return"]
    )

    # Cumulative Returns

    df["Strategy_Cumulative"] = (
        1 + df["Strategy_Return"]
    ).cumprod()

    df["BuyHold_Cumulative"] = (
        1 + df["Daily_Return"]
    ).cumprod()

    # Win Rate

    trades = df[df["Prediction"] == 1]

    wins = (
        trades["Daily_Return"] > 0
    ).sum()

    total = len(trades)

    win_rate = wins / total if total else 0

    print("\n========== RESULTS ==========")

    print(
        f"Strategy Return : "
        f"{(df['Strategy_Cumulative'].iloc[-1]-1)*100:.2f}%"
    )

    print(
        f"Buy & Hold Return : "
        f"{(df['BuyHold_Cumulative'].iloc[-1]-1)*100:.2f}%"
    )

    print(
        f"Trades : {total}"
    )

    print(
        f"Win Rate : {win_rate*100:.2f}%"
    )

    df.to_csv(
        "reports/backtest_results.csv",
        index=False
    )

    print(
        "\nSaved backtest_results.csv"
    )


if __name__ == "__main__":
    main()
