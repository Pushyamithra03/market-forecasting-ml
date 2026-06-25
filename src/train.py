import joblib
import os

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def train_model():

    df = pd.read_csv("data/processed/AAPL_features.csv")

    # Remove Date column
    df = df.drop(columns=["Date"])

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=5)

    train_index, test_index = list(tscv.split(X))[-1]

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n===== Model Performance =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/linear_regression.pkl")

    print("\nModel saved successfully!")


if __name__ == "__main__":
    train_model()
