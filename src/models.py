import os
import joblib
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def evaluate(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2, model


def main():

    df = pd.read_csv("data/processed/AAPL_features.csv")

    X = df.drop(columns=["Date", "Target"])
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)

    train_idx, test_idx = list(tscv.split(X))[-1]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            ),

        "LightGBM":
            LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            )
    }

    os.makedirs("models", exist_ok=True)

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        mae, rmse, r2, trained = evaluate(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

        filename = (
            name.lower()
                .replace(" ", "_")
                + ".pkl"
        )

        joblib.dump(
            trained,
            f"models/{filename}"
        )

        results.append([name, mae, rmse, r2])

    results = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    )

    print("\n==========================")
    print(results)
    print("==========================")

    results.to_csv(
        "reports/model_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()
