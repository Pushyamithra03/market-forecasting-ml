import joblib
import os
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def evaluate(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    return mae, rmse, r2, model


def main():

    df = pd.read_csv("data/processed/AAPL_features.csv")

    X = df.drop(columns=["Date", "Target"])
    y = df["Target"]

    split = TimeSeriesSplit(n_splits=5)

    train_idx, test_idx = list(split.split(X))[-1]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    models = {

        "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42
        ),

        "XGBoost":
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),

        "LightGBM":
        LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42
        )
    }

    os.makedirs("models", exist_ok=True)

    results = []

    for name, model in models.items():

        print(f"\n{name}")

        mae, rmse, r2, trained = evaluate(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(f"MAE  : {mae:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R²   : {r2:.6f}")

        filename = (
            name.lower()
            .replace(" ", "_")
            + "_return.pkl"
        )

        joblib.dump(trained, f"models/{filename}")

        results.append([name, mae, rmse, r2])

    results = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "R2"]
    )

    results.to_csv(
        "reports/return_model_results.csv",
        index=False
    )

    print("\n==============================")
    print(results)
    print("==============================")


if __name__ == "__main__":
    main()
