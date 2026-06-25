import os
import joblib
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def evaluate(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return (
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred),
        roc_auc_score(y_test, prob),
        model
    )


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

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

        "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        ),

        "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )

    }

    os.makedirs("models", exist_ok=True)

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        accuracy, precision, recall, f1, roc, trained = evaluate(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc:.4f}")

        filename = (
            name.lower()
            .replace(" ", "_")
            + "_classifier.pkl"
        )

        joblib.dump(
            trained,
            f"models/{filename}"
        )

        results.append([
            name,
            accuracy,
            precision,
            recall,
            f1,
            roc
        ])

    results = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC"
        ]
    )

    os.makedirs("reports", exist_ok=True)
    results.to_csv(
        "reports/classification_results.csv",
        index=False
    )

    print("\n==============================")
    print(results)
    print("==============================")


if __name__ == "__main__":
    main()
