import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(model_path, output_image, output_csv):

    # Load trained model
    model = joblib.load(model_path)

    # Load processed dataset
    df = pd.read_csv("data/processed/AAPL_features.csv")

    # Features
    X = df.drop(columns=["Date", "Target"])

    # Get feature importances
    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n===== Top 15 Features =====")
    print(feature_df.head(15))

    os.makedirs("reports", exist_ok=True)

    feature_df.to_csv(output_csv, index=False)

    # Plot Top 15
    plt.figure(figsize=(10, 7))

    top = feature_df.head(15)

    plt.barh(
        top["Feature"],
        top["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances")

    plt.tight_layout()

    plt.savefig(output_image)

    plt.close()

    print(f"\nSaved image to {output_image}")
    print(f"Saved csv to {output_csv}")


if __name__ == "__main__":

    plot_feature_importance(
        "models/random_forest_classifier.pkl",
        "reports/random_forest_feature_importance.png",
        "reports/random_forest_feature_importance.csv"
    )
