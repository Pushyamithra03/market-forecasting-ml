import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("reports/backtest_results.csv")

plt.figure(figsize=(12,6))

plt.plot(
    df["Strategy_Cumulative"],
    label="Strategy"
)

plt.plot(
    df["BuyHold_Cumulative"],
    label="Buy & Hold"
)

plt.title("Backtest Performance")

plt.xlabel("Days")

plt.ylabel("Portfolio Value")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("reports/backtest_equity_curve.png")

plt.show()

print("Saved reports/backtest_equity_curve.png")
