import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("noise_robustness_eval_only.csv")

plt.figure()
plt.plot(df["noise_p"], df["test_rmse"], marker="o")
plt.xlabel("Depolarizing noise probability (p)")
plt.ylabel("Test RMSE")
plt.title("Noise Robustness of Hybrid Quantum GRU")
plt.grid(True)
plt.savefig("noise_vs_rmse.png", dpi=300)
plt.show()

plt.figure()
plt.plot(df["noise_p"], df["test_mae"], marker="o")
plt.xlabel("Depolarizing noise probability (p)")
plt.ylabel("Test MAE")
plt.title("Noise Robustness of Hybrid Quantum GRU")
plt.grid(True)
plt.savefig("noise_vs_mae.png", dpi=300)
plt.show()

print("Saved: noise_vs_rmse.png and noise_vs_mae.png")
