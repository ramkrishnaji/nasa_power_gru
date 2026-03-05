import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("noise_robustness_eval_only.csv")

noise = df["noise_p"]

plt.figure(figsize=(8,5))

plt.plot(noise, df["test_mae"], marker="o", label="MAE")
plt.plot(noise, df["test_rmse"], marker="s", label="RMSE")

plt.xlabel("Depolarizing Noise Probability (p)")
plt.ylabel("Error")
plt.title("Hybrid Quantum GRU Robustness to Noise")
plt.legend()
plt.grid(True)

plt.savefig("combined_noise_metrics.png", dpi=300)
plt.show()