import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("noise_robustness_eval_only.csv")

gru_mae = 0.024366

ratio = df["test_mae"] / gru_mae

plt.figure(figsize=(8,5))

plt.plot(df["noise_p"], ratio, marker="o")

plt.xlabel("Noise Probability (p)")
plt.ylabel("Hybrid Error / GRU Error")
plt.title("Relative Performance Degradation under Quantum Noise")
plt.grid(True)

plt.savefig("relative_performance_ratio.png", dpi=300)
plt.show()