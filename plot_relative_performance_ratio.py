import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD NOISE RESULTS
# =========================

df = pd.read_csv("noise_robustness_eval_only.csv")

noise = df["noise_p"]
hybrid_mae = df["test_mae"]

# =========================
# GRU BASELINE ERROR
# =========================

gru_mae = 0.024366  # from your GRU baseline results

# =========================
# COMPUTE RATIO
# =========================

ratio = hybrid_mae / gru_mae

# =========================
# PLOT
# =========================

plt.figure(figsize=(8,5))

plt.plot(noise, ratio, marker="o")

plt.xlabel("Noise Probability (p)")
plt.ylabel("Hybrid Error / GRU Error")
plt.title("Relative Performance Degradation under Quantum Noise")

plt.grid(True)

plt.tight_layout()

plt.savefig("relative_performance_ratio.png", dpi=300)

plt.show()