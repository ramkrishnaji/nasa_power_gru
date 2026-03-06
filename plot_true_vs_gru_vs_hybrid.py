import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

data = np.load("nasa_power_dataset_19.076_72.8777_20220101_20260110_win24_h1.npz")

y_test = data["y_test"]

# =========================
# LOAD PREDICTIONS
# =========================

gru_preds = np.load("pred_gru.npy")
hybrid_preds = np.load("pred_hybrid_no_noise.npy")

# =========================
# PLOT (only first 200 points for clarity)
# =========================

n = 200

plt.figure(figsize=(10,5))

plt.plot(y_test[:n], label="True GHI", linewidth=2)
plt.plot(gru_preds[:n], label="GRU Baseline", linestyle="--")
plt.plot(hybrid_preds[:n], label="Hybrid Quantum GRU", linestyle=":")

plt.xlabel("Time Step")
plt.ylabel("Normalized GHI")
plt.title("Comparison of True GHI vs GRU and Hybrid Predictions")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("true_vs_gru_vs_hybrid.png", dpi=300)

plt.show()