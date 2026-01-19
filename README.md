# 🌞 Solar Power Forecasting using Hybrid Quantum GRU (Noise Robustness Study)

This project implements **short-term solar power forecasting** using:

- ✅ **Classical GRU baseline**
- ✅ **Hybrid Quantum GRU (PennyLane + PyTorch)**
- ✅ **Quantum noise robustness evaluation** using a **mixed-state simulator**
- 📊 Plots: **Noise vs MAE / RMSE**

The goal is to analyze whether hybrid quantum models remain reliable under **realistic quantum noise**, which is often ignored in most forecasting papers.

---

## 📌 Problem Statement
Predict **next-hour solar power / irradiance** (t+1 hour) using the **previous 24 hours** of features.

---

## 📂 Dataset
Dataset is generated from **NASA POWER (hourly)** weather + solar parameters.

- Window size: **24 hours**
- Forecast horizon: **1 hour ahead**
- Dataset range used: **2022-01-01 → 2026-01-10**
- Saved format: `.npz` with train/val/test splits

---

## 🧠 Models Implemented

### 1) Classical GRU Baseline
A standard GRU model trained on the dataset.

📌 Script:
- `train_gru_baseline.py`

Output:
- MAE, RMSE on test set

---

### 2) Hybrid Quantum GRU (Noiseless)
A hybrid model where the classical GRU output is passed into a **quantum layer (PennyLane QNode)** and then mapped to prediction.

📌 Script:
- `train_hybrid_quantum_gru_full.py`

Output:
- Trained model file:
  - `hybrid_quantum_gru_noiseless_full.pth`
- MAE, RMSE on test set

---

### 3) Noise Robustness Evaluation (Main Contribution)
This evaluates the trained hybrid quantum model under **depolarizing quantum noise** using `default.mixed`.

Noise levels tested:
`p = [0.0, 0.01, 0.03, 0.05, 0.10]`

📌 Script:
- `eval_noisy_hybrid_from_trained.py`

Output:
- `noise_robustness_eval_only.csv`

---

## 📊 Plots
Noise sensitivity plots:

- `noise_vs_mae.png`
- `noise_vs_rmse.png`

Generate them using:

📌 Script:
- `plot_noise_curve.py`

---

## 📁 Project Files
| File | Purpose |
|------|---------|
| `make_nasa_power_dataset_2022_2026.py` | Downloads NASA POWER data + creates dataset windows |
| `train_gru_baseline.py` | Trains classical GRU baseline |
| `train_hybrid_quantum_gru_full.py` | Trains hybrid quantum GRU on full dataset |
| `eval_noisy_hybrid_from_trained.py` | Evaluates trained model under quantum noise (fast, no retraining) |
| `plot_noise_curve.py` | Generates plots from CSV results |
| `noise_robustness_eval_only.csv` | Final noise robustness results |
| `noise_vs_mae.png` | Noise vs MAE plot |
| `noise_vs_rmse.png` | Noise vs RMSE plot |
| `hybrid_quantum_gru_noiseless_full.pth` | Saved trained hybrid model weights |

---
### 1) Create environment (recommended)
conda create -n qml_gpu python=3.12 -y
conda activate qml_gpu
### 2) Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pennylane pandas scikit-learn matplotlib



🚀 How to Run (Step-by-step)

Step 1: Create dataset
python make_nasa_power_dataset_2022_2026.py

Step 2: Train GRU baseline
python train_gru_baseline.py

Step 3: Train Hybrid Quantum GRU (Noiseless)
python train_hybrid_quantum_gru_full.py

Step 4: Evaluate Hybrid model under quantum noise
python eval_noisy_hybrid_from_trained.py

Step 5: Plot Noise Robustness Curves
python plot_noise_curve.py
📌 Conclusion: Hybrid quantum forecasting pipelines are highly noise-sensitive, making robustness analysis critical for real-world deployment.

📌 Research Contribution / Novelty

Most hybrid quantum forecasting papers evaluate only under noiseless simulators.

This project adds:
✅ systematic noise robustness evaluation
✅ noise vs error curves (MAE/RMSE)
✅ realistic deployment-oriented insight

📎 Notes

Quantum simulation can be slow depending on system specs.

GPU helps for PyTorch operations, but quantum simulation is still the main bottleneck.
conda create -n qml_gpu python=3.12 -y
conda activate qml_gpu
