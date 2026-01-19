import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# =========================
# CONFIG
# =========================
DATA_FILE = "nasa_power_dataset_19.076_72.8777_20220101_20260110_win24_h1.npz"
MODEL_FILE = "hybrid_quantum_gru_noiseless_full.pth"  # your saved model

BATCH_SIZE = 4

NOISE_LEVELS = [0.00, 0.01, 0.03, 0.05, 0.10]

# =========================
# DEVICE
# =========================
print("CUDA available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATA
# =========================
data = np.load(DATA_FILE)

X_test = data["X_test"]
y_test = data["y_test"]

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

print("\nTest shape:", X_test.shape, y_test.shape)

# =========================
# EVAL FUNCTION
# =========================
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            yhat = model(xb)

            preds.append(yhat.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds = np.vstack(preds).flatten()
    trues = np.vstack(trues).flatten()

    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    return mae, rmse

# =========================
# BUILD MODEL WITH NOISE
# (same architecture, but quantum layer is noisy)
# =========================
def build_noisy_model(noise_p):

    dev = qml.device("default.mixed", wires=4)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        # input encoding
        for i in range(4):
            qml.RY(inputs[i], wires=i)

        # noise
        for i in range(4):
            qml.DepolarizingChannel(noise_p, wires=i)

        # trainable circuit
        qml.templates.StronglyEntanglingLayers(weights, wires=range(4))

        # noise again
        for i in range(4):
            qml.DepolarizingChannel(noise_p, wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    weight_shapes = {"weights": (2, 4, 3)}
    qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    class HybridGRUQuantum(nn.Module):
        def __init__(self, input_features=4, hidden_size=32):
            super().__init__()
            self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True)

            self.fc1 = nn.Linear(hidden_size, 4)
            self.q_layer = qlayer
            self.fc2 = nn.Linear(4, 16)
            self.out = nn.Linear(16, 1)

            self.relu = nn.ReLU()

        def forward(self, x):
            gru_out, _ = self.gru(x)
            last = gru_out[:, -1, :]

            x = self.fc1(last)
            x = torch.tanh(x)

            q_out = []
            for i in range(x.shape[0]):
                q_out.append(self.q_layer(x[i]))
            x = torch.stack(q_out)

            x = self.relu(self.fc2(x))
            y = self.out(x)
            return y

    return HybridGRUQuantum(input_features=4, hidden_size=32)

# =========================
# LOAD TRAINED WEIGHTS INTO NOISY MODEL
# =========================
print("\nLoading trained weights from:", MODEL_FILE)

results = []

for p in NOISE_LEVELS:
    print("\n==============================")
    print(f"Evaluating with noise p = {p}")
    print("==============================")

    model = build_noisy_model(p).to(device)

    # Load weights (works because architecture matches)
    state = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state)

    test_mae, test_rmse = evaluate(model, test_loader)

    print("✅ Test MAE :", test_mae)
    print("✅ Test RMSE:", test_rmse)

    results.append({"noise_p": p, "test_mae": test_mae, "test_rmse": test_rmse})

df = pd.DataFrame(results)
print("\n📌 Final Noise Robustness Table:")
print(df)

df.to_csv("noise_robustness_eval_only.csv", index=False)
print("\nSaved: noise_robustness_eval_only.csv")
