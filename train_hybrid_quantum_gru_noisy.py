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
BATCH_SIZE = 4
EPOCHS = 5  # keep 5 for noisy runs (faster + enough for comparison)

NOISE_LEVELS = [0.00, 0.01, 0.03, 0.05, 0.10]

# =========================
# LOAD DATA
# =========================
data = np.load(DATA_FILE)

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("Dataset shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

print("\nCUDA available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
# TRAIN ONE MODEL WITH GIVEN NOISE
# =========================
def train_one_noise(noise_p):

    # Quantum device with noise
    dev = qml.device("default.mixed", wires=4)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        # Encode inputs
        for i in range(4):
            qml.RY(inputs[i], wires=i)

        # Add noise (depolarizing channel)
        for i in range(4):
            qml.DepolarizingChannel(noise_p, wires=i)

        # Variational circuit
        qml.templates.StronglyEntanglingLayers(weights, wires=range(4))

        # Add noise again after entangling
        for i in range(4):
            qml.DepolarizingChannel(noise_p, wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    weight_shapes = {"weights": (2, 4, 3)}
    qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    class HybridGRUQuantumNoisy(nn.Module):
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

            # quantum per-sample
            q_out = []
            for i in range(x.shape[0]):
                q_out.append(self.q_layer(x[i]))
            x = torch.stack(q_out)

            x = self.relu(self.fc2(x))
            y = self.out(x)
            return y

    model = HybridGRUQuantumNoisy(input_features=4, hidden_size=32).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_rmse = float("inf")
    best_state = None

    print(f"\n==============================")
    print(f"Training with noise p = {noise_p}")
    print(f"==============================")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_mae, val_rmse = evaluate(model, val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f} | Val RMSE: {val_rmse:.6f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    test_mae, test_rmse = evaluate(model, test_loader)

    print(f"\n✅ Final Test Results for noise p={noise_p}")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)

    return test_mae, test_rmse

# =========================
# RUN ALL NOISE LEVELS
# =========================
results = []

for p in NOISE_LEVELS:
    mae, rmse = train_one_noise(p)
    results.append({"noise_p": p, "test_mae": mae, "test_rmse": rmse})

df = pd.DataFrame(results)
print("\n📌 Summary Results:")
print(df)

df.to_csv("noisy_hybrid_results.csv", index=False)
print("\nSaved: noisy_hybrid_results.csv")
