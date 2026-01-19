import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# 1) LOAD DATA
# =========================
DATA_FILE = "nasa_power_dataset_19.076_72.8777_20220101_20260110_win24_h1.npz"
data = np.load(DATA_FILE)

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("Original full dataset shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# =========================
# 2) FAST DEBUG MODE (SUBSET)
# =========================
# This makes training finish quickly on laptop CPU
DEBUG_TRAIN_SAMPLES = 2000
DEBUG_VAL_SAMPLES   = 400
DEBUG_TEST_SAMPLES  = 400

X_train = X_train[:DEBUG_TRAIN_SAMPLES]
y_train = y_train[:DEBUG_TRAIN_SAMPLES]

X_val = X_val[:DEBUG_VAL_SAMPLES]
y_val = y_val[:DEBUG_VAL_SAMPLES]

X_test = X_test[:DEBUG_TEST_SAMPLES]
y_test = y_test[:DEBUG_TEST_SAMPLES]

print("\nDEBUG subset shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# =========================
# 3) DATALOADERS (SMALL BATCH)
# =========================
BATCH_SIZE = 4  # IMPORTANT for speed + quantum stability

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 4) DEVICE
# =========================
print("\nCUDA available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 5) QUANTUM LAYER (VQC)
# =========================
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # inputs shape: (n_qubits,)
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# =========================
# 6) HYBRID MODEL
# =========================
class HybridGRUQuantum(nn.Module):
    def __init__(self, input_features=4, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_features, hidden_size=hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, n_qubits)
        self.q_layer = qlayer
        self.fc2 = nn.Linear(n_qubits, 16)
        self.out = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 24, features)
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]   # (batch, hidden)

        x = self.fc1(last)         # (batch, n_qubits)
        x = torch.tanh(x)          # keep bounded for quantum rotations

        # IMPORTANT FIX: quantum layer cannot handle batch directly here
        q_out = []
        for i in range(x.shape[0]):
            q_out.append(self.q_layer(x[i]))   # each output is (n_qubits,)
        x = torch.stack(q_out)                 # (batch, n_qubits)

        x = self.relu(self.fc2(x))
        y = self.out(x)
        return y

model = HybridGRUQuantum(input_features=X_train.shape[2], hidden_size=32).to(device)

# =========================
# 7) TRAINING SETUP
# =========================
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(loader):
    model.eval()
    preds = []
    trues = []
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
# 8) TRAIN LOOP (SHORT)
# =========================
EPOCHS = 3  # FAST debug run

best_val_rmse = float("inf")
best_state = None

print("\nTraining Hybrid Quantum GRU (NOISELESS) [FAST DEBUG MODE]...")

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

    val_mae, val_rmse = evaluate(val_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f} | Val RMSE: {val_rmse:.6f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = model.state_dict()

# Load best model
model.load_state_dict(best_state)

# =========================
# 9) TEST RESULTS
# =========================
test_mae, test_rmse = evaluate(test_loader)

print("\n✅ Hybrid Quantum GRU Results (NOISELESS) [DEBUG SUBSET]:")
print("MAE :", test_mae)
print("RMSE:", test_rmse)

torch.save(model.state_dict(), "hybrid_quantum_gru_noiseless_debug.pth")
print("\nSaved model: hybrid_quantum_gru_noiseless_debug.pth")
