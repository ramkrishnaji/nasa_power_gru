import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

data = np.load("nasa_power_dataset_19.076_72.8777_20220101_20260110_win24_h1.npz")

X_test = data["X_test"]
y_test = data["y_test"]

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

# =========================
# QUANTUM DEVICE
# =========================

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):

    for i in range(4):
        qml.RY(inputs[i], wires=i)

    qml.templates.StronglyEntanglingLayers(weights, wires=range(4))

    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


weight_shapes = {"weights": (2, 4, 3)}

# =========================
# HYBRID MODEL
# =========================

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(4, 32, batch_first=True)

        self.fc1 = nn.Linear(32, 4)

        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.fc2 = nn.Linear(4, 16)

        self.out = nn.Linear(16, 1)

    def forward(self, x):

        out, _ = self.gru(x)

        out = out[:, -1, :]

        out = torch.tanh(self.fc1(out))

        q_out = []

        for i in range(out.shape[0]):
            q_out.append(self.q_layer(out[i]))

        q_out = torch.stack(q_out)

        q_out = torch.relu(self.fc2(q_out))

        return self.out(q_out)

# =========================
# LOAD TRAINED MODEL
# =========================

hybrid_model = HybridModel().to(device)

hybrid_model.load_state_dict(
    torch.load("hybrid_quantum_gru_noiseless_full.pth", map_location=device)
)

hybrid_model.eval()

# =========================
# PREDICTIONS
# =========================

with torch.no_grad():

    hybrid_preds = hybrid_model(X_test).cpu().numpy().flatten()

# =========================
# SAVE
# =========================

np.save("pred_hybrid_no_noise.npy", hybrid_preds)

print("Saved:")
print("pred_hybrid_no_noise.npy")