import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =========================
# 1) SETTINGS (EDIT CITY HERE)
# =========================
LAT = 19.0760      # Mumbai
LON = 72.8777

START_DATE = "20220101"   # 2022-01-01
END_DATE   = "20260110"   # 2026-01-10

WINDOW_HOURS = 24         # past 24 hours input
HORIZON = 1               # predict t+1 hour

PARAMS = [
    "ALLSKY_SFC_SW_DWN",  # Solar irradiance (main)
    "T2M",                # Temperature at 2m
    "RH2M",               # Relative humidity at 2m
    "WS2M"                # Wind speed at 2m
]

# =========================
# 2) DOWNLOAD FROM NASA POWER (JSON)
# =========================
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
query = f"?parameters={','.join(PARAMS)}"
query += "&community=RE"
query += f"&longitude={LON}&latitude={LAT}"
query += f"&start={START_DATE}&end={END_DATE}"
query += "&format=JSON"

url = BASE_URL + query
print("\n[1/7] Downloading NASA POWER hourly data...")
print("URL:", url)

res = requests.get(url)
res.raise_for_status()
data = res.json()

# =========================
# 3) PARSE JSON -> DATAFRAME
# =========================
print("\n[2/7] Parsing JSON to DataFrame...")
params_data = data["properties"]["parameter"]

df = None
for p in PARAMS:
    s = pd.Series(params_data[p], name=p)
    if df is None:
        df = s.to_frame()
    else:
        df = df.join(s, how="outer")

df.reset_index(inplace=True)
df.rename(columns={"index": "DATE_TIME"}, inplace=True)

# Convert NASA timestamp "YYYYMMDDHH" to datetime
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%Y%m%d%H")
df = df.sort_values("DATE_TIME").reset_index(drop=True)

print("Raw shape:", df.shape)
print(df.head(3))

# =========================
# 4) CLEANING
# =========================
print("\n[3/7] Cleaning missing values...")

# NASA uses -999 for missing
df.replace(-999, np.nan, inplace=True)

# Drop missing rows (simple + safe)
df = df.dropna().reset_index(drop=True)

# Remove negative irradiance (should not happen)
df = df[df["ALLSKY_SFC_SW_DWN"] >= 0].reset_index(drop=True)

print("Cleaned shape:", df.shape)

# =========================
# 5) SAVE CLEAN RAW CSV
# =========================
raw_csv_name = f"nasa_power_hourly_RAW_{LAT}_{LON}_{START_DATE}_{END_DATE}.csv"
df.to_csv(raw_csv_name, index=False)
print("Saved raw cleaned CSV:", raw_csv_name)

# =========================
# 6) SCALE FEATURES (MinMax)
# =========================
print("\n[4/7] Scaling features...")

feature_cols = PARAMS
target_col = "ALLSKY_SFC_SW_DWN"  # (we'll forecast irradiance at t+1)

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

scaled_csv_name = f"nasa_power_hourly_SCALED_{LAT}_{LON}_{START_DATE}_{END_DATE}.csv"
df_scaled.to_csv(scaled_csv_name, index=False)
print("Saved scaled CSV:", scaled_csv_name)

# =========================
# 7) CREATE WINDOWED DATASET (X,y)
# =========================
print("\n[5/7] Creating time-series windows...")

values = df_scaled[feature_cols].values.astype(np.float32)
target = df_scaled[target_col].values.astype(np.float32)

X_list, y_list = [], []

for i in range(WINDOW_HOURS, len(df_scaled) - HORIZON):
    X_window = values[i - WINDOW_HOURS:i]   # shape: (24, features)
    y_value = target[i + HORIZON]           # t+1
    X_list.append(X_window)
    y_list.append(y_value)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print("X shape:", X.shape)  # (samples, 24, features)
print("y shape:", y.shape)  # (samples,)

# =========================
# 8) TRAIN/VAL/TEST SPLIT (chronological)
# =========================
print("\n[6/7] Creating chronological splits (70/15/15)...")

n = len(X)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# =========================
# 9) SAVE NPZ (READY FOR ML)
# =========================
print("\n[7/7] Saving dataset to NPZ...")

npz_name = f"nasa_power_dataset_{LAT}_{LON}_{START_DATE}_{END_DATE}_win{WINDOW_HOURS}_h{HORIZON}.npz"
np.savez_compressed(
    npz_name,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

print("\nDONE ✅")
print("Files created:")
print("1)", raw_csv_name)
print("2)", scaled_csv_name)
print("3)", npz_name)
print("\nNext: train LSTM/GRU baselines + Hybrid Quantum model on X_train/y_train.")
