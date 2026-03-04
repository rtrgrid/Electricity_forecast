import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows
from src.arima_model import run_arima
from src.lstm_model import LSTMModel
from src.transformer_model import TransformerModel

@st.cache_data(show_spinner=False)
def train_dl_models(X_train_np, y_train_np, X_test_np, y_test_np, _scaler, epochs, horizon, batch_size, input_size):

    device = torch.device("cpu")

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    # LSTM
    lstm_model = LSTMModel(
        input_size=input_size,
        hidden_size=64,
        horizon=horizon
    )

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    for _ in range(epochs):
        lstm_model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = lstm_model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(X_test)

    lstm_preds = _scaler.inverse_transform(
        lstm_preds.numpy().reshape(-1,1)
    ).reshape(lstm_preds.shape)

    y_true = _scaler.inverse_transform(
        y_test.numpy().reshape(-1,1)
    ).reshape(y_test.shape)

    lstm_rmse = np.sqrt(
        mean_squared_error(y_true.flatten(), lstm_preds.flatten())
    )

    # Transformer
    transformer_model = TransformerModel(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=horizon
    )

    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=5e-4)

    for _ in range(epochs):
        transformer_model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = transformer_model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    transformer_model.eval()
    with torch.no_grad():
        transformer_preds = transformer_model(X_test)

    transformer_preds = _scaler.inverse_transform(
        transformer_preds.numpy().reshape(-1,1)
    ).reshape(transformer_preds.shape)

    transformer_rmse = np.sqrt(
        mean_squared_error(y_true.flatten(), transformer_preds.flatten())
    )

    return lstm_preds, transformer_preds, lstm_rmse, transformer_rmse, y_true


st.set_page_config(layout="wide")
st.title("⚡ Electricity Forecasting Research Dashboard")

st.markdown("""
Interactive comparison of **ARIMA**, **LSTM**, and **Transformer** models  
using mini-batch training for stable learning.
""")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Experiment Controls")

context = st.sidebar.slider("Context Length", 12, 72, 24, step=12)
horizon = st.sidebar.slider("Prediction Length", 6, 24, 12, step=6)
epochs = st.sidebar.slider("Training Epochs", 30, 100, 50, step=10)

selected_models = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA", "LSTM", "Transformer"],
    default=["ARIMA", "LSTM", "Transformer"]
)

show_confidence = st.sidebar.checkbox("Show Transformer Confidence Interval", True)

batch_size = 32

# =========================
# LOAD DATA
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values
train, test = train_test_split(series)

if len(test) <= context + horizon:
    st.warning("Context + Horizon too large for dataset.")
    st.stop()

# =========================
# NAIVE BASELINE (for interpretation)
# =========================
naive_preds, y_true_naive = [], []

for i in range(len(test) - horizon):
    naive_preds.append(np.repeat(test[i], horizon))
    y_true_naive.append(test[i:i+horizon])

naive_preds = np.array(naive_preds)
y_true_naive = np.array(y_true_naive)

naive_rmse = np.sqrt(
    mean_squared_error(y_true_naive.flatten(), naive_preds.flatten())
)

# =========================
# ARIMA
# =========================
arima_rmse = None
if "ARIMA" in selected_models:
    arima_preds = run_arima(train, test, horizon=horizon)

    y_true_arima = np.array([
        test[i:i+horizon] for i in range(len(test) - horizon)
    ])

    arima_rmse = np.sqrt(
        mean_squared_error(y_true_arima.flatten(), arima_preds.flatten())
    )

# =========================
# SCALE DATA FOR DL
# =========================
train_scaled, test_scaled, scaler = scale_data(train, test)
train_dates = df.index[:len(train)]
test_dates = df.index[len(train):]

X_train, y_train = create_windows(train_scaled, train_dates, context, horizon)
X_test, y_test = create_windows(test_scaled, test_dates, context, horizon)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Safety squeeze
if X_train.dim() == 4:
    X_train = X_train.squeeze(-1)
    X_test = X_test.squeeze(-1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_fn = torch.nn.MSELoss()

lstm_rmse = None
transformer_rmse = None

# =========================
# TRAIN MODELS
# =========================
with st.spinner("Training selected models..."):

    lstm_preds = None
    transformer_preds = None

    if "LSTM" in selected_models or "Transformer" in selected_models:

        lstm_preds, transformer_preds, lstm_rmse, transformer_rmse, y_true = train_dl_models(
            X_train.cpu().numpy(),
            y_train.cpu().numpy(),
            X_test.cpu().numpy(),
            y_test.cpu().numpy(),
            scaler,
            epochs,
            horizon,
            batch_size,
            X_train.shape[-1]
        )

    if "LSTM" not in selected_models:
        lstm_rmse = None

    if "Transformer" not in selected_models:
        transformer_rmse = None
# =========================
# METRICS DISPLAY
# =========================
st.header("📊 RMSE Comparison")

metrics = {}

if arima_rmse is not None:
    metrics["ARIMA"] = arima_rmse
if lstm_rmse is not None:
    metrics["LSTM"] = lstm_rmse
if transformer_rmse is not None:
    metrics["Transformer"] = transformer_rmse

cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.2f}")

# =========================
# BAR CHART
# =========================
fig_bar, ax_bar = plt.subplots(figsize=(6,3))
bars = ax_bar.bar(metrics.keys(), metrics.values())
bars[np.argmin(list(metrics.values()))].set_color("green")
ax_bar.set_ylabel("RMSE")
ax_bar.grid(axis="y")
st.pyplot(fig_bar)

# =========================
# MODEL VALUE INTERPRETATION
# =========================
st.header("🧠 What Do These Model Values Mean?")

best_model = min(metrics, key=metrics.get)
best_rmse = metrics[best_model]

st.markdown(f"""
### 📊 Understanding RMSE Values

The numbers shown above represent **RMSE (Root Mean Square Error)**.

RMSE tells us:

> On average, how far the model's predictions are from actual electricity production values.

If electricity production values are around 100–130 units:

- RMSE = 10 → Model is off by ~10 units on average  
- RMSE = 5 → Model is off by ~5 units  
- RMSE = 3 → Model is very accurate (off by ~3 units)

Lower RMSE = Better forecasting accuracy.

---

### 🏆 What Your Current Results Say

Best Performing Model: **{best_model}**  
Best RMSE: **{best_rmse:.2f}**

This means:

- The {best_model} model makes the smallest average prediction error.
- It captures seasonal patterns and trends better than others.
- It generalizes better on unseen test data.

---

### 📉 How to Read the Bar Chart

- Each bar represents average prediction error.
- Shorter bar → More accurate model.
- Large gaps between bars → Significant performance difference.

If deep learning models (LSTM / Transformer) show lower RMSE than ARIMA:

→ They are capturing nonlinear patterns better.

If ARIMA performs best:

→ The dataset is mostly linear and seasonal.

---

### 🔍 What Happens When You Change Sliders?

**Context Length**
- Larger context → Model sees more historical data.
- Helps capture longer seasonal cycles.

**Prediction Horizon**
- Larger horizon → Harder forecasting.
- RMSE usually increases.

**Epochs**
- More epochs → Better training (until overfitting).
- Too few epochs → Underfitting.

---

### 📈 Practical Meaning

If Transformer RMSE = 3.7:

It means predictions are, on average, only 3.7 units away from actual production values.

That is strong forecasting accuracy.
""")

# =========================
# FORECAST VISUALIZATION
# =========================
st.header("📈 Forecast Example")

# idx = st.slider("Backtest Window", 0, len(y_true_naive)-1, 0)# determine safe max index for all models
max_windows = min(
    len(y_true_naive),
    len(arima_preds) if "ARIMA" in selected_models else len(y_true_naive),
    len(lstm_preds) if "LSTM" in selected_models and lstm_rmse is not None else len(y_true_naive),
    len(transformer_preds) if "Transformer" in selected_models and transformer_rmse is not None else len(y_true_naive)
)

idx = st.slider("Backtest Window", 0, max_windows - 1, 0)
dates = df.index[-len(test):][idx:idx+horizon]

fig, ax = plt.subplots(figsize=(9,4))
ax.plot(dates, y_true_naive[idx], label="Actual", linewidth=2)

if "ARIMA" in selected_models:
    ax.plot(dates, arima_preds[idx], linestyle=":", label="ARIMA")

if "LSTM" in selected_models:
    ax.plot(dates, lstm_preds[idx], linestyle="--", label="LSTM")

if "Transformer" in selected_models:
    ax.plot(dates, transformer_preds[idx], linestyle="--", label="Transformer")

    if show_confidence:
        residuals = y_true.flatten() - transformer_preds.flatten()
        std = np.std(residuals)
        lower = transformer_preds[idx] - 1.96 * std
        upper = transformer_preds[idx] + 1.96 * std
        ax.fill_between(dates, lower, upper, alpha=0.2)

ax.legend()
ax.grid(True)
st.pyplot(fig)

# =========================
# MATHEMATICAL RMSE EXPLANATION
# =========================
st.header("📐 Evaluation Metric")

st.latex(r"""
RMSE = \sqrt{\frac{1}{n} \sum (y_{actual} - y_{predicted})^2}
""")

st.markdown("""
RMSE (Root Mean Square Error) measures the average magnitude of prediction errors.

• It squares errors (so large mistakes matter more)  
• It averages them  
• Then takes the square root  

Lower RMSE = Better forecasting performance.
""")
# =========================
# RESIDUAL ANALYSIS
# =========================
if "Transformer" in selected_models and transformer_rmse is not None:
    st.header("📉 Transformer Residual Analysis")

    residuals = y_true.flatten() - transformer_preds.flatten()

    fig_res, ax_res = plt.subplots(figsize=(9,3))
    ax_res.plot(residuals)
    ax_res.axhline(0, linestyle="--")
    ax_res.grid(True)
    st.pyplot(fig_res)

# =========================
# DOWNLOAD RESULTS
# =========================
results_df = pd.DataFrame(metrics.items(), columns=["Model", "RMSE"])
st.download_button(
    "Download RMSE Results",
    results_df.to_csv(index=False),
    "model_results.csv"
)