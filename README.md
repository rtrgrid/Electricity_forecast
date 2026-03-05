# ⚡ Electricity Forecasting with ARIMA, LSTM, and Transformer

This project builds and compares multiple **time-series forecasting
models** to predict electricity production.\
The goal is to evaluate how traditional statistical models compare with
modern deep learning models.

An interactive dashboard is provided using **Streamlit** to visualize
predictions and model performance.

------------------------------------------------------------------------

# 🌐 Live Dashboard

Streamlit App:\
https://electricityforecast-cwdb6vgjhljudsnxn5z9kn.streamlit.app/

The dashboard allows users to:

-   Compare model performance
-   Visualize predictions vs actual values
-   Explore forecasting parameters
-   Analyze residual errors

------------------------------------------------------------------------

# 📊 Problem Statement

Electricity production follows **time-dependent patterns** such as
trends and seasonality.

The objective of this project is to:

1.  Train multiple forecasting models on electricity production data\
2.  Compare their prediction accuracy\
3.  Visualize model predictions and performance

------------------------------------------------------------------------

# 📁 Dataset

Dataset: **Electric Production Dataset**

It contains monthly electricity production values.

  Column   Description
  -------- ------------------------------
  Date     Timestamp
  Value    Electricity production index

Total observations:

397 time steps

------------------------------------------------------------------------

# 🧠 Models Implemented

## Naive Baseline

The naive model predicts future values using the **last observed
value**.

Prediction = Last value in the time series

This model acts as a **baseline benchmark**.

------------------------------------------------------------------------

## ARIMA

ARIMA stands for **AutoRegressive Integrated Moving Average**.

It models the time series using:

-   past values
-   differencing
-   past forecast errors

ARIMA works well when the data has **clear linear patterns and
seasonality**.

------------------------------------------------------------------------

## LSTM

LSTM (Long Short-Term Memory) is a **recurrent neural network designed
for sequence data**.

It learns patterns from previous time steps and captures:

-   nonlinear relationships
-   temporal dependencies
-   long-term patterns

------------------------------------------------------------------------

## Transformer

Transformers use a **self-attention mechanism** to model relationships
between time steps.

Unlike LSTM, transformers analyze **all time steps simultaneously**
instead of sequentially.

This allows the model to capture **global temporal patterns more
effectively**.

------------------------------------------------------------------------

# ⚙️ Forecasting Approach

The project uses a **sliding window forecasting technique**.

Example:

Context Length = 24\
Prediction Horizon = 12

Meaning:

Past 24 time steps → used to predict the next 12 steps

Pipeline:

Raw Time Series\
↓\
Train/Test Split\
↓\
Scaling\
↓\
Sliding Window Creation\
↓\
Model Training\
↓\
Prediction\
↓\
Evaluation

------------------------------------------------------------------------

# 📈 Evaluation Metric

Model performance is measured using **RMSE (Root Mean Square Error)**.

RMSE = sqrt(mean((actual − predicted)\^2))

RMSE represents the **average prediction error magnitude**.

Lower RMSE means predictions are **closer to the true values**.

------------------------------------------------------------------------

# 🧪 Experiment Setup

Training configuration used in the main experiment:

Context Length : 24\
Prediction Horizon : 12\
Epochs : 50\
Batch Size : 32

Models were trained using **mini-batch gradient descent**.

------------------------------------------------------------------------

# 📊 Final Results

  Model         RMSE
  ------------- ----------
  Naive         13.02
  ARIMA         5.16
  LSTM          4.15
  Transformer   **3.81**

------------------------------------------------------------------------

# 🏆 Key Findings

-   The **Naive model performs poorly** because it does not learn
    patterns.
-   **ARIMA improves performance** by modeling linear temporal
    relationships.
-   **LSTM performs better** by learning nonlinear sequence patterns.
-   **Transformer achieves the best performance** by capturing global
    temporal dependencies using attention.

------------------------------------------------------------------------

# 📊 Dashboard Features

The Streamlit dashboard provides:

### Model Comparison

Displays RMSE values for each model.

### Forecast Visualization

Shows predicted vs actual electricity production.

### Residual Analysis

Analyzes prediction errors.

### Interactive Controls

Users can modify:

-   context length
-   prediction horizon
-   training epochs

------------------------------------------------------------------------

# 🗂 Project Structure

electricity-forecasting/

data/\
  Electric_Production.csv

src/\
  data_loader.py\
  preprocessing.py\
  windows.py\
  arima_model.py\
  lstm_model.py\
  transformer_model.py

experiments/\
  run_experiments.py

app/\
  streamlit_app.py

outputs/\
  experiment_results.csv\
  model_results.csv

plots/\
  rmse_comparison.png

models/\
  lstm_model.pth\
  transformer_model.pth

main.py\
requirements.txt\
README.md

------------------------------------------------------------------------

# 🚀 How to Run the Project

## Install dependencies

pip install -r requirements.txt

## Train models

python main.py

## Run Streamlit dashboard

streamlit run app/streamlit_app.py

------------------------------------------------------------------------

# 🛠 Technologies Used

-   Python\
-   PyTorch\
-   Streamlit\
-   Pandas\
-   NumPy\
-   Matplotlib\
-   Scikit-learn\
-   Statsmodels

------------------------------------------------------------------------

