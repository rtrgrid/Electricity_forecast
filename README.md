# ⚡ Electricity Forecasting using ARIMA, LSTM, and Transformer

An end-to-end **time-series forecasting project** that predicts
electricity production using classical statistical models and modern
deep learning architectures.

The project compares multiple forecasting approaches and demonstrates
how **Transformer models outperform traditional models** for sequential
time-series prediction.

------------------------------------------------------------------------

# 🌐 Live Streamlit Dashboard

Explore the interactive dashboard here:

**Streamlit App:**\
https://electricityforecast-cwdb6vgjhljudsnxn5z9kn.streamlit.app/

The dashboard allows users to:

-   Compare forecasting models
-   Change experiment parameters
-   Visualize predictions
-   Analyze model errors
-   Understand model performance

------------------------------------------------------------------------

# 📊 Project Overview

Electricity production forecasting is an important **time-series
prediction problem** used in:

-   Energy grid management\
-   Demand forecasting\
-   Infrastructure planning\
-   Resource allocation

This project compares **four forecasting approaches**:

  Model            Type
  ---------------- -------------------------------------
  Naive Baseline   Simple heuristic
  ARIMA            Statistical time-series model
  LSTM             Recurrent neural network
  Transformer      Attention-based deep learning model

The goal is to determine **which architecture best captures temporal
patterns in electricity production data**.

------------------------------------------------------------------------

# 📁 Dataset

Dataset used: **Electric Production Dataset**

The dataset contains **monthly electricity production values**.

  Column   Description
  -------- ------------------------------
  Date     Timestamp
  Value    Electricity production index

Dataset size:

397 time steps

The data contains clear **seasonal patterns and trends**, making it
ideal for forecasting experiments.

------------------------------------------------------------------------

# 🧠 Models Implemented

## 1. Naive Baseline

The Naive model predicts the **last observed value**.

Prediction = Last Observed Value

Purpose:

-   Establishes a baseline performance
-   Helps determine if complex models improve predictions

------------------------------------------------------------------------

## 2. ARIMA Model

ARIMA stands for:

AutoRegressive Integrated Moving Average

It models time series using three components:

-   AR (AutoRegressive) → relationship with past values\
-   I (Integrated) → differencing to remove trends\
-   MA (Moving Average) → relationship with past errors

Advantages:

-   Works well on smaller datasets\
-   Interpretable statistical model\
-   Captures linear trends

Limitations:

-   Cannot easily model nonlinear patterns

------------------------------------------------------------------------

## 3. LSTM Model

LSTM (**Long Short-Term Memory**) is a type of **Recurrent Neural
Network (RNN)** designed for sequential data.

It contains three gates:

-   Forget Gate
-   Input Gate
-   Output Gate

These gates allow the model to:

-   remember important past information
-   forget irrelevant information
-   maintain long-term dependencies

Advantages:

-   Captures nonlinear relationships
-   Learns temporal patterns
-   Handles sequential data well

Limitations:

-   Sequential computation slows training

------------------------------------------------------------------------

## 4. Transformer Model

Transformers use **self-attention mechanisms** to model relationships
between time steps.

Instead of processing sequences step-by-step like RNNs, transformers
analyze **relationships between all time steps simultaneously**.

Advantages:

-   Captures global temporal dependencies
-   Highly parallelizable
-   Excellent performance on sequential data

Transformers are widely used in:

-   GPT models
-   BERT
-   Vision Transformers
-   Time-series forecasting

------------------------------------------------------------------------

# ⚙️ Forecasting Method

The project uses a **sliding window forecasting approach**.

Example configuration:

Context Length = 24\
Prediction Horizon = 12

Meaning:

Past 24 time steps → used to predict next 12 time steps

Pipeline:

Time Series → Sliding Window Creation → Model Training → Prediction →
Evaluation

------------------------------------------------------------------------

# 📈 Evaluation Metric

The models are evaluated using **RMSE (Root Mean Square Error)**.

RMSE = sqrt((1/n) \* Σ(actual − predicted)²)

Lower RMSE indicates **better forecasting performance**.

------------------------------------------------------------------------

# 📊 Final Model Performance

Best experiment configuration:

Context Length: 24\
Prediction Horizon: 12\
Epochs: 50\
Batch Size: 32

Results:

  Model         RMSE
  ------------- ----------
  Naive         13.02
  ARIMA         5.16
  LSTM          4.15
  Transformer   **3.81**

------------------------------------------------------------------------

# 🏆 Key Result

The **Transformer model achieved the lowest RMSE**, making it the best
performing model.

Compared to the naive baseline:

≈ 71% reduction in prediction error

------------------------------------------------------------------------

# 🗂 Project Structure

    electricity-forecasting/

    data/
       Electric_Production.csv

    src/
       data_loader.py
       preprocessing.py
       windows.py
       lstm_model.py
       transformer_model.py
       arima_model.py

    experiments/
       run_experiments.py

    app/
       streamlit_app.py

    outputs/
       experiment_results.csv
       model_results.csv

    plots/
       rmse_comparison.png

    models/
       lstm_model.pth
       transformer_model.pth

    main.py
    requirements.txt
    README.md

------------------------------------------------------------------------

# 🚀 Running the Project

## Install dependencies

pip install -r requirements.txt

## Run experiments

python experiments/run_experiments.py

## Launch Streamlit dashboard

streamlit run app/streamlit_app.py

------------------------------------------------------------------------

# 🛠 Technologies Used

-   Python
-   PyTorch
-   Streamlit
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   Statsmodels

------------------------------------------------------------------------


