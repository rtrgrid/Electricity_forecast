⚡ Electricity Forecasting using ARIMA, LSTM, and Transformer

An end-to-end time-series forecasting project that predicts electricity production using classical statistical models and modern deep learning architectures.

The project compares multiple forecasting approaches and demonstrates how Transformer models can outperform traditional models for sequential data prediction.

🌐 Live Demo

You can explore the interactive forecasting dashboard here:

Streamlit App
https://electricityforecast-cwdb6vgjhljudsnxn5z9kn.streamlit.app/

The dashboard allows users to:

Compare forecasting models

Change experiment parameters

Visualize predictions

Analyze model errors

Understand model performance

📊 Project Overview

Electricity production forecasting is an important time-series prediction problem used in:

Energy grid management

Demand forecasting

Infrastructure planning

Resource allocation

This project compares four forecasting approaches:

Model	Type
Naive Baseline	Simple heuristic
ARIMA	Statistical time-series model
LSTM	Recurrent neural network
Transformer	Attention-based deep learning model

The objective is to evaluate which architecture best captures temporal patterns in electricity production data.

📁 Dataset

Dataset used: Electric Production Dataset

The dataset contains monthly electricity production values.

Column	Description
Date	Timestamp
Value	Electricity production index

Dataset size:

397 time steps

The data contains clear trend and seasonal patterns, making it suitable for time-series forecasting experiments.

🧠 Models Implemented
1. Naive Baseline

The Naive model predicts the future value using the last observed value.

Prediction = Last Observed Value

Purpose:

Establishes a baseline performance

Helps determine if complex models actually improve predictions

2. ARIMA Model

ARIMA stands for:

AutoRegressive Integrated Moving Average

It models time series using three components:

AR (AutoRegressive) → relationship with past values

I (Integrated) → differencing to remove trends

MA (Moving Average) → relationship with past errors

Advantages:

Works well on small datasets

Interpretable statistical model

Captures linear time patterns

Limitations:

Cannot easily model nonlinear patterns

3. LSTM Model

LSTM (Long Short-Term Memory) is a Recurrent Neural Network designed for sequential data.

It contains three main gates:

Forget Gate

Input Gate

Output Gate

These gates allow the network to:

remember important past information

forget irrelevant information

maintain long-term dependencies

Advantages:

Learns nonlinear relationships

Good for sequential data

Handles longer dependencies than basic RNNs

Limitations:

Sequential computation slows training

Performance may degrade with very long sequences

4. Transformer Model

Transformers use self-attention mechanisms to model relationships between time steps.

Instead of processing sequences sequentially like RNNs, transformers evaluate relationships between all time steps simultaneously.

Advantages:

Captures global temporal dependencies

Highly parallelizable

Strong performance on sequential data

Transformers are widely used in:

GPT models

BERT

Vision Transformers

Time-series forecasting

⚙️ Forecasting Approach

The project uses a sliding window forecasting method.

Example configuration:

Context Length = 24
Prediction Horizon = 12

Meaning:

Past 24 time steps → used to predict next 12 time steps

Pipeline:

Time Series
      ↓
Sliding Window Creation
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Prediction
      ↓
Evaluation
📈 Evaluation Metric

The models are evaluated using RMSE (Root Mean Square Error).

Formula:

RMSE = sqrt( (1/n) * Σ(actual − predicted)² )

Interpretation:

RMSE	Meaning
High RMSE	Poor predictions
Low RMSE	Accurate predictions

Lower RMSE indicates better forecasting accuracy.

🧪 Experiment Configuration

Multiple experiments were conducted with different hyperparameters.

Parameters explored:

Context Length: 12 → 72
Prediction Horizon: 6 → 24
Transformer Dimension: 32, 64, 128
Epochs: 30 → 100
Batch Size: 32

Experiments were automated using:

experiments/run_experiments.py

Each experiment stores:

predictions

RMSE results

visualization plots

📊 Final Model Performance

Best experiment configuration:

Context Length: 24
Prediction Horizon: 12
Epochs: 50
Batch Size: 32

Results:

Model	RMSE
Naive	13.02
ARIMA	5.16
LSTM	4.15
Transformer	3.81
🏆 Key Result

The Transformer model achieved the lowest RMSE, making it the best performing model.

Compared to the naive baseline:

≈ 71% reduction in prediction error

This demonstrates that attention-based architectures can better capture temporal dependencies in time-series data.

📊 Dashboard Features

The Streamlit dashboard provides interactive model analysis.

Key features:

Model Comparison

Visual comparison of RMSE values across models.

Forecast Visualization

Graph showing predicted vs actual electricity production.

Residual Analysis

Analysis of prediction errors to identify bias and variance.

Parameter Exploration

Users can experiment with:

context length

prediction horizon

training epochs

🗂 Project Structure
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
   training_config.csv

plots/
   rmse_comparison.png

models/
   lstm_model.pth
   transformer_model.pth

main.py
requirements.txt
README.md
🚀 Running the Project
1. Install dependencies
pip install -r requirements.txt
2. Run experiments
python experiments/run_experiments.py
3. Launch the dashboard
streamlit run app/streamlit_app.py
🛠 Technologies Used

Python

PyTorch

Streamlit

NumPy

Pandas

Matplotlib

Scikit-learn

Statsmodels

📌 Key Learnings

This project demonstrates:

time-series preprocessing

sliding window forecasting

classical vs deep learning model comparison

transformer architecture for time-series prediction

experiment automation

interactive ML dashboards
