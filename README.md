⚡ Electricity Forecasting using ARIMA, LSTM and Transformer

An end-to-end time-series forecasting project that predicts electricity production using classical statistical models and modern deep learning architectures.

The project compares multiple forecasting approaches and demonstrates how Transformer models outperform traditional methods for sequential data prediction.

🌐 Live Dashboard

You can explore the interactive forecasting dashboard here:

👉 Streamlit App
https://electricityforecast-cwdb6vgjhljudsnxn5z9kn.streamlit.app/

The dashboard allows users to:

Compare forecasting models

Change experiment parameters

Visualize predictions

Analyze model errors

Understand model performance

📊 Project Overview

Electricity production forecasting is an important time-series prediction problem used in:

energy grid management

demand forecasting

infrastructure planning

resource allocation

In this project we compare four approaches:

Model	Type
Naive Baseline	Simple heuristic
ARIMA	Statistical time-series model
LSTM	Recurrent neural network
Transformer	Attention-based deep learning model

The goal is to evaluate which architecture best captures temporal patterns in electricity production data.

📁 Dataset

Dataset used:

Electric Production Dataset

Contains monthly electricity production values.

Columns:

Column	Description
Date	timestamp
Value	electricity production index

Dataset size:

397 time steps

The dataset contains clear trend and seasonal patterns, making it ideal for forecasting experiments.

🧠 Models Implemented
1️⃣ Naive Baseline

The Naive model simply predicts:

future value = last observed value

This serves as a baseline benchmark.

It helps evaluate whether more complex models actually learn meaningful patterns.

2️⃣ ARIMA Model

ARIMA stands for:

AutoRegressive Integrated Moving Average

It models time series as a combination of:

autoregression

differencing

moving average

General form:

y(t) = c + Σ φi y(t-i) + Σ θj ε(t-j)

Strengths:

interpretable

good for small datasets

captures linear trends

Limitations:

struggles with nonlinear patterns

3️⃣ LSTM Model

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) designed for sequence learning.

It uses three gates:

Forget gate

Input gate

Output gate

These gates allow the network to maintain long-term memory of past time steps.

Advantages:

captures nonlinear temporal relationships

learns complex sequence patterns

Limitations:

sequential computation

slower training

struggles with very long context windows

4️⃣ Transformer Model

Transformers use self-attention mechanisms to model relationships between time steps.

Instead of processing sequences step-by-step like RNNs, transformers compute attention across the entire sequence.

Attention mechanism:

Attention(Q,K,V) = softmax(QKᵀ / √d) V

Advantages:

captures global temporal dependencies

parallel computation

strong representation learning

This architecture is widely used in:

NLP (GPT, BERT)

computer vision

time-series forecasting

⚙️ Forecasting Method

The project uses sliding window forecasting.

Example:

Context Length = 24
Prediction Horizon = 12

Meaning:

Use past 24 time steps
→ predict next 12 time steps

Window creation pipeline:

Time series
    ↓
Sliding window
    ↓
Training sequences
    ↓
Model prediction
📈 Evaluation Metric

We evaluate models using RMSE (Root Mean Square Error).

RMSE measures the average prediction error magnitude.

Interpretation:

RMSE	Meaning
high	poor predictions
low	accurate predictions

Lower RMSE indicates better forecasting performance.

🧪 Experiment Configuration

Experiments were run with multiple hyperparameters.

Parameters explored:

Context length: 12 → 72
Prediction horizon: 6 → 24
Transformer model dimension: 32, 64, 128
Training epochs: 30 → 100
Batch size: 32

Experiments were automated using:

experiments/run_experiments.py

Each experiment saves:

predictions

visualizations

RMSE scores

📊 Final Model Performance

Best experiment configuration:

Context length: 24
Prediction horizon: 12
Epochs: 50
Batch size: 32

Results:

Model	RMSE
Naive	13.02
ARIMA	5.16
LSTM	4.15
Transformer	3.81
🏆 Key Result

The Transformer model achieved the best forecasting accuracy.

Improvement compared to Naive baseline:

≈ 71% reduction in prediction error

This shows that attention-based models capture temporal patterns more effectively than traditional methods.

📊 Dashboard Features

The Streamlit dashboard provides interactive visualizations.

Features include:

Model Comparison

Compare RMSE values across models.

Forecast Visualization

Shows predicted vs actual electricity production.

Residual Analysis

Analyzes prediction errors to evaluate model bias.

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
🚀 How to Run the Project
Install dependencies
pip install -r requirements.txt
Run experiments
python experiments/run_experiments.py
Launch dashboard
streamlit run app/streamlit_app.py
📚 Technologies Used

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

time-series preprocessing techniques

sliding window forecasting

classical vs deep learning model comparison

transformer architectures for time series

experiment automation

ML visualization dashboards
