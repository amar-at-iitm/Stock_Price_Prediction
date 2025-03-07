# Stock Price Prediction using Python and Machine Learning

## Overview
This project implements **Stock Price Prediction** using **Linear Regression** and **LSTM (Long Short-Term Memory) models** to analyze and forecast stock prices based on historical data.

## Features
- **Linear Regression Model**: Predicts future stock prices using a simple regression approach.
- **LSTM Model**: A deep learning-based time series model that captures long-term dependencies in stock prices.
- **Data Preprocessing**: Uses MinMax scaling to normalize data for better model performance.
- **Model Training & Evaluation**: Implements training and validation using historical stock price data.
- **Visualization**: Plots stock price trends and prediction results.

## Dataset
The model expects a **CSV file** containing historical stock prices with at least the following columns:
- `Date` (format: YYYY-MM-DD)
- `Close` (Closing stock price for the given date)

Example:
```
Date,Close
2023-01-01,150.75
2023-01-02,152.30
```

## Installation
Ensure you have Python installed, then install dependencies using:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

## Usage
1. Place your dataset file (`stock_data.csv`) in the project directory.
2. Run the Python script to train the models:
   ```bash
   python stock_price_prediction.py
   ```
3. The script will train **Linear Regression** and **LSTM** models, displaying accuracy and loss metrics.

## Results & Evaluation
- **Linear Regression Accuracy**: Displays how well the model fits stock price trends.
- **LSTM Model Loss**: The training process outputs loss metrics to track model performance.
- **Prediction Plots**: The script generates visualizations for actual vs. predicted prices.

## Future Improvements
- Incorporate additional stock market indicators (e.g., volume, moving averages).
- Experiment with different deep learning architectures (GRU, Transformers).
- Implement real-time stock price prediction using live financial APIs.

## Acknowledgments
This project is a simple yet effective implementation of stock price prediction using Machine Learning and Deep Learning techniques.

Results
Linear Regression Accuracy is displayed in the console.
LSTM Model Loss is shown during training.
Future Improvements
Add more features like volume, moving averages.
Experiment with deeper LSTM architectures.
