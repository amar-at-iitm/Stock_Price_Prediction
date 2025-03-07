import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df = df[['Close']]
    return df

# Preprocess data for LSTM
def prepare_lstm_data(df, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dense(units=25),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate models
def train_models(file_path):
    df = load_data(file_path)
    
    # Linear Regression
    df['Shifted'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df[['Close']], df['Shifted'], test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_accuracy = lr_model.score(X_test, y_test)
    print(f'Linear Regression Accuracy: {lr_accuracy:.2f}')
    
    # LSTM
    X_lstm, y_lstm, scaler = prepare_lstm_data(df[['Close']])
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
    
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    
    return lr_model, lstm_model, scaler

# Run training
file_path = 'stock_data.csv'  # Replace with your actual file path
train_models(file_path)
