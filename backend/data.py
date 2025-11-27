# backend/data.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(ticker, period=period)

    # Flatten MultiIndex columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(c) for c in col if c]) for col in data.columns]

    # Reset index to get date column
    data = data.reset_index()

    # Normalize date column
    for col in data.columns:
        if "date" in col.lower():
            data.rename(columns={col: "date"}, inplace=True)

    # Normalize close column
    close_col = None
    for col in data.columns:
        if "close" in col.lower():
            close_col = col
            break

    if close_col is None:
        raise ValueError(f"No close price column found. Columns: {data.columns}")

    data.rename(columns={close_col: "close"}, inplace=True)

    # Ensure correct types
    data["date"] = pd.to_datetime(data["date"])
    data["close"] = data["close"].astype(float)

    return data

def create_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(
    ticker: str = "AAPL",
    period: str = "5y",
    seq_len: int = 60,
    test_ratio: float = 0.2
):
    df = download_stock_data(ticker, period)
    values = df['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_sequences(scaled, seq_len)

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Add feature dimension for LSTM: (batch, seq_len, features)
    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
        scaler,
        df
    )
