# backend/inference.py
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import joblib

from backend.model_def import StockLSTM
from backend.data import download_stock_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_scaler(ticker: str, seq_len: int, save_dir: str = "backend/saved_models"):
    model_path = os.path.join(save_dir, f"{ticker}_lstm_stock.pt")
    scaler_path = os.path.join(save_dir, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train the model first.")

    model = StockLSTM(input_size=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler


def forecast_future(
    ticker: str,
    days_ahead: int,
    seq_len: int,
    period: str = "5y"
) -> Dict:

    # ───────────────────────────────────────────────
    # FIX 1: Normalize columns from yfinance
    # ───────────────────────────────────────────────
    df_raw = download_stock_data(ticker, period)

    df = df_raw.copy()
    df = df.rename(columns={
        "Close": "close",
        "close": "close",
        "Date": "date",
        "date": "date"
    })

    # ensure dtype correct
    df['close'] = df['close'].astype(float)

    close_vals = df['close'].values.reshape(-1, 1)

    # Load model + scaler
    model, scaler = load_model_and_scaler(ticker, seq_len)

    scaled_vals = scaler.transform(close_vals)

    last_seq = scaled_vals[-seq_len:]

    preds_scaled = []
    current_seq = last_seq.copy()

    # ───────────────────────────────────────────────
    # Predict future N days
    # ───────────────────────────────────────────────
    for _ in range(days_ahead):
        inp = torch.tensor(current_seq[np.newaxis, :, :].astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            pred = model(inp).cpu().numpy()[0, 0]

        preds_scaled.append(pred)
        current_seq = np.vstack((current_seq[1:], [[pred]]))

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    preds_smooth = pd.Series(preds).rolling(window=5, min_periods=1).mean().values

    # ───────────────────────────────────────────────
    # Prepare return data
    # ───────────────────────────────────────────────
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead)

    return {
        "historical_dates": df['date'].dt.strftime("%Y-%m-%d").to_list(),
        "historical_prices": df['close'].to_list(),
        "future_dates": future_dates.strftime("%Y-%m-%d").to_list(),
        "future_prices": preds_smooth.tolist()
    }
