# backend/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from backend.data import prepare_data
from backend.model_def import StockLSTM
import joblib  # pip install joblib or use sklearn.externals.joblib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(
    ticker: str = "AAPL",
    period: str = "5y",
    seq_len: int = 60,
    batch_size: int = 32,
    lr: float = 5e-4,
    epochs: int = 30,
    test_ratio: float = 0.2,
    save_dir: str = "backend/saved_models"
):
    os.makedirs(save_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, scaler, df = prepare_data(
        ticker=ticker,
        period=period,
        seq_len=seq_len,
        test_ratio=test_ratio
    )

    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_test_t = torch.tensor(X_test).to(DEVICE)
    y_test_t = torch.tensor(y_test).to(DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = StockLSTM(input_size=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.6f}")

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).cpu().numpy()
        y_true_test = y_test_t.cpu().numpy()

    # Inverse scale
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_true_test_inv = scaler.inverse_transform(y_true_test)

    mse = mean_squared_error(y_true_test_inv, y_pred_test_inv)
    mae = mean_absolute_error(y_true_test_inv, y_pred_test_inv)
    rmse = np.sqrt(mse)

    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save model & scaler
    model_path = os.path.join(save_dir, f"{ticker}_lstm_stock.pt")
    scaler_path = os.path.join(save_dir, f"{ticker}_scaler.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "seq_len": seq_len
    }

if __name__ == "__main__":
    # Simple default training run
    train_model(ticker="AAPL", period="5y")
