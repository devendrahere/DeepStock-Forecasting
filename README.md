# DL Stock Forecasting

This repository contains a small FastAPI-based web app and training/inference utilities for LSTM-based stock forecasting.

**Project Layout**
- `backend/`: Python backend (FastAPI) and model code.
  - `app.py`: FastAPI application and API endpoints.
  - `train.py`: Training utilities to build and save models.
  - `inference.py`: Forecasting utilities used by the API.
  - `model_def.py`, `data.py`, `debug_df.py`: helper modules.
  - `saved_models/`: pre-trained model artifacts (e.g., `AAPL_lstm_stock.pt`).
- `static/`, `templates/`: frontend static files and `index.html` used by FastAPI.
- `requirements.txt`: Python dependencies.

**Quick Start**
- **Requirements**: Python 3.8+ and `pip`.
- **Install deps**: Install the repository requirements.

```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- **Run the server**: Start the FastAPI app (uses `uvicorn` internally).

```cmd
python backend\app.py
```

- **Open in browser**: Visit `http://localhost:8000` to view the UI.

**API Endpoints**
- `GET /` : Returns the frontend `index.html`.
- `GET /api/logo/{ticker}` : Returns a small logo URL for `{ticker}`.
- `POST /api/train` : Triggers training for a ticker. Expects `application/x-www-form-urlencoded` form data with fields:
  - `ticker` (required) — stock ticker symbol (e.g., `AAPL`).
  - `period` (optional, default `5y`) — historical period to fetch.

  Example (PowerShell/curl):

  ```cmd
  curl -X POST "http://localhost:8000/api/train" -F "ticker=AAPL" -F "period=5y"
  ```

- `POST /api/predict` : Request JSON body with `ticker` and `days` to forecast.

  Example:
  ```cmd
  curl -X POST "http://localhost:8000/api/predict" -H "Content-Type: application/json" -d "{ \"ticker\": \"AAPL\", \"days\": 7 }"
  ```

**Saved Models**
- Pretrained models are stored in `backend/saved_models/`. Example files included:
  - `AAPL_lstm_stock.pt`
  - `GLD_lstm_stock.pt`
  - `TSLA_lstm_stock.pt`

**Development Notes**
- The server runs on port `8000` by default (`uvicorn` call inside `backend/app.py`).
- Training and inference functions live in `backend/train.py` and `backend/inference.py`.
- Sequence length for models is set to `SEQ_LEN = 60` in `backend/app.py`; keep training and inference consistent.

**Next Steps**
- Add a `README.dev.md` with developer tips and how to run training experiments.
- Add automated tests for `train` and `inference` functions.
- Add a simple GitHub Actions workflow to run linting and tests on push.

**Contact / Help**
- If you want, I can:
  - Run the app locally and verify the endpoints.
  - Add a small `Makefile` or `scripts/` folder with common commands.
  - Create a minimal `README.dev.md` or CI workflow.
