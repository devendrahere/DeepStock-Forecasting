# backend/app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
import uvicorn

from backend.train import train_model
from backend.inference import forecast_future
import requests

app = FastAPI()

app.mount("/static", StaticFiles(directory="backend/static"), name="static")
templates = Jinja2Templates(directory="backend/templates")

class PredictRequest(BaseModel):
    ticker: str
    days: int

SEQ_LEN = 60  # Should match training

@app.get("/api/logo/{ticker}")
def get_logo(ticker: str):
    ticker = ticker.upper()
    logo_url = f"https://companiesmarketcap.com/img/company-logos/64/{ticker}.png"
    return { "logo": logo_url }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )

@app.post("/api/train")
async def api_train(
    ticker: str = Form(...),
    period: str = Form("5y")
):
    result = train_model(ticker=ticker, period=period, seq_len=SEQ_LEN)
    return JSONResponse({
        "message": "Training completed",
        "ticker": ticker,
        "metrics": {
            "mse": result["mse"],
            "rmse": result["rmse"],
            "mae": result["mae"]
        }
    })

@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    data = forecast_future(
        ticker=req.ticker,
        days_ahead=req.days,
        seq_len=SEQ_LEN
    )
    return JSONResponse(data)

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
