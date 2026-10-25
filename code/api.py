from fastapi import FastAPI
from pathlib import Path
import joblib


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / 'models' / 'ridepulse_rf.joblib')

@app.get('/health')
def health():
    return {'status': 'ok'}