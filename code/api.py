from fastapi import FastAPI
from pathlib import Path
import joblib
from pydantic import BaseModel, Field
import pandas as pd

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'taxi_zone_lookup.csv')
model = joblib.load(BASE_DIR / 'models' / 'ridepulse_rf.joblib')

class NewData(BaseModel):
    place_id_from: int = Field(ge=0, le=265)
    place_id_to: int = Field(ge=0, le=265)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_weekday: int = Field(ge=0, le=6)

@app.post('/predict')
def predict(data: NewData):
    obj = [
        data.place_id_from,
        data.place_id_to,
        data.pickup_hour,
        data.pickup_weekday,
        df[df['LocationID' == data.place_id_from]]['Borough'].iloc[0],
        df[df['LocationID' == data.place_id_to]]['Borough'].iloc[0]
        ]
    
    prediction = model.predict(pd.DataFrame(
        {
        'PULocationID': obj[0],
        'DOLocationID': obj[1],
        'pickup_hour': obj[2],
        'pickup_weekday': obj[3],
        'Borough_pickup': obj[4],
        'Borough_dropoff': obj[5]
        }
    ))
    
    return {'Ваше приблизительное время поездки в минутах -': round(float(prediction[0]), 2)}