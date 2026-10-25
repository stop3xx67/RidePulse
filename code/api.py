from fastapi import FastAPI, HTTPException
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
    pickup_row = df[df['LocationID'] == data.place_id_from]

    if pickup_row.empty:
        raise HTTPException(
            status_code=422,
            detail='Извините, зона посадки не найдена'
            )

    pickup_borough = pickup_row['Borough'][0]

    if pd.isna(pickup_borough):
        raise HTTPException(
            status_code=422,
            detail='Извините зона посадки не определён'
            )
    
    dropoff_row = df[df['LocationID'] == data.place_id_to]

    if dropoff_row.empty:
        raise HTTPException(
            status_code=422,
            detail='Извините, зона высадки не найдена'
            )

    dropoff_borough = dropoff_row['Borough'][0]

    if pd.isna(dropoff_row):
        raise HTTPException(
            status_code=422,
            detail='Извините, зона высадки не определена'
            )

    obj = [
        data.place_id_from,
        data.place_id_to,
        data.pickup_hour,
        data.pickup_weekday,
        df[df['LocationID'] == data.place_id_from]['Borough'].iloc[0],
        df[df['LocationID'] == data.place_id_to]['Borough'].iloc[0]
        ]

    print(obj)
    print(type(obj[4]), type(obj[5]))
    
    prediction = model.predict(pd.DataFrame([
        {
        'PULocationID': obj[0],
        'DOLocationID': obj[1],
        'pickup_hour': obj[2],
        'pickup_weekday': obj[3],
        'Borough_pickup': obj[4],
        'Borough_dropoff': obj[5]
        }
    ]))
    
    return {'Ваше приблизительное время поездки в минутах -': round(float(prediction[0]), 2)}