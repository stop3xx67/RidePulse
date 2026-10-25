import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / 'models' / 'ridepulse_rf.joblib')

trip = pd.DataFrame([{
    'PULocationID': 5,
    'DOLocationID': 10,
    'pickup_hour': 13,
    'pickup_weekday': 6,
    'Borough_pickup': 'Manhattan',
    'Borough_dropoff': 'Queens'
    }])

prediction = model.predict(trip)

print(f'Предположительное время поездки: {prediction[0]:.2f} минут')