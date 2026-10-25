FROM python:3.15-slim

WORKDIR /ridepulse

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY code/api.py code/api.py
COPY models/ridepulse_rf.joblib models/ridepulse_rf.joblib
COPY data/raw/taxi_zone_lookup.csv data/raw/taxi_zone_lookup.csv

WORKDIR /ridepulse/code

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]