# Manuscript Moisture Risk MLOps System

This project predicts moisture damage risk for climate-controlled manuscript storage rooms.

## Run locally

```powershell
python -m pip install -r requirements.txt
python src/data_generation.py --n 250
python src/preprocess.py --version all
python src/train.py
uvicorn src.api:app --reload --port 8000
pytest -q
```

Open `http://127.0.0.1:8000/docs` for the interactive API.

## Dataset versions

- `v1`: baseline synthetic room observations.
- `v2`: monsoon period with higher humidity, more door openings, lower ventilation efficiency, and 250 observations.

Processed data is stored in `data/processed/` and reused for training and API feature engineering.

## MLflow

Training writes local runs to `mlruns/` and registers the best model as `ManuscriptMoistureRisk`.

## API example

```json
{
  "relative_humidity_pct": 82,
  "room_temperature_c": 24,
  "wall_surface_temperature_c": 21,
  "door_openings_per_hour": 7,
  "ventilation_duration_min": 30,
  "dehumidifier_operating_hours": 5,
  "room_occupancy": 2
}
```
