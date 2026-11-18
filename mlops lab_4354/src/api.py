from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
try:
    from .features import engineer_features
except ImportError:
    from features import engineer_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "moisture_model.joblib"
app = FastAPI(title="Manuscript Moisture Risk API")


class StorageObservation(BaseModel):
    relative_humidity_pct: float = Field(..., ge=0, le=100)
    room_temperature_c: float = Field(..., ge=-10, le=60)
    wall_surface_temperature_c: float = Field(..., ge=-10, le=60)
    door_openings_per_hour: int = Field(..., ge=0)
    ventilation_duration_min: float = Field(..., ge=0)
    dehumidifier_operating_hours: float = Field(..., ge=0, le=24)
    room_occupancy: int = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(observation: StorageObservation):
    if not MODEL_PATH.exists():
        return {"prediction": 1, "status": "Moisture Risk",
                "detail": "Model artifact unavailable"}
    row = engineer_features(pd.DataFrame([observation.model_dump()]))
    features = json.loads((ROOT / "artifacts" / "feature_names.json").read_text())
    model = joblib.load(MODEL_PATH)
    risk = int(model.predict(row[features])[0])
    return {
        "prediction": risk,
        "status": "Moisture Risk" if risk == 1 else "Storage Conditions Acceptable",
        "condensation_risk_index": float(row.iloc[0]["condensation_risk_index"]),
        "ventilation_effectiveness_ratio": float(row.iloc[0]["ventilation_effectiveness_ratio"]),
    }
