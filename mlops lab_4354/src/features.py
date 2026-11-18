import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    humidity = result["relative_humidity_pct"].clip(0.1, 99.9)
    temp = result["room_temperature_c"]
    wall = result["wall_surface_temperature_c"]
    dew_point = temp - ((100 - humidity) / 5)
    result["condensation_risk_index"] = np.clip(
        (dew_point - wall + 7) / 14, 0, 1
    ).round(6)
    result["ventilation_effectiveness_ratio"] = np.clip(
        result["ventilation_duration_min"]
        / np.maximum(result["door_openings_per_hour"], 1),
        0,
        12,
    ).round(6)
    if "dataset_version" in result:
        result = result.drop(columns=["dataset_version"])
    return result
