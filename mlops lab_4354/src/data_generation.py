from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def create_records(version: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    monsoon = version == "v2"
    humidity = np.clip(rng.normal(72 if monsoon else 58, 12 if monsoon else 10, n), 30, 98)
    temperature = np.clip(rng.normal(25 if monsoon else 23, 3.5, n), 15, 38)
    wall_temperature = np.clip(temperature - rng.normal(3.0 if monsoon else 2.2, 1.2, n), 12, 36)
    door_openings = np.clip(rng.poisson(8 if monsoon else 5, n), 0, 30)
    ventilation = np.clip(rng.normal(25 if monsoon else 55, 12, n), 0, 120)
    dehumidifier = np.clip(rng.normal(7 if monsoon else 10, 3, n), 0, 24)
    occupancy = np.clip(rng.poisson(2.2, n), 0, 12)

    dew_point = temperature - ((100 - humidity) / 5)
    condensation = np.clip((dew_point - wall_temperature + 7) / 14, 0, 1)
    vent_ratio = np.clip(ventilation / np.maximum(door_openings, 1), 0, 12)
    if monsoon:
        vent_ratio *= rng.uniform(0.45, 0.75, n)

    risk_probability = (
        0.035 * np.maximum(humidity - 55, 0)
        + 0.17 * condensation
        + 0.025 * door_openings
        - 0.012 * ventilation
        - 0.018 * dehumidifier
        + 0.015 * occupancy
        - 0.28
    )
    risk = (risk_probability + rng.normal(0, 0.24, n) > 0).astype(int)

    return pd.DataFrame({
        "relative_humidity_pct": humidity.round(2),
        "room_temperature_c": temperature.round(2),
        "wall_surface_temperature_c": wall_temperature.round(2),
        "door_openings_per_hour": door_openings.astype(int),
        "ventilation_duration_min": ventilation.round(2),
        "dehumidifier_operating_hours": dehumidifier.round(2),
        "room_occupancy": occupancy.astype(int),
        "moisture_risk": risk,
        "dataset_version": version,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=250)
    args = parser.parse_args()
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    v1 = create_records("v1", args.n, 4354)
    v2 = create_records("v2", args.n, 4355)
    v1.to_csv(out / "manuscript_rooms_v1.csv", index=False)
    v2.to_csv(out / "manuscript_rooms_v2.csv", index=False)
    pd.concat([v1, v2]).to_csv(out / "manuscript_rooms.csv", index=False)
    print(f"Wrote {len(v1)} v1 and {len(v2)} v2 records")


if __name__ == "__main__":
    main()
