from pathlib import Path
import argparse
import pandas as pd
from features import engineer_features

ROOT = Path(__file__).resolve().parents[1]


def preprocess(version: str = "v1") -> Path:
    raw = ROOT / "data" / "raw" / "manuscript_rooms.csv"
    df = pd.read_csv(raw)
    df = df[df["dataset_version"] == version].copy() if version != "all" else df
    df = engineer_features(df)
    out = ROOT / "data" / "processed" / f"manuscript_rooms_{version}_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all")
    args = parser.parse_args()
    preprocess(args.version)
