from pathlib import Path
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = ROOT / "mlruns"
MODEL_NAME = "ManuscriptMoistureRisk"
FEATURES = [
    "relative_humidity_pct", "room_temperature_c",
    "wall_surface_temperature_c", "door_openings_per_hour",
    "ventilation_duration_min", "dehumidifier_operating_hours",
    "room_occupancy", "condensation_risk_index",
    "ventilation_effectiveness_ratio",
]


def train() -> dict:
    data = pd.read_csv(ROOT / "data" / "processed" / "manuscript_rooms_all_features.csv")
    X = data[FEATURES]
    y = data["moisture_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=4354, stratify=y
    )
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=7, random_state=4354, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=9, random_state=4354,
                                                class_weight="balanced", n_jobs=-1),
    }
    mlflow.set_tracking_uri(TRACKING_URI.as_uri())
    mlflow.set_experiment("manuscript_moisture_risk")
    results = {}

    for name, classifier in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline([("scale", StandardScaler()), ("model", classifier)])
            pipeline.fit(X_train, y_train)
            predicted = pipeline.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1": f1_score(y_test, predicted, zero_division=0),
            }
            mlflow.log_params(classifier.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline, "model",
                signature=infer_signature(X_test, predicted),
            )
            results[name] = {"run_id": run.info.run_id, "metrics": metrics}

    best_name = max(results, key=lambda name: (
        results[name]["metrics"]["recall"],
        results[name]["metrics"]["f1"],
    ))
    best = results[best_name]
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    with mlflow.start_run(run_name="register_best") as registration:
        model_run_uri = f"runs:/{best['run_id']}/model"
        registered = mlflow.register_model(model_run_uri, MODEL_NAME)
        mlflow.set_tag("selected_model", best_name)

    report = {"best_model": best_name, "models": results, "registered_model": registered.name,
              "registered_version": registered.version}
    (ROOT / "artifacts" / "model_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Local serving fallback copy, explicitly paired with API feature contract.
    best_run = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    joblib.dump(best_run, artifact_dir / "moisture_model.joblib")
    (artifact_dir / "feature_names.json").write_text(json.dumps(FEATURES))
    return report


if __name__ == "__main__":
    train()
