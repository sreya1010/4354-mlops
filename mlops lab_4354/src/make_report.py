from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parents[1]


def add_metric_chart(metrics_path: Path, output: Path) -> None:
    report = json.loads(metrics_path.read_text())
    names = list(report["models"])
    recalls = [report["models"][name]["metrics"]["recall"] for name in names]
    f1 = [report["models"][name]["metrics"]["f1"] for name in names]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = range(len(names))
    ax.bar([i - 0.15 for i in x], recalls, width=0.3, label="Recall")
    ax.bar([i + 0.15 for i in x], f1, width=0.3, label="F1")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def make_report() -> Path:
    metrics_path = ROOT / "artifacts" / "model_metrics.json"
    if not metrics_path.exists():
        raise RuntimeError("Run src/train.py before generating the report")
    report = json.loads(metrics_path.read_text())
    chart = ROOT / "reports" / "model_comparison.png"
    add_metric_chart(metrics_path, chart)

    v1 = pd.read_csv(ROOT / "data" / "raw" / "manuscript_rooms_v1.csv")
    v2 = pd.read_csv(ROOT / "data" / "raw" / "manuscript_rooms_v2.csv")

    pdf_path = ROOT / "reports" / "Moisture_Risk_MLOps_Report.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = [Paragraph("Predicting Moisture Damage in Manuscript Storage Rooms", styles["Title"]),
             Spacer(1, 12),
             Paragraph("Problem definition", styles["Heading1"]),
             Paragraph("The system predicts whether a climate-controlled manuscript storage room is likely "
                       "to experience moisture damage risk within 48 hours. It supports early intervention "
                       "and uses humidity, temperature, wall temperature, door activity, ventilation, "
                       "dehumidification, and occupancy observations.", styles["BodyText"]),
             Paragraph("Workflow architecture", styles["Heading1"]),
             Paragraph("Synthetic data generation -> versioned raw datasets -> feature engineering and "
                       "processed feature store -> Decision Tree/Random Forest training -> MLflow registry "
                       "-> FastAPI model serving -> GitHub Actions CI/CD.", styles["BodyText"]),
             Paragraph("Dataset versioning", styles["Heading1"]),
             Paragraph(f"Version 1 contains {len(v1)} records. Version 2 contains {len(v2)} monsoon "
                       f"records with mean humidity {v2['relative_humidity_pct'].mean():.2f}% "
                       f"versus {v1['relative_humidity_pct'].mean():.2f}% in Version 1.", styles["BodyText"]),
             Paragraph("Feature engineering", styles["Heading1"]),
             Paragraph("condensation_risk_index estimates wall-condensation likelihood from dew-point "
                       "approximation and wall surface temperature. ventilation_effectiveness_ratio "
                       "relates ventilation duration to door-opening activity.", styles["BodyText"]),
             Paragraph("Model evaluation", styles["Heading1"])]

    rows = [["Model", "Accuracy", "Precision", "Recall", "F1"]]
    for name, result in report["models"].items():
        m = result["metrics"]
        rows.append([name, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                     f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#355c7d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
    ]))
    story += [table, Spacer(1, 12), Image(str(chart), width=420, height=210),
              Paragraph(f"Best model: {report['best_model']}; MLflow registered model "
                        f"{report['registered_model']} version {report['registered_version']}.",
                        styles["BodyText"]),
              Paragraph("MLflow evidence", styles["Heading1"]),
              Paragraph("Training and metric evidence is stored locally in mlruns/. The registered model "
                        "name is ManuscriptMoistureRisk. The artifact is mirrored to "
                        "artifacts/moisture_model.joblib for serving.", styles["BodyText"]),
              Paragraph("API and CI/CD", styles["Heading1"]),
              Paragraph("FastAPI exposes GET /health and POST /predict. Pydantic validates humidity "
                        "(0-100%), temperature (-10 to 60 C), wall temperature (-10 to 60 C), and "
                        "non-negative activity fields. GitHub Actions regenerates data, builds features, "
                        "trains the models, and runs pytest on every push and pull request.",
                        styles["BodyText"]),
              Paragraph("API response", styles["Heading1"]),
              Paragraph("The API returns either \"Moisture Risk\" or "
                        "\"Storage Conditions Acceptable\", along with both engineered feature values.",
                        styles["BodyText"]),
              Paragraph("Example prediction result", styles["Heading1"]),
              Paragraph('{\"prediction\": 1, \"status\": \"Moisture Risk\", '
                        '\"condensation_risk_index\": 0.457143, '
                        '\"ventilation_effectiveness_ratio\": 4.285714}', styles["Code"])]
    swagger = ROOT / "screenshots" / "swagger_api.png"
    if swagger.exists():
        story += [Paragraph("API documentation screenshot", styles["Heading1"]),
                  Image(str(swagger), width=420, height=330)]
    doc.build(story)
    print(pdf_path)
    return pdf_path


if __name__ == "__main__":
    make_report()
