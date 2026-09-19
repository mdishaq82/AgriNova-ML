
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "06_Final_Model"

MODEL = joblib.load(MODEL_DIR / "best_model.joblib")
META = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

def predict_crop(features: dict):
    row = pd.DataFrame([[features[name] for name in META["features"]]], columns=META["features"])
    prediction = MODEL.predict(row)[0]

    confidence = None
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(row)[0]
        confidence = float(max(probs))

    return {
        "crop": str(prediction),
        "confidence": confidence,
        "model": META["best_model"]
    }
