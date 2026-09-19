
from pathlib import Path
import sys, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"06_Final_Model"))
from prediction_pipeline import predict_crop

def test_prediction():
    meta = json.loads((ROOT/"06_Final_Model/metadata.json").read_text())
    row = pd.read_csv(ROOT/"01_Data/raw/agriNova_prototype.csv").iloc[0]
    features = {f: float(row[f]) for f in meta["features"]}
    result = predict_crop(features)
    assert result["crop"] in meta["classes"]
    print("PASS: prediction returns a valid crop.")

if __name__ == "__main__":
    test_prediction()
