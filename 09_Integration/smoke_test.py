
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"06_Final_Model"))
from prediction_pipeline import predict_crop
META = json.loads((ROOT/"06_Final_Model/metadata.json").read_text())
sample = {f: 1.0 for f in META["features"]}
print("Integration import OK.")
print("Expected input fields:", len(META["features"]))
