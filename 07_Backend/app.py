
from pathlib import Path
import sys
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"06_Final_Model"))
from prediction_pipeline import predict_crop

app = Flask(__name__, template_folder=str(ROOT/"08_Website/frontend"))

FEATURES = [
    "nitrogen","phosphorus","potassium","temperature_c","humidity_pct",
    "soil_ph","rainfall_mm","ndvi","soil_moisture",
    "forecast_temperature_c","forecast_rainfall_7d_mm",
    "market_price","market_demand_index"
]

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        features = {k: float(data[k]) for k in FEATURES}
        return jsonify({"ok": True, "result": predict_crop(features)})
    except KeyError as e:
        return jsonify({"ok": False, "error": f"Missing field: {e.args[0]}"}), 400
    except ValueError:
        return jsonify({"ok": False, "error": "All input values must be numeric."}), 400

if __name__ == "__main__":
    app.run(debug=True)
