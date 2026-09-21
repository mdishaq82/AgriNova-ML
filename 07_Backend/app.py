from pathlib import Path
import sys
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '06_Final_Model'))
from prediction_pipeline import predict_crop, META

app = Flask(__name__, template_folder=str(ROOT / '08_Website' / 'frontend'))
FEATURES = META['features']

@app.get('/')
def home():
    return render_template('index.html')

@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'model': META['best_model'], 'accuracy': META['test_accuracy']})

@app.post('/api/predict')
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        features = {k: float(data[k]) for k in FEATURES}
        result = predict_crop(features)
        return jsonify({'ok': True, 'result': result})
    except KeyError as e:
        return jsonify({'ok': False, 'error': f'Missing field: {e.args[0]}'}), 400
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'All basic input values must be numeric.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
