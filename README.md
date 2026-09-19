
# AgriNova — ML Crop Recommendation System

## Purpose
AgriNova is a learning-oriented prototype that predicts a suitable crop from soil, weather, forecast, satellite-style, and market-related features.

## Important data note
The bundled `agriNova_prototype.csv` is **synthetic data generated for development/learning**. It is not a real historical agricultural dataset and must not be used to claim real-world agricultural accuracy.

## Architecture
Farmer inputs → validation → preprocessing/model pipeline → ML classifier → crop recommendation → Flask API → website.

## Models
- Random Forest
- SVM
- XGBoost (enabled when the `xgboost` package is installed)

## Run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python 03_Preprocessing/preprocess.py
python 04_ML_Models/train_models.py
python 02_EDA/run_eda.py
python 10_Testing/test_prediction.py
python 07_Backend/app.py
```

Open `http://127.0.0.1:5000`.

## Suggested learning order
1. Read the dataset dictionary.
2. Run EDA.
3. Understand train/test split.
4. Read each model implementation.
5. Compare metrics.
6. Understand `prediction_pipeline.py`.
7. Understand Flask API.
8. Understand the website-to-API flow.

## Next upgrade
Replace the synthetic prototype data with properly sourced real agricultural, weather, market, and satellite-derived data. Only then should real-world performance claims be made.
