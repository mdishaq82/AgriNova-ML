# AgriNova — AI-Based Crop Recommendation

AgriNova is a machine-learning crop recommendation prototype. The current model uses seven inputs: N, P, K, temperature, humidity, pH and rainfall.

## Current workflow

Dataset → preprocessing → Random Forest / XGBoost / SVM → evaluation → selected model → Flask API → HTML/JavaScript website.

## Current dataset

`01_Data/raw/Crop_recommendation.csv` contains 2,200 rows, 7 model features and a crop target. It is the dataset supplied for this project. Do not describe it as a specified number of years of raw field observations unless independently verified.

## Run locally

```bash
python -m pip install -r requirements.txt
python 03_Preprocessing/preprocess.py
python 02_EDA/run_eda.py
python 04_ML_Models/train_models.py
python 09_Integration/smoke_test.py
python -m pytest 10_Testing/test_prediction.py
python 07_Backend/app.py
```

Open `http://127.0.0.1:5000`.

## Website modes

**Basic Prediction:** N, P, K, temperature, humidity, pH and rainfall. These are the inputs supported by the current trained model.

**Advanced Mode:** The interface reserves space for soil moisture, NDVI, weather forecast and market information. These fields are not sent to the current model because the current training dataset does not contain them. They should only be activated after a compatible dataset is obtained and the model is retrained.

## Accuracy vs confidence

The website displays the model's **held-out test-set accuracy**. It does not display per-prediction confidence as accuracy. These are different concepts.

## Deployment

For Render, use:

Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn --chdir 07_Backend app:app`
