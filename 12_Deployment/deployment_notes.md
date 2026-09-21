# Render deployment

Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn --chdir 07_Backend app:app`

The model file `06_Final_Model/best_model.joblib` must be committed to Git because the deployed service needs it at runtime.
