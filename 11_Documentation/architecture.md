# AgriNova Architecture

Browser → HTML/CSS/JavaScript → Flask `/api/predict` → prediction pipeline → saved Random Forest/XGBoost/SVM model → JSON response → website result.

The current website sends exactly seven features supported by the training data. Advanced fields are intentionally UI-only until a compatible training dataset is added.
