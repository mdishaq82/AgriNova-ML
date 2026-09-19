
"""
AgriNova model training pipeline.
Run from project root:
    python 04_ML_Models/train_models.py

The bundled dataset is synthetic/prototype data. Do not present its metrics
as real-world agricultural performance.
"""
from pathlib import Path
import shutil
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_Data" / "raw" / "agriNova_prototype.csv"
MODEL_DIR = ROOT / "06_Final_Model"
EVAL_DIR = ROOT / "05_Evaluation"
MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

TARGET = "crop"
FEATURES = [c for c in pd.read_csv(DATA, nrows=1).columns if c != TARGET]

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=3, probability=True, random_state=42))
    ])
}

# XGBoost is optional so the core project remains runnable even if xgboost
# is not installed. requirements.txt includes it.
try:
    from xgboost import XGBClassifier
    classes = sorted(y.unique())
    class_to_int = {c:i for i,c in enumerate(classes)}
    int_to_class = {i:c for c,i in class_to_int.items()}
    y_train_xgb = y_train.map(class_to_int)
    models["XGBoost"] = Pipeline([
        ("model", XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=len(classes),
            eval_metric="mlogloss", random_state=42
        ))
    ])
except ImportError:
    print("xgboost not installed; skipping XGBoost. Install requirements.txt to enable it.")

results = []
best_name, best_f1 = None, -1

for name, model in models.items():
    if name == "XGBoost":
        model.fit(X_train, y_train_xgb)
        pred_int = model.predict(X_test).astype(int)
        pred = pd.Series(pred_int).map(int_to_class).values
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    results.append(metrics)

    safe = name.lower().replace(" ", "_")
    joblib.dump(model, MODEL_DIR / f"{safe}.joblib")

    report = classification_report(y_test, pred, zero_division=0)
    (EVAL_DIR / f"{safe}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
    pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())).to_csv(
        EVAL_DIR / f"{safe}_confusion_matrix.csv"
    )

    if metrics["f1_weighted"] > best_f1:
        best_f1 = metrics["f1_weighted"]
        best_name = name

results_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
results_df.to_csv(EVAL_DIR / "model_comparison.csv", index=False)

best_file = MODEL_DIR / f"{best_name.lower().replace(' ', '_')}.joblib"
shutil.copy2(best_file, MODEL_DIR / "best_model.joblib")

metadata = {
    "features": FEATURES,
    "target": TARGET,
    "best_model": best_name,
    "classes": sorted(y.unique()),
    "dataset_note": "Synthetic prototype dataset generated for learning/demo; not real-world agricultural observations."
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
