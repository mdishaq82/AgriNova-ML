from pathlib import Path
import json, shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '01_Data' / 'processed' / 'crop_recommendation_processed.csv'
MODEL_DIR = ROOT / '06_Final_Model'; MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR = ROOT / '05_Evaluation'; EVAL_DIR.mkdir(exist_ok=True)

FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
TARGET = 'label'
df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
classes = sorted(y.unique())
class_to_int = {c:i for i,c in enumerate(classes)}
int_to_class = {i:c for c,i in class_to_int.items()}
y_train_xgb = y_train.map(class_to_int)
y_test_xgb = y_test.map(class_to_int)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, objective='multi:softprob', num_class=len(classes), eval_metric='mlogloss', random_state=42, n_jobs=-1),
    'SVM': Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf', C=3, probability=True, random_state=42))])
}
results=[]
for name, model in models.items():
    if name == 'XGBoost':
        model.fit(X_train, y_train_xgb)
        pred = np.asarray(model.predict(X_test), dtype=int)
        pred = np.array([int_to_class[int(v)] for v in pred])
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    metrics = {'model':name, 'accuracy':acc, 'precision_weighted':precision_score(y_test,pred,average='weighted',zero_division=0), 'recall_weighted':recall_score(y_test,pred,average='weighted',zero_division=0), 'f1_weighted':f1_score(y_test,pred,average='weighted',zero_division=0)}
    results.append(metrics)
    slug=name.lower().replace(' ','_')
    joblib.dump(model, MODEL_DIR/f'{slug}.joblib')
    (EVAL_DIR/f'{slug}_classification_report.txt').write_text(classification_report(y_test,pred,zero_division=0), encoding='utf-8')
    cm=confusion_matrix(y_test,pred,labels=classes)
    pd.DataFrame(cm,index=classes,columns=classes).to_csv(EVAL_DIR/f'{slug}_confusion_matrix.csv')

res=pd.DataFrame(results).sort_values(['f1_weighted','accuracy'],ascending=False)
res.to_csv(EVAL_DIR/'model_comparison.csv',index=False)
best_name=res.iloc[0]['model']; best_slug=best_name.lower().replace(' ','_')
shutil.copy2(MODEL_DIR/f'{best_slug}.joblib', MODEL_DIR/'best_model.joblib')

# 5-fold CV on the selected model family for an additional robustness check.
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
if best_name=='XGBoost':
    cv_model=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.08,subsample=0.9,colsample_bytree=0.9,objective='multi:softprob',num_class=len(classes),eval_metric='mlogloss',random_state=42,n_jobs=-1)
    cv_y=y.map(class_to_int)
    cv_scores=cross_val_score(cv_model,X,cv_y,cv=cv,scoring='accuracy',n_jobs=1)
elif best_name=='SVM':
    cv_scores=cross_val_score(models['SVM'],X,y,cv=cv,scoring='accuracy',n_jobs=1)
else:
    cv_scores=cross_val_score(models['Random Forest'],X,y,cv=cv,scoring='accuracy',n_jobs=1)

metadata={'features':FEATURES,'target':TARGET,'best_model':best_name,'test_accuracy':float(res.iloc[0]['accuracy']),'test_precision':float(res.iloc[0]['precision_weighted']),'test_recall':float(res.iloc[0]['recall_weighted']),'test_f1':float(res.iloc[0]['f1_weighted']),'cv_accuracy_mean':float(cv_scores.mean()),'cv_accuracy_std':float(cv_scores.std()),'classes':classes,'dataset_rows':int(len(df)),'dataset_note':'Crop Recommendation Dataset supplied for AgriNova. It contains 2,200 rows and 7 model features. Do not claim it represents a specified number of historical years or raw field observations without source verification.'}
(MODEL_DIR/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(res.to_string(index=False))
print(f'\nBest model: {best_name}')
print(f"Test accuracy: {metadata['test_accuracy']:.4%}")
print(f"5-fold CV accuracy: {metadata['cv_accuracy_mean']:.4%} +/- {metadata['cv_accuracy_std']:.4%}")
