from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '01_Data' / 'processed' / 'crop_recommendation_processed.csv'
OUT = ROOT / '02_EDA' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
print('\n=== AGRINOVA DATASET CHECK ===')
print('Rows:', len(df))
print('Columns:', list(df.columns))
print('Missing values:\n', df.isna().sum())
print('Duplicate rows:', df.duplicated().sum())
print('\nCrop distribution:\n', df['label'].value_counts().sort_index())
print('\nNumeric summary:\n', df.describe().round(3))
print('\n=== DONE ===')
df.describe().T.to_csv(OUT / 'numeric_summary.csv')
df['label'].value_counts().rename_axis('crop').reset_index(name='samples').to_csv(OUT / 'crop_distribution.csv', index=False)
