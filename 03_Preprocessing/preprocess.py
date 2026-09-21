from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / '01_Data' / 'raw' / 'Crop_recommendation.csv'
DST = ROOT / '01_Data' / 'processed' / 'crop_recommendation_processed.csv'

FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
TARGET = 'label'

df = pd.read_csv(SRC)
required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

df = df[required].copy()
df = df.drop_duplicates().dropna(subset=required)
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=FEATURES)
DST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DST, index=False)
print(f'Rows: {len(df)}')
print(f'Saved: {DST}')
