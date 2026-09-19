
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT/"01_Data/raw/agriNova_prototype.csv"
dst = ROOT/"01_Data/processed/agriNova_processed.csv"

df = pd.read_csv(src)

# Basic validation/cleaning
df = df.drop_duplicates()
numeric_cols = df.select_dtypes("number").columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=["crop"])
df.to_csv(dst, index=False)

print(f"Saved {len(df)} rows to {dst}")
