
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"01_Data/raw/agriNova_prototype.csv"
OUT = ROOT/"02_EDA"
df = pd.read_csv(DATA)

print(df.info())
print(df.describe(include="all"))
print("\nMissing values:\n", df.isna().sum())
print("\nClass counts:\n", df["crop"].value_counts())

df["crop"].value_counts().plot(kind="bar", figsize=(10,5))
plt.title("Crop Class Distribution")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT/"crop_distribution.png")
plt.close()

numeric = df.select_dtypes("number")
corr = numeric.corr()
corr.to_csv(OUT/"correlation_matrix.csv")

plt.figure(figsize=(10,8))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
plt.colorbar()
plt.title("Numeric Feature Correlation")
plt.tight_layout()
plt.savefig(OUT/"correlation_matrix.png")
plt.close()
