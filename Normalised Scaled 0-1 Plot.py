# ============================================================
# Kapiti Seasonal: Normalised Seasonal Relationship Plot
# SWC, NDVI, and GPP on comparable 0-1 scale
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")
INPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df = df[df["date"] <= "2024-12-31"].copy()

def minmax_scale(series):
    return (series - series.min()) / (series.max() - series.min())

df["SWC_shallow_scaled"] = minmax_scale(df["SWC_shallow_7d_mean"])
df["GPP_scaled"] = minmax_scale(df["GPP_7d_mean"])
df["NDVI_scaled"] = minmax_scale(df["NDVI_rolling_3obs"])

plt.figure(figsize=(16, 6))

plt.plot(
    df["date"],
    df["SWC_shallow_scaled"],
    label="Shallow SWC, scaled"
)

plt.plot(
    df["date"],
    df["GPP_scaled"],
    label="GPP, scaled"
)

plt.scatter(
    df["date"],
    df["NDVI_scaled"],
    s=25,
    label="NDVI, scaled"
)

plt.ylabel("Scaled value, 0-1")
plt.xlabel("Date")
plt.title("Kapiti seasonal coupling of SWC, NDVI, and GPP")
plt.legend()
plt.tight_layout()
plt.show()
