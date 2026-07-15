# ============================================================
# Kapiti Seasonal: First Time-Series Visualisation
# Rainfall, SWC, NDVI, and GPP
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 1. Load analysis-ready file
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

INPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# Restrict to years with EC/met data
df = df[df["date"] <= "2024-12-31"].copy()

# ------------------------------------------------------------
# 2. Plot rainfall, SWC, NDVI, and GPP
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(16, 11),
    sharex=True
)

# Rainfall
axes[0].bar(
    df["date"],
    df["rain_fluxnet_mm_day"],
    width=1.0
)
axes[0].set_ylabel("Rainfall\n(mm day$^{-1}$)")
axes[0].set_title("Kapiti seasonal dynamics: rainfall, soil water, NDVI, and GPP")

# Soil water content
axes[1].plot(
    df["date"],
    df["SWC_shallow_7d_mean"],
    label="Shallow SWC, 7-day mean"
)
axes[1].plot(
    df["date"],
    df["SWC_middle_7d_mean"],
    label="Middle SWC, 7-day mean"
)
axes[1].plot(
    df["date"],
    df["SWC_deep"],
    label="Deep SWC, daily mean",
    alpha=0.7
)
axes[1].set_ylabel("SWC (%)")
axes[1].legend(loc="upper right")

# NDVI
axes[2].scatter(
    df["date"],
    df["NDVI"],
    s=25,
    label="NDVI observations"
)
axes[2].plot(
    df["date"],
    df["NDVI_rolling_3obs"],
    label="NDVI, 3-observation rolling mean"
)
axes[2].set_ylabel("NDVI")
axes[2].legend(loc="upper right")

# GPP
axes[3].plot(
    df["date"],
    df["GPP_7d_mean"],
    label="GPP, 7-day mean"
)
axes[3].set_ylabel("GPP")
axes[3].set_xlabel("Date")
axes[3].legend(loc="upper right")

plt.tight_layout()
plt.show()
