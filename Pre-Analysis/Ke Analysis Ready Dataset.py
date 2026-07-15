# ============================================================
# Kapiti Seasonal: Create Analysis-Ready Daily Dataset
# Collapses annual FLUXNET columns into continuous variables
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

INPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_MEGA_MERGED_DAILY.csv"
OUTPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv"
OUTPUT_DIAGNOSTICS = PROJECT_DIR / "Kapiti_Seasonal_analysis_ready_diagnostics.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])

# ------------------------------------------------------------
# Helper: collapse annual FLUXNET variables
# ------------------------------------------------------------

def collapse_fluxnet_variable(df, var_name):
    pattern = re.compile(rf"^fluxnet_\d{{4}}__{re.escape(var_name)}$")
    cols = [c for c in df.columns if pattern.match(c)]

    if not cols:
        return pd.Series(np.nan, index=df.index), []

    out = pd.Series(np.nan, index=df.index)

    for c in sorted(cols):
        out = out.combine_first(df[c])

    return out, cols


# ------------------------------------------------------------
# Build clean analysis dataframe
# ------------------------------------------------------------

analysis = pd.DataFrame()
analysis["date"] = df["date"]

analysis["year"] = df["date"].dt.year
analysis["month"] = df["date"].dt.month
analysis["day_of_year"] = df["date"].dt.dayofyear

# ------------------------------------------------------------
# Core seasonal variables
# ------------------------------------------------------------

core_fluxnet_vars = [
    "P_RAIN_1_1_1",
    "SWC_1_1_1",
    "SWC_2_1_1",
    "SWC_3_1_1",
    "FC_1_1_1",
    "TA_1_1_1",
    "TS_1_1_1",
    "TS_1_2_1",
    "TS_1_3_1",
    "TS_1_4_1",
    "TS_1_5_1",
    "TS_1_6_1",
    "VPD_1_1_1",
    "RH_1_1_1",
    "PA_1_1_1",
    "SW_IN_1_1_1",
    "LW_IN_1_1_1",
    "NETRAD_1_1_1",
    "PPFD_IN_1_1_1",
    "USTAR_1_1_1",
    "WS_1_1_1",
    "WD_1_1_1",
    "FETCH_70_1_1_1",
    "FETCH_80_1_1_1",
    "FETCH_90_1_1_1",
]

diagnostics = []

for var in core_fluxnet_vars:
    collapsed, source_cols = collapse_fluxnet_variable(df, var)
    analysis[var] = collapsed

    diagnostics.append({
        "clean_variable": var,
        "source": "annual_fluxnet_files",
        "n_source_columns": len(source_cols),
        "source_columns": " | ".join(source_cols),
        "n_non_missing": analysis[var].notna().sum(),
        "missing_percent": round(analysis[var].isna().mean() * 100, 2),
    })

# ------------------------------------------------------------
# Add footprint / partitioned GPP variables
# ------------------------------------------------------------

if "footprint__GPP_DT_U50" in df.columns:
    analysis["GPP_DT_U50"] = df["footprint__GPP_DT_U50"]
else:
    analysis["GPP_DT_U50"] = np.nan

diagnostics.append({
    "clean_variable": "GPP_DT_U50",
    "source": "footprint_file",
    "n_source_columns": int("footprint__GPP_DT_U50" in df.columns),
    "source_columns": "footprint__GPP_DT_U50" if "footprint__GPP_DT_U50" in df.columns else "",
    "n_non_missing": analysis["GPP_DT_U50"].notna().sum(),
    "missing_percent": round(analysis["GPP_DT_U50"].isna().mean() * 100, 2),
})

# ------------------------------------------------------------
# Add meteorology file variables
# ------------------------------------------------------------

met_vars = {
    "meteorology__Rain": "met_Rain",
    "meteorology__NEE": "met_NEE",
    "meteorology__LE": "met_LE",
    "meteorology__H": "met_H",
    "meteorology__Rg": "met_Rg",
    "meteorology__Tair": "met_Tair",
    "meteorology__Tsoil": "met_Tsoil",
    "meteorology__rH": "met_rH",
    "meteorology__VPD": "met_VPD",
    "meteorology__Ustar": "met_Ustar",
}

for source_col, clean_col in met_vars.items():
    if source_col in df.columns:
        analysis[clean_col] = df[source_col]
    else:
        analysis[clean_col] = np.nan

    diagnostics.append({
        "clean_variable": clean_col,
        "source": "meteorology_file",
        "n_source_columns": int(source_col in df.columns),
        "source_columns": source_col if source_col in df.columns else "",
        "n_non_missing": analysis[clean_col].notna().sum(),
        "missing_percent": round(analysis[clean_col].isna().mean() * 100, 2),
    })

# ------------------------------------------------------------
# Add NDVI variables
# ------------------------------------------------------------

ndvi_vars = {
    "ndvi__Center": "NDVI_center",
    "ndvi__Neigh3x3": "NDVI_neigh3x3",
    "ndvi__Weighted": "NDVI_weighted",
    "ndvi__Window_Mean_GPP": "NDVI_window_mean_GPP",
    "ndvi__N_pixels": "NDVI_n_pixels",
    "ndvi__Num_Timestamps": "NDVI_num_timestamps",
}

for source_col, clean_col in ndvi_vars.items():
    if source_col in df.columns:
        analysis[clean_col] = df[source_col]
    else:
        analysis[clean_col] = np.nan

    diagnostics.append({
        "clean_variable": clean_col,
        "source": "ndvi_file",
        "n_source_columns": int(source_col in df.columns),
        "source_columns": source_col if source_col in df.columns else "",
        "n_non_missing": analysis[clean_col].notna().sum(),
        "missing_percent": round(analysis[clean_col].isna().mean() * 100, 2),
    })

# ------------------------------------------------------------
# Convenience aliases for seasonal analysis
# ------------------------------------------------------------

analysis["rain_fluxnet_mm_day"] = analysis["P_RAIN_1_1_1"]
analysis["rain_met_mm_day"] = analysis["met_Rain"]

analysis["SWC_shallow"] = analysis["SWC_1_1_1"]
analysis["SWC_middle"] = analysis["SWC_2_1_1"]
analysis["SWC_deep"] = analysis["SWC_3_1_1"]

analysis["NEE_fluxnet"] = analysis["FC_1_1_1"]
analysis["GPP"] = analysis["GPP_DT_U50"]
analysis["NDVI"] = analysis["NDVI_weighted"]

analysis["Tair"] = analysis["TA_1_1_1"].combine_first(analysis["met_Tair"])
analysis["Tsoil"] = (
    analysis["TS_1_1_1"]
    .combine_first(analysis["TS_1_2_1"])
    .combine_first(analysis["TS_1_3_1"])
    .combine_first(analysis["met_Tsoil"])
)

analysis["VPD"] = analysis["VPD_1_1_1"].combine_first(analysis["met_VPD"])
analysis["RH"] = analysis["RH_1_1_1"].combine_first(analysis["met_rH"])
analysis["Rg"] = analysis["SW_IN_1_1_1"].combine_first(analysis["met_Rg"])
analysis["Ustar"] = analysis["USTAR_1_1_1"].combine_first(analysis["met_Ustar"])

# ------------------------------------------------------------
# Rolling variables useful for seasonal hysteresis
# ------------------------------------------------------------

analysis = analysis.sort_values("date").reset_index(drop=True)

analysis["rain_7d_sum"] = analysis["rain_fluxnet_mm_day"].rolling(7, min_periods=1).sum()
analysis["rain_14d_sum"] = analysis["rain_fluxnet_mm_day"].rolling(14, min_periods=1).sum()
analysis["rain_30d_sum"] = analysis["rain_fluxnet_mm_day"].rolling(30, min_periods=1).sum()

analysis["SWC_shallow_7d_mean"] = analysis["SWC_shallow"].rolling(7, min_periods=1).mean()
analysis["SWC_middle_7d_mean"] = analysis["SWC_middle"].rolling(7, min_periods=1).mean()
analysis["SWC_deep_7d_mean"] = analysis["SWC_deep"].rolling(7, min_periods=1).mean()

analysis["GPP_7d_mean"] = analysis["GPP"].rolling(7, min_periods=1).mean()
analysis["NDVI_rolling_3obs"] = analysis["NDVI"].rolling(3, min_periods=1).mean()

# ------------------------------------------------------------
# Nominal rain seasons
# ------------------------------------------------------------

def assign_nominal_rain_season(month):
    if month in [3, 4, 5]:
        return "long_rains"
    if month in [10, 11, 12]:
        return "short_rains"
    return "dry_or_transition"

analysis["nominal_rain_season"] = analysis["month"].apply(assign_nominal_rain_season)

# ------------------------------------------------------------
# Wetting / drying phase proxy
# Based on 7-day shallow SWC change
# ------------------------------------------------------------

analysis["delta_SWC_shallow_7d"] = analysis["SWC_shallow_7d_mean"].diff(7)

analysis["hydrological_phase"] = np.where(
    analysis["delta_SWC_shallow_7d"] > 0,
    "wetting_or_recharge",
    np.where(
        analysis["delta_SWC_shallow_7d"] < 0,
        "drying_or_drawdown",
        "stable_or_unclear"
    )
)

# ------------------------------------------------------------
# Export
# ------------------------------------------------------------

analysis.to_csv(OUTPUT_FILE, index=False)

diagnostics_df = pd.DataFrame(diagnostics)
diagnostics_df.to_csv(OUTPUT_DIAGNOSTICS, index=False)

print("\nDone.")
print(f"Analysis-ready file saved to:")
print(OUTPUT_FILE)

print(f"\nDiagnostics saved to:")
print(OUTPUT_DIAGNOSTICS)

print("\nShape:")
print(analysis.shape)

print("\nDate range:")
print(analysis["date"].min(), "to", analysis["date"].max())

print("\nCore variable missingness:")
core_check = [
    "rain_fluxnet_mm_day",
    "rain_met_mm_day",
    "SWC_shallow",
    "SWC_middle",
    "SWC_deep",
    "NDVI",
    "GPP",
    "NEE_fluxnet",
    "Tair",
    "Tsoil",
    "VPD",
    "Rg",
]

print(
    analysis[core_check]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values()
)
