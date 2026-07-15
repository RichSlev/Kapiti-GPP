# ============================================================
# Kapiti Seasonal: Dataset Column Explorer + Data Dictionary
# Works with CSV, XLSX, and XLS files
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ------------------------------------------------------------
# 1. Set project folder
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

OUTPUT_EXCEL = PROJECT_DIR / "Kapiti_Seasonal_column_exploration.xlsx"
OUTPUT_CSV = PROJECT_DIR / "Kapiti_Seasonal_column_dictionary.csv"

# ------------------------------------------------------------
# 2. Variable interpretation helper
# ------------------------------------------------------------

def guess_variable_meaning(col):
    c = str(col).lower()

    meanings = {
        "timestamp": "Datetime/timestamp variable, likely used for temporal alignment.",
        "datetime": "Datetime/timestamp variable, likely used for temporal alignment.",
        "date": "Date variable.",
        "time": "Time variable.",
        "precip": "Precipitation or rainfall variable.",
        "rain": "Rainfall or rain-event related variable.",
        "p": "Possible precipitation or pressure variable. Needs checking.",
        "swc": "Soil water content variable.",
        "vwc": "Volumetric water content, usually soil moisture.",
        "soil_water": "Soil water content variable.",
        "ndvi": "Normalized Difference Vegetation Index, proxy for vegetation greenness.",
        "evi": "Enhanced Vegetation Index, proxy for vegetation greenness/productivity.",
        "gpp": "Gross Primary Productivity, ecosystem carbon uptake.",
        "nee": "Net Ecosystem Exchange.",
        "reco": "Ecosystem respiration.",
        "fc": "CO2 flux variable, often equivalent to NEE after conventions/filtering.",
        "le": "Latent heat flux.",
        "h": "Sensible heat flux or height-related variable. Needs checking.",
        "ta": "Air temperature.",
        "air_temperature": "Air temperature.",
        "ts": "Soil temperature.",
        "soil_temperature": "Soil temperature.",
        "rh": "Relative humidity.",
        "vpd": "Vapour pressure deficit.",
        "rg": "Incoming shortwave/global radiation.",
        "sw_in": "Incoming shortwave radiation.",
        "lw_in": "Incoming longwave radiation.",
        "par": "Photosynthetically active radiation.",
        "ustar": "Friction velocity, turbulence metric.",
        "u_star": "Friction velocity, turbulence metric.",
        "wd": "Wind direction.",
        "ws": "Wind speed.",
        "co2": "CO2 concentration or CO2 flux-related variable.",
        "qc": "Quality-control flag.",
        "flag": "Quality-control or filtering flag.",
        "footprint": "Flux footprint or source-area contribution variable.",
        "fetch": "Flux footprint or fetch-related variable.",
        "lat": "Latitude coordinate.",
        "lon": "Longitude coordinate.",
        "longitude": "Longitude coordinate.",
        "latitude": "Latitude coordinate.",
    }

    matches = []

    for key, meaning in meanings.items():
        if key in c:
            matches.append(meaning)

    if matches:
        return " ".join(sorted(set(matches)))

    return "Meaning unclear from name alone. Needs manual interpretation."


# ------------------------------------------------------------
# 3. Safe file reader
# ------------------------------------------------------------

def read_file_safely(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        encodings = ["utf-8", "latin1", "cp1252"]

        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, low_memory=False)
            except Exception:
                continue

        raise ValueError(f"Could not read CSV file: {file_path.name}")

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file type: {file_path.name}")


# ------------------------------------------------------------
# 4. Column-level summary
# ------------------------------------------------------------

def summarise_column(df, file_name, col):
    s = df[col]
    non_missing = s.dropna()

    result = {
        "file": file_name,
        "column": col,
        "guessed_meaning": guess_variable_meaning(col),
        "dtype_raw": str(s.dtype),
        "n_rows": len(s),
        "n_missing": int(s.isna().sum()),
        "missing_percent": round(float(s.isna().mean() * 100), 2),
        "n_unique": int(s.nunique(dropna=True)),
        "example_values": "",
        "numeric_min": np.nan,
        "numeric_max": np.nan,
        "numeric_mean": np.nan,
        "numeric_median": np.nan,
        "datetime_min": "",
        "datetime_max": "",
    }

    if len(non_missing) > 0:
        examples = non_missing.astype(str).unique()[:8]
        result["example_values"] = " | ".join(examples)

    numeric = pd.to_numeric(s, errors="coerce")

    if numeric.notna().sum() > 0:
        result["numeric_min"] = numeric.min()
        result["numeric_max"] = numeric.max()
        result["numeric_mean"] = numeric.mean()
        result["numeric_median"] = numeric.median()

    if any(x in str(col).lower() for x in ["date", "time", "timestamp", "datetime"]):
        dt = pd.to_datetime(s, errors="coerce")

        if dt.notna().sum() > 0:
            result["datetime_min"] = str(dt.min())
            result["datetime_max"] = str(dt.max())

    return result


# ------------------------------------------------------------
# 5. File-level summary
# ------------------------------------------------------------

def summarise_file(df, file_path):
    return {
        "file": file_path.name,
        "file_type": file_path.suffix.lower(),
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": ", ".join([str(c) for c in df.columns]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_000_000, 2),
    }


# ------------------------------------------------------------
# 6. Find all usable data files
# ------------------------------------------------------------

data_files = (
    sorted(PROJECT_DIR.glob("*.csv")) +
    sorted(PROJECT_DIR.glob("*.xlsx")) +
    sorted(PROJECT_DIR.glob("*.xls"))
)

if not data_files:
    raise FileNotFoundError(f"No CSV, XLSX, or XLS files found in {PROJECT_DIR}")

print("Files found:")
for f in data_files:
    print(" -", f.name)


# ------------------------------------------------------------
# 7. Main exploration
# ------------------------------------------------------------

file_summaries = []
column_summaries = []
preview_tables = {}
loaded_columns = {}

for file_path in data_files:
    print(f"\nReading: {file_path.name}")

    try:
        df = read_file_safely(file_path)
    except Exception as e:
        print(f"Could not read {file_path.name}: {e}")
        continue

    df.columns = [str(c).strip() for c in df.columns]

    file_summaries.append(summarise_file(df, file_path))
    loaded_columns[file_path.name] = list(df.columns)

    for col in df.columns:
        column_summaries.append(summarise_column(df, file_path.name, col))

    preview_name = re.sub(r"[^A-Za-z0-9_]", "_", file_path.stem)[:31]
    preview_tables[preview_name] = df.head(10)


file_summary_df = pd.DataFrame(file_summaries)
column_summary_df = pd.DataFrame(column_summaries)

if column_summary_df.empty:
    raise ValueError("No readable columns were found in the input files.")


# ------------------------------------------------------------
# 8. Cross-file column presence matrix
# ------------------------------------------------------------

all_columns = sorted(column_summary_df["column"].unique())

presence_rows = []

for col in all_columns:
    row = {"column": col}

    for file_name, cols in loaded_columns.items():
        row[file_name] = col in cols

    presence_rows.append(row)

presence_df = pd.DataFrame(presence_rows)


# ------------------------------------------------------------
# 9. Seasonal-analysis relevance table
# ------------------------------------------------------------

def classify_relevance(col):
    c = str(col).lower()

    if any(x in c for x in ["precip", "rain", "ppt"]):
        return "Precipitation / rainfall"
    if any(x in c for x in ["swc", "vwc", "soil_water", "wfps"]):
        return "Soil water content"
    if "ndvi" in c:
        return "NDVI / greenness"
    if "gpp" in c:
        return "GPP / productivity"
    if any(x in c for x in ["nee", "fc"]):
        return "Carbon flux / NEE"
    if "reco" in c:
        return "Respiration"
    if any(x in c for x in ["date", "time", "timestamp", "datetime"]):
        return "Time variable"
    if any(x in c for x in ["ta", "ts", "temp"]):
        return "Temperature"
    if any(x in c for x in ["vpd", "rh"]):
        return "Atmospheric moisture"
    if any(x in c for x in ["rg", "par", "sw_in", "radiation"]):
        return "Radiation"
    if any(x in c for x in ["qc", "flag"]):
        return "Quality control"
    if any(x in c for x in ["footprint", "fetch"]):
        return "Footprint"
    return "Other / unclear"

seasonal_relevance_df = column_summary_df.copy()
seasonal_relevance_df["seasonal_relevance_group"] = seasonal_relevance_df["column"].apply(classify_relevance)

seasonal_relevance_df = seasonal_relevance_df[
    [
        "file",
        "column",
        "seasonal_relevance_group",
        "guessed_meaning",
        "dtype_raw",
        "n_missing",
        "missing_percent",
        "n_unique",
        "numeric_min",
        "numeric_max",
        "numeric_mean",
        "example_values",
    ]
]


# ------------------------------------------------------------
# 10. Export outputs
# ------------------------------------------------------------

OUTPUT_FOLDER = PROJECT_DIR / "Kapiti_Seasonal_column_exploration_outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

file_summary_df.to_csv(OUTPUT_FOLDER / "file_summary.csv", index=False)
column_summary_df.to_csv(OUTPUT_FOLDER / "column_dictionary.csv", index=False)
seasonal_relevance_df.to_csv(OUTPUT_FOLDER / "seasonal_relevance.csv", index=False)
presence_df.to_csv(OUTPUT_FOLDER / "column_presence.csv", index=False)

for sheet_name, preview_df in preview_tables.items():
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", sheet_name)[:50]
    preview_df.to_csv(OUTPUT_FOLDER / f"preview_{safe_name}.csv", index=False)

print("\nDone.")
print(f"Outputs saved to folder: {OUTPUT_FOLDER}")
print("\nPlease send me these files:")
print(" - file_summary.csv")
print(" - column_dictionary.csv")
print(" - seasonal_relevance.csv")
print(" - column_presence.csv")
