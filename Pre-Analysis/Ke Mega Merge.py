# ============================================================
# Kapiti Seasonal: Mega Merge All CSV Files
# Corrected version for footprint yyyy/mm/day/HH/MM
# and meteorology Year/DoY/Hour formats
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re
from functools import reduce

# ------------------------------------------------------------
# 1. Project folder
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

OUTPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_MEGA_MERGED_DAILY.csv"
DIAGNOSTIC_FILE = PROJECT_DIR / "Kapiti_Seasonal_merge_diagnostics.csv"

# ------------------------------------------------------------
# 2. Files to merge
# ------------------------------------------------------------

files = {
    "footprint": PROJECT_DIR / "Kapiti_KE_footprint_UTC.csv",
    "meteorology": PROJECT_DIR / "Kapiti_KE_Meteorology.csv",
    "ndvi": PROJECT_DIR / "Kapiti_KE_ndvi_timeseries.csv",
    "fluxnet_2018": PROJECT_DIR / "KE_kpt_FLUXNET_2018.csv",
    "fluxnet_2019": PROJECT_DIR / "KE_kpt_FLUXNET_2019.csv",
    "fluxnet_2020": PROJECT_DIR / "KE_kpt_FLUXNET_2020.csv",
    "fluxnet_2022": PROJECT_DIR / "KE_kpt_FLUXNET_2022.csv",
    "fluxnet_2023": PROJECT_DIR / "KE_kpt_FLUXNET_2023.csv",
    "fluxnet_2024": PROJECT_DIR / "KE_kpt_FLUXNET_2024.csv",
}

# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------

def clean_colname(x):
    x = str(x).strip()
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def standardise_missing_values(df):
    missing_codes = [
        -9999, -9999.0, -9999.00,
        -6999, -7999,
        -999, -999.0,
        "NA", "NaN", "nan", "NAN",
        "", " "
    ]
    return df.replace(missing_codes, np.nan)


def convert_numeric_safely(df):
    df = df.copy()

    for c in df.columns:
        if c in ["datetime", "date"]:
            continue

        converted = pd.to_numeric(df[c], errors="coerce")
        original_non_missing = df[c].notna().sum()

        if original_non_missing == 0:
            continue

        numeric_fraction = converted.notna().sum() / original_non_missing

        if numeric_fraction >= 0.5:
            df[c] = converted

    return df


def parse_fluxnet_timestamp(s):
    s_str = s.astype(str).str.strip()
    return pd.to_datetime(s_str, format="%Y%m%d%H%M", errors="coerce")


def detect_regular_datetime_column(df):
    candidates = [
        "TIMESTAMP_START",
        "TIMESTAMP",
        "Date",
        "date",
        "datetime",
        "DateTime"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        cl = c.lower()
        if "timestamp" in cl or "datetime" in cl:
            return c

    return None


def add_datetime(df, source_name):
    """
    Adds a datetime column using source-specific logic.
    """

    df = df.copy()

    if source_name == "footprint":
        required = ["yyyy", "mm", "day", "HH", "MM"]

        if all(c in df.columns for c in required):
            df["datetime"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(df["yyyy"], errors="coerce"),
                    month=pd.to_numeric(df["mm"], errors="coerce"),
                    day=pd.to_numeric(df["day"], errors="coerce"),
                    hour=pd.to_numeric(df["HH"], errors="coerce"),
                    minute=pd.to_numeric(df["MM"], errors="coerce"),
                ),
                errors="coerce"
            )
            return df, "yyyy/mm/day/HH/MM"

    if source_name == "meteorology":
        if all(c in df.columns for c in ["Year", "DoY", "Hour"]):
            year = pd.to_numeric(df["Year"], errors="coerce")
            doy = pd.to_numeric(df["DoY"], errors="coerce")
            hour_decimal = pd.to_numeric(df["Hour"], errors="coerce")

            base = pd.to_datetime(year.astype("Int64").astype(str), format="%Y", errors="coerce")
            df["datetime"] = (
                base
                + pd.to_timedelta(doy - 1, unit="D")
                + pd.to_timedelta(hour_decimal, unit="h")
            )
            return df, "Year/DoY/Hour"

    dt_col = detect_regular_datetime_column(df)

    if dt_col is None:
        return df, None

    if dt_col == "TIMESTAMP_START":
        df["datetime"] = parse_fluxnet_timestamp(df[dt_col])
    else:
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

    return df, dt_col


def aggregate_daily(df, source_name):
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]
    df = standardise_missing_values(df)

    df, dt_source = add_datetime(df, source_name)

    if dt_source is None:
        print(f"WARNING: no valid datetime found for {source_name}. Skipping.")
        print("Columns were:")
        print(list(df.columns[:40]))
        return None

    print(f"  Date source used: {dt_source}")

    df = df.dropna(subset=["datetime"])

    if df.empty:
        print(f"WARNING: datetime parsing failed for {source_name}. Skipping.")
        return None

    df["date"] = pd.to_datetime(df["datetime"].dt.date)

    df = convert_numeric_safely(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Do not aggregate source date component columns as data variables
    drop_from_numeric = [
        "yyyy", "mm", "day", "HH", "MM",
        "Year", "DoY", "Hour",
        "TIMESTAMP_START", "TIMESTAMP_END"
    ]

    numeric_cols = [c for c in numeric_cols if c not in drop_from_numeric]

    if not numeric_cols:
        print(f"WARNING: no numeric columns found for {source_name}. Skipping.")
        return None

    agg_dict = {}

    for c in numeric_cols:
        cl = c.lower()

        if any(key in cl for key in ["rain", "precip", "ppt", "p_rain"]):
            agg_dict[c] = "sum"
        else:
            agg_dict[c] = "mean"

    daily = df.groupby("date").agg(agg_dict).reset_index()

    rename_dict = {
        c: f"{source_name}__{c}"
        for c in daily.columns
        if c != "date"
    }

    daily = daily.rename(columns=rename_dict)

    return daily


# ------------------------------------------------------------
# 4. Read and aggregate files
# ------------------------------------------------------------

daily_dfs = []
diagnostics = []

for source_name, file_path in files.items():

    if not file_path.exists():
        print(f"Missing file: {file_path.name}")
        diagnostics.append({
            "source": source_name,
            "file": file_path.name,
            "status": "missing"
        })
        continue

    print(f"\nReading and aggregating: {file_path.name}")

    try:
        df = pd.read_csv(file_path, low_memory=False)
        daily = aggregate_daily(df, source_name)

        if daily is not None:
            print(
                f"  {source_name}: {daily.shape[0]} daily rows, "
                f"{daily.shape[1]} columns"
            )

            diagnostics.append({
                "source": source_name,
                "file": file_path.name,
                "status": "success",
                "daily_rows": daily.shape[0],
                "daily_columns": daily.shape[1],
                "date_min": daily["date"].min(),
                "date_max": daily["date"].max()
            })

            daily_dfs.append(daily)

        else:
            diagnostics.append({
                "source": source_name,
                "file": file_path.name,
                "status": "skipped"
            })

    except Exception as e:
        print(f"ERROR reading {file_path.name}: {e}")

        diagnostics.append({
            "source": source_name,
            "file": file_path.name,
            "status": f"error: {e}"
        })


if not daily_dfs:
    raise ValueError("No files were successfully read and aggregated.")


# ------------------------------------------------------------
# 5. Merge everything by date
# ------------------------------------------------------------

mega = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="outer"),
    daily_dfs
)

mega = mega.sort_values("date").reset_index(drop=True)
mega = mega.copy()


# ------------------------------------------------------------
# 6. Add seasonal time variables
# ------------------------------------------------------------

time_vars = pd.DataFrame({
    "year": mega["date"].dt.year,
    "month": mega["date"].dt.month,
    "day_of_year": mega["date"].dt.dayofyear
})

def assign_nominal_rain_season(month):
    if month in [3, 4, 5]:
        return "long_rains"
    if month in [10, 11, 12]:
        return "short_rains"
    return "dry_or_transition"

time_vars["nominal_rain_season"] = time_vars["month"].apply(assign_nominal_rain_season)

mega = pd.concat([mega, time_vars], axis=1)


# ------------------------------------------------------------
# 7. Export
# ------------------------------------------------------------

mega.to_csv(OUTPUT_FILE, index=False)

diagnostics_df = pd.DataFrame(diagnostics)
diagnostics_df.to_csv(DIAGNOSTIC_FILE, index=False)

print("\nDone.")
print(f"Mega merged daily file saved to:")
print(OUTPUT_FILE)

print(f"\nDiagnostics saved to:")
print(DIAGNOSTIC_FILE)

print("\nShape:")
print(mega.shape)

print("\nDate range:")
print(mega["date"].min(), "to", mega["date"].max())

print("\nFirst 60 columns:")
print(list(mega.columns[:60]))
