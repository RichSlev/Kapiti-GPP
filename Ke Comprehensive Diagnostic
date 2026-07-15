# ============================================================
# Kapiti Seasonal: Comprehensive Diagnostic Check
# For Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. Load file
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

INPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv"
OUTPUT_FOLDER = PROJECT_DIR / "Kapiti_Seasonal_diagnostics"

OUTPUT_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("\n============================================================")
print("KAPITI SEASONAL ANALYSIS-READY FILE DIAGNOSTIC")
print("============================================================")

print("\nFile loaded:")
print(INPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# ------------------------------------------------------------
# 2. Basic structure checks
# ------------------------------------------------------------

basic_checks = []

basic_checks.append({
    "check": "n_rows",
    "value": len(df),
    "status": "info"
})

basic_checks.append({
    "check": "n_columns",
    "value": df.shape[1],
    "status": "info"
})

basic_checks.append({
    "check": "duplicate_dates",
    "value": df["date"].duplicated().sum(),
    "status": "ok" if df["date"].duplicated().sum() == 0 else "warning"
})

full_date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
missing_dates = full_date_range.difference(df["date"])

basic_checks.append({
    "check": "missing_dates_in_daily_sequence",
    "value": len(missing_dates),
    "status": "ok" if len(missing_dates) == 0 else "warning"
})

basic_checks.append({
    "check": "first_missing_dates",
    "value": ", ".join([str(x.date()) for x in missing_dates[:20]]),
    "status": "info"
})

basic_checks_df = pd.DataFrame(basic_checks)

print("\nBasic checks:")
print(basic_checks_df)


# ------------------------------------------------------------
# 3. Column inventory
# ------------------------------------------------------------

column_inventory = []

for c in df.columns:
    column_inventory.append({
        "column": c,
        "dtype": str(df[c].dtype),
        "n_non_missing": df[c].notna().sum(),
        "n_missing": df[c].isna().sum(),
        "missing_percent": round(df[c].isna().mean() * 100, 2),
        "n_unique": df[c].nunique(dropna=True),
        "example_values": " | ".join(df[c].dropna().astype(str).unique()[:6])
    })

column_inventory_df = pd.DataFrame(column_inventory)

print("\nTop 20 most missing columns:")
print(
    column_inventory_df
    .sort_values("missing_percent", ascending=False)
    .head(20)
    [["column", "missing_percent", "n_non_missing"]]
)


# ------------------------------------------------------------
# 4. Core variable checks
# ------------------------------------------------------------

core_vars = [
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
    "RH",
    "Rg",
    "Ustar"
]

core_vars = [c for c in core_vars if c in df.columns]

core_summary = []

for c in core_vars:
    s = pd.to_numeric(df[c], errors="coerce")

    core_summary.append({
        "variable": c,
        "n_non_missing": s.notna().sum(),
        "missing_percent": round(s.isna().mean() * 100, 2),
        "min": s.min(),
        "p01": s.quantile(0.01),
        "p05": s.quantile(0.05),
        "median": s.median(),
        "mean": s.mean(),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
        "max": s.max(),
        "n_negative": int((s < 0).sum()),
        "n_zero": int((s == 0).sum())
    })

core_summary_df = pd.DataFrame(core_summary)

print("\nCore variable summary:")
print(core_summary_df)


# ------------------------------------------------------------
# 5. Expected range checks
# ------------------------------------------------------------

range_rules = {
    "rain_fluxnet_mm_day": {"min": 0, "max": 300},
    "rain_met_mm_day": {"min": 0, "max": 300},
    "SWC_shallow": {"min": 0, "max": 1},
    "SWC_middle": {"min": 0, "max": 1},
    "SWC_deep": {"min": 0, "max": 1},
    "NDVI": {"min": -0.2, "max": 1},
    "GPP": {"min": -50, "max": 50},
    "Tair": {"min": -10, "max": 50},
    "Tsoil": {"min": -10, "max": 60},
    "VPD": {"min": 0, "max": 10},
    "RH": {"min": 0, "max": 100},
    "Rg": {"min": 0, "max": 1500},
    "Ustar": {"min": 0, "max": 5},
}

range_checks = []

for c, rule in range_rules.items():
    if c not in df.columns:
        range_checks.append({
            "variable": c,
            "status": "missing_column",
            "n_below_expected": np.nan,
            "n_above_expected": np.nan,
            "min_observed": np.nan,
            "max_observed": np.nan
        })
        continue

    s = pd.to_numeric(df[c], errors="coerce")

    below = s < rule["min"]
    above = s > rule["max"]

    range_checks.append({
        "variable": c,
        "expected_min": rule["min"],
        "expected_max": rule["max"],
        "min_observed": s.min(),
        "max_observed": s.max(),
        "n_below_expected": int(below.sum()),
        "n_above_expected": int(above.sum()),
        "status": "ok" if below.sum() == 0 and above.sum() == 0 else "check"
    })

range_checks_df = pd.DataFrame(range_checks)

print("\nExpected range checks:")
print(range_checks_df)


# ------------------------------------------------------------
# 6. Date coverage by year
# ------------------------------------------------------------

coverage_by_year = (
    df
    .assign(year=df["date"].dt.year)
    .groupby("year")
    .agg(
        n_days=("date", "count"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        rain_days=("rain_fluxnet_mm_day", lambda x: x.notna().sum() if "rain_fluxnet_mm_day" in df.columns else np.nan),
        swc_days=("SWC_shallow", lambda x: x.notna().sum() if "SWC_shallow" in df.columns else np.nan),
        gpp_days=("GPP", lambda x: x.notna().sum() if "GPP" in df.columns else np.nan),
        ndvi_obs=("NDVI", lambda x: x.notna().sum() if "NDVI" in df.columns else np.nan),
    )
    .reset_index()
)

print("\nCoverage by year:")
print(coverage_by_year)


# ------------------------------------------------------------
# 7. Coverage by month
# ------------------------------------------------------------

df["year_month"] = df["date"].dt.to_period("M").astype(str)

monthly_coverage = (
    df
    .groupby("year_month")
    .agg(
        n_days=("date", "count"),
        rain_available=("rain_fluxnet_mm_day", lambda x: x.notna().sum() if "rain_fluxnet_mm_day" in df.columns else np.nan),
        swc_available=("SWC_shallow", lambda x: x.notna().sum() if "SWC_shallow" in df.columns else np.nan),
        gpp_available=("GPP", lambda x: x.notna().sum() if "GPP" in df.columns else np.nan),
        ndvi_available=("NDVI", lambda x: x.notna().sum() if "NDVI" in df.columns else np.nan),
        rain_total=("rain_fluxnet_mm_day", lambda x: pd.to_numeric(x, errors="coerce").sum(skipna=True) if "rain_fluxnet_mm_day" in df.columns else np.nan),
        swc_mean=("SWC_shallow", lambda x: pd.to_numeric(x, errors="coerce").mean(skipna=True) if "SWC_shallow" in df.columns else np.nan),
        gpp_mean=("GPP", lambda x: pd.to_numeric(x, errors="coerce").mean(skipna=True) if "GPP" in df.columns else np.nan),
        ndvi_mean=("NDVI", lambda x: pd.to_numeric(x, errors="coerce").mean(skipna=True) if "NDVI" in df.columns else np.nan),
    )
    .reset_index()
)

print("\nMonthly coverage preview:")
print(monthly_coverage.head(20))


# ------------------------------------------------------------
# 8. Rainfall consistency: fluxnet rain versus met rain
# ------------------------------------------------------------

rain_consistency = pd.DataFrame()

if "rain_fluxnet_mm_day" in df.columns and "rain_met_mm_day" in df.columns:
    rain_tmp = df[["date", "rain_fluxnet_mm_day", "rain_met_mm_day"]].copy()
    rain_tmp["rain_fluxnet_mm_day"] = pd.to_numeric(rain_tmp["rain_fluxnet_mm_day"], errors="coerce")
    rain_tmp["rain_met_mm_day"] = pd.to_numeric(rain_tmp["rain_met_mm_day"], errors="coerce")

    rain_tmp["rain_difference"] = rain_tmp["rain_fluxnet_mm_day"] - rain_tmp["rain_met_mm_day"]
    rain_tmp["abs_rain_difference"] = rain_tmp["rain_difference"].abs()

    rain_consistency = pd.DataFrame([{
        "n_overlap_days": rain_tmp.dropna(subset=["rain_fluxnet_mm_day", "rain_met_mm_day"]).shape[0],
        "correlation": rain_tmp[["rain_fluxnet_mm_day", "rain_met_mm_day"]].corr().iloc[0, 1],
        "mean_abs_difference": rain_tmp["abs_rain_difference"].mean(),
        "max_abs_difference": rain_tmp["abs_rain_difference"].max(),
        "n_days_abs_difference_gt_1mm": int((rain_tmp["abs_rain_difference"] > 1).sum()),
        "n_days_abs_difference_gt_5mm": int((rain_tmp["abs_rain_difference"] > 5).sum()),
    }])

    rain_large_diffs = rain_tmp.sort_values("abs_rain_difference", ascending=False).head(50)

    print("\nRainfall consistency:")
    print(rain_consistency)


# ------------------------------------------------------------
# 9. Simple ecological sanity checks
# ------------------------------------------------------------

sanity_checks = []

def add_sanity_check(name, condition, detail):
    sanity_checks.append({
        "check": name,
        "n_flagged": int(condition.sum()),
        "percent_flagged": round(condition.mean() * 100, 2),
        "detail": detail
    })

if "rain_fluxnet_mm_day" in df.columns:
    rain = pd.to_numeric(df["rain_fluxnet_mm_day"], errors="coerce")
    add_sanity_check(
        "negative_rainfall",
        rain < 0,
        "Rainfall should not be negative."
    )

if "SWC_shallow" in df.columns:
    swc = pd.to_numeric(df["SWC_shallow"], errors="coerce")
    add_sanity_check(
        "SWC_outside_0_1",
        (swc < 0) | (swc > 1),
        "SWC is expected to be volumetric fraction between 0 and 1. If values are percent, this rule needs changing."
    )

if "NDVI" in df.columns:
    ndvi = pd.to_numeric(df["NDVI"], errors="coerce")
    add_sanity_check(
        "NDVI_outside_possible_range",
        (ndvi < -1) | (ndvi > 1),
        "NDVI should always be between -1 and 1."
    )

if "GPP" in df.columns:
    gpp = pd.to_numeric(df["GPP"], errors="coerce")
    add_sanity_check(
        "very_negative_GPP",
        gpp < -5,
        "Large negative GPP may indicate sign convention or bad values."
    )

if "VPD" in df.columns:
    vpd = pd.to_numeric(df["VPD"], errors="coerce")
    add_sanity_check(
        "negative_VPD",
        vpd < 0,
        "VPD should not be negative."
    )

if "RH" in df.columns:
    rh = pd.to_numeric(df["RH"], errors="coerce")
    add_sanity_check(
        "RH_outside_0_100",
        (rh < 0) | (rh > 100),
        "Relative humidity should normally be between 0 and 100 percent."
    )

sanity_checks_df = pd.DataFrame(sanity_checks)

print("\nEcological sanity checks:")
print(sanity_checks_df)


# ------------------------------------------------------------
# 10. Lagged relationship previews
# ------------------------------------------------------------

lag_summary_rows = []

pairs = [
    ("rain_7d_sum", "SWC_shallow_7d_mean"),
    ("rain_14d_sum", "SWC_shallow_7d_mean"),
    ("SWC_shallow_7d_mean", "GPP_7d_mean"),
    ("SWC_shallow_7d_mean", "NDVI"),
    ("GPP_7d_mean", "NDVI"),
]

for x, y in pairs:
    if x not in df.columns or y not in df.columns:
        continue

    tmp = df[[x, y]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()

    if len(tmp) < 10:
        corr = np.nan
    else:
        corr = tmp[x].corr(tmp[y])

    lag_summary_rows.append({
        "x": x,
        "y": y,
        "n_overlap": len(tmp),
        "pearson_correlation_same_day": corr
    })

lag_summary_df = pd.DataFrame(lag_summary_rows)

print("\nPreliminary relationship checks:")
print(lag_summary_df)


# ------------------------------------------------------------
# 11. Export all diagnostics
# ------------------------------------------------------------

basic_checks_df.to_csv(OUTPUT_FOLDER / "01_basic_checks.csv", index=False)
column_inventory_df.to_csv(OUTPUT_FOLDER / "02_column_inventory.csv", index=False)
core_summary_df.to_csv(OUTPUT_FOLDER / "03_core_variable_summary.csv", index=False)
range_checks_df.to_csv(OUTPUT_FOLDER / "04_expected_range_checks.csv", index=False)
coverage_by_year.to_csv(OUTPUT_FOLDER / "05_coverage_by_year.csv", index=False)
monthly_coverage.to_csv(OUTPUT_FOLDER / "06_monthly_coverage.csv", index=False)

if not rain_consistency.empty:
    rain_consistency.to_csv(OUTPUT_FOLDER / "07_rainfall_consistency_summary.csv", index=False)
    rain_large_diffs.to_csv(OUTPUT_FOLDER / "08_largest_rainfall_differences.csv", index=False)

sanity_checks_df.to_csv(OUTPUT_FOLDER / "09_ecological_sanity_checks.csv", index=False)
lag_summary_df.to_csv(OUTPUT_FOLDER / "10_preliminary_relationship_checks.csv", index=False)

print("\n============================================================")
print("DIAGNOSTIC EXPORT COMPLETE")
print("============================================================")
print(f"Outputs saved to: {OUTPUT_FOLDER}")

print("\nPlease send me these key outputs:")
print("01_basic_checks.csv")
print("03_core_variable_summary.csv")
print("04_expected_range_checks.csv")
print("05_coverage_by_year.csv")
print("06_monthly_coverage.csv")
print("07_rainfall_consistency_summary.csv")
print("09_ecological_sanity_checks.csv")
print("10_preliminary_relationship_checks.csv")
