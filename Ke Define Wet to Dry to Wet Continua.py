# ============================================================
# Kapiti Seasonal: Define Wet to Dry to Wet Continua
# Gap-aware improved version:
# 1. Keeps hydrological wet-season scaffold
# 2. Reindexes to a complete daily calendar
# 3. Separates observed dry gaps from data gaps
# 4. Bridges GPP seasons across defensible equipment/data gaps
# 5. Separates broad GPP seasons, GPP pulses, and true season breaks
# 6. Adds GPP season data-quality diagnostics
# 7. Exports daily, cycle-level, season-level, pulse-level, and data-gap outputs
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

INPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv"

OUTPUT_DAILY = PROJECT_DIR / "Kapiti_Seasonal_DAILY_WITH_GPP_SEASONS_GAP_AWARE.csv"
OUTPUT_CYCLES = PROJECT_DIR / "Kapiti_Seasonal_CYCLE_SUMMARY_GPP_SEASONS_GAP_AWARE.csv"
OUTPUT_GAPS = PROJECT_DIR / "Kapiti_Seasonal_DATA_GAP_SUMMARY.csv"
OUTPUT_GPP_SEASONS = PROJECT_DIR / "Kapiti_Seasonal_GPP_SEASON_SUMMARY.csv"
OUTPUT_GPP_PULSES = PROJECT_DIR / "Kapiti_Seasonal_GPP_PULSE_SUMMARY.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Keep EC-relevant period only
df = df[df["date"] <= "2024-12-31"].copy()

# Make sure key variables are numeric
for col in [
    "rain_fluxnet_mm_day",
    "SWC_shallow",
    "SWC_middle",
    "SWC_deep",
    "GPP",
    "NDVI",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 1B. Reindex to complete daily calendar and mark missing data
# ------------------------------------------------------------

# This is critical for equipment downtime:
# a missing month must be represented as missing daily rows,
# not silently skipped by rolling windows or seasonal blocks.
full_dates = pd.DataFrame({
    "date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")
})

df["_observed_row"] = True

df = (
    full_dates
    .merge(df, on="date", how="left")
    .sort_values("date")
    .reset_index(drop=True)
)

df["_observed_row"] = df["_observed_row"].fillna(False)

swc_cols = [c for c in ["SWC_shallow", "SWC_middle", "SWC_deep"] if c in df.columns]

df["has_rain"] = df["rain_fluxnet_mm_day"].notna()
df["has_GPP"] = df["GPP"].notna()
df["has_NDVI"] = df["NDVI"].notna()

if len(swc_cols) > 0:
    df["has_any_SWC"] = df[swc_cols].notna().any(axis=1)
else:
    df["has_any_SWC"] = False

# Core ecological-response data are GPP and SWC.
# Rain may come from a different stream, so it is tracked separately.
df["is_gpp_data_gap"] = ~df["has_GPP"]
df["is_swc_data_gap"] = ~df["has_any_SWC"]
df["is_response_data_gap"] = (~df["has_GPP"]) & (~df["has_any_SWC"])

# A broader gap flag, useful for diagnostics.
df["has_any_core_data"] = df["has_rain"] | df["has_GPP"] | df["has_any_SWC"]
df["is_core_data_gap"] = ~df["has_any_core_data"]


def make_gap_summary(df_in, gap_col, block_col, prefix):
    """
    Identify continuous data-gap blocks for a boolean gap column.
    """
    out = df_in.copy()
    out[block_col] = (out[gap_col] != out[gap_col].shift()).cumsum()

    gap_summary = (
        out[out[gap_col]]
        .groupby(block_col)
        .agg(
            gap_start=("date", "min"),
            gap_end=("date", "max"),
            gap_duration_days=("date", "count"),
            missing_GPP_days=("has_GPP", lambda x: (~x).sum()),
            missing_SWC_days=("has_any_SWC", lambda x: (~x).sum()),
            missing_rain_days=("has_rain", lambda x: (~x).sum()),
            missing_core_days=("has_any_core_data", lambda x: (~x).sum()),
        )
        .reset_index()
    )

    gap_summary["gap_type"] = prefix

    return out[block_col], gap_summary


df["response_gap_block"], response_gap_summary = make_gap_summary(
    df,
    gap_col="is_response_data_gap",
    block_col="response_gap_block",
    prefix="response_data_gap"
)

df["core_gap_block"], core_gap_summary = make_gap_summary(
    df,
    gap_col="is_core_data_gap",
    block_col="core_gap_block",
    prefix="core_data_gap"
)

gap_summary_df = pd.concat(
    [response_gap_summary, core_gap_summary],
    ignore_index=True
)

# ------------------------------------------------------------
# 2. Prepare smoothed variables
# ------------------------------------------------------------

# Rainfall sums should remain trailing because they represent accumulated recent rain.
# min_periods is used so missing data do not get treated as zero rainfall.
df["rain_7d_sum"] = df["rain_fluxnet_mm_day"].rolling(7, min_periods=4).sum()
df["rain_14d_sum"] = df["rain_fluxnet_mm_day"].rolling(14, min_periods=8).sum()

# Soil water smoothing
df["SWC_shallow_7d"] = df["SWC_shallow"].rolling(7, min_periods=3).mean()
df["SWC_middle_7d"] = df["SWC_middle"].rolling(7, min_periods=3).mean()
df["SWC_deep_7d"] = df["SWC_deep"].rolling(7, min_periods=3).mean()

# GPP smoothing
# GPP_7d keeps sharper seasonal peaks.
# GPP_14d gives a more stable seasonal envelope for detecting productive periods.
df["GPP_7d"] = df["GPP"].rolling(7, min_periods=3, center=True).mean()
df["GPP_14d"] = df["GPP"].rolling(14, min_periods=5, center=True).mean()

# NDVI is sparse, so keep observed NDVI and a lightly interpolated version
df["NDVI_obs"] = df["NDVI"]

df["NDVI_interp"] = df["NDVI"].interpolate(
    method="linear",
    limit=30,
    limit_direction="both"
)

df["NDVI_interp_7d"] = df["NDVI_interp"].rolling(
    7,
    min_periods=3,
    center=True
).mean()

# ------------------------------------------------------------
# 3. Detect hydrological wet-season candidate periods
# ------------------------------------------------------------

RAIN_THRESHOLD_14D = 20
SWC_RECHARGE_THRESHOLD = 2.0

# Dryland-specific tolerance:
# observed dry interruptions can be part of the same wet season,
# but data gaps must be handled separately.
MAX_OBSERVED_DRY_GAP_DAYS = 30
MAX_DATA_GAP_BRIDGE_DAYS = 45

df["delta_SWC_7d"] = df["SWC_shallow_7d"].diff(7)

df["wet_signal"] = (
    (df["rain_14d_sum"] >= RAIN_THRESHOLD_14D) |
    (df["delta_SWC_7d"] >= SWC_RECHARGE_THRESHOLD)
).astype(float)

# Do not treat missing response data as an observed dry signal.
# These rows are allowed to be bridged later if before/after evidence supports it.
df.loc[df["is_response_data_gap"], "wet_signal"] = np.nan

# Smooth wet signal to avoid short gaps breaking a season.
df["wet_signal_rolling"] = (
    df["wet_signal"]
    .astype(float)
    .rolling(7, min_periods=3, center=True)
    .mean()
)

df["is_wet_phase_raw"] = df["wet_signal_rolling"] >= 0.3
df.loc[df["wet_signal_rolling"].isna(), "is_wet_phase_raw"] = False

# ------------------------------------------------------------
# 4. Clean wet/dry phase labels with gap-aware bridging
# ------------------------------------------------------------

df["phase_block"] = (
    df["is_wet_phase_raw"] != df["is_wet_phase_raw"].shift()
).cumsum()

block_summary = (
    df.groupby("phase_block")
    .agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_days=("date", "count"),
        is_wet=("is_wet_phase_raw", "first"),
        response_gap_days=("is_response_data_gap", "sum"),
        core_gap_days=("is_core_data_gap", "sum"),
        rain_total=("rain_fluxnet_mm_day", "sum"),
        swc_gain=("SWC_shallow_7d", lambda x: x.max() - x.min())
    )
    .reset_index()
)

block_summary["response_gap_fraction"] = (
    block_summary["response_gap_days"] / block_summary["n_days"]
)

# Previous and next block state allow us to bridge only interruptions
# that sit between wet phases.
block_summary["prev_is_wet"] = block_summary["is_wet"].shift(1)
block_summary["next_is_wet"] = block_summary["is_wet"].shift(-1)

short_observed_dry_blocks = block_summary[
    (block_summary["is_wet"] == False) &
    (block_summary["n_days"] <= MAX_OBSERVED_DRY_GAP_DAYS) &
    (block_summary["response_gap_fraction"] < 0.5) &
    (block_summary["prev_is_wet"] == True) &
    (block_summary["next_is_wet"] == True)
]["phase_block"]

bridgeable_data_gap_blocks = block_summary[
    (block_summary["is_wet"] == False) &
    (block_summary["n_days"] <= MAX_DATA_GAP_BRIDGE_DAYS) &
    (block_summary["response_gap_fraction"] >= 0.5) &
    (block_summary["prev_is_wet"] == True) &
    (block_summary["next_is_wet"] == True)
]["phase_block"]

df["is_wet_phase"] = df["is_wet_phase_raw"]

df["hydro_observed_dry_gap_bridge"] = False
df["hydro_data_gap_bridge"] = False

df.loc[df["phase_block"].isin(short_observed_dry_blocks), "is_wet_phase"] = True
df.loc[df["phase_block"].isin(short_observed_dry_blocks), "hydro_observed_dry_gap_bridge"] = True

df.loc[df["phase_block"].isin(bridgeable_data_gap_blocks), "is_wet_phase"] = True
df.loc[df["phase_block"].isin(bridgeable_data_gap_blocks), "hydro_data_gap_bridge"] = True

# Recalculate cleaned blocks
df["clean_phase_block"] = (
    df["is_wet_phase"] != df["is_wet_phase"].shift()
).cumsum()

clean_blocks = (
    df.groupby("clean_phase_block")
    .agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_days=("date", "count"),
        is_wet=("is_wet_phase", "first"),
        rain_total=("rain_fluxnet_mm_day", "sum"),
        swc_gain=("SWC_shallow_7d", lambda x: x.max() - x.min()),
        response_gap_days=("is_response_data_gap", "sum")
    )
    .reset_index()
)

# Remove tiny wet periods that are likely noise.
# This check uses observed rain and SWC evidence only.
valid_wet_blocks = clean_blocks[
    (clean_blocks["is_wet"] == True) &
    (clean_blocks["n_days"] >= 20) &
    (
        (clean_blocks["rain_total"] >= 30) |
        (clean_blocks["swc_gain"] >= 3)
    )
]["clean_phase_block"]

df["is_wet_phase"] = df["clean_phase_block"].isin(valid_wet_blocks)

# Final clean blocks
df["phase_block_final"] = (
    df["is_wet_phase"] != df["is_wet_phase"].shift()
).cumsum()

# ------------------------------------------------------------
# 5. Define wet to dry to wet continua
# ------------------------------------------------------------

wet_blocks = (
    df[df["is_wet_phase"]]
    .groupby("phase_block_final")
    .agg(
        wet_start=("date", "min"),
        wet_end=("date", "max"),
        hydro_data_gap_bridge_days=("hydro_data_gap_bridge", "sum"),
        hydro_observed_dry_gap_bridge_days=("hydro_observed_dry_gap_bridge", "sum")
    )
    .reset_index()
)

wet_blocks = wet_blocks.sort_values("wet_start").reset_index(drop=True)
wet_blocks["cycle_id"] = np.arange(1, len(wet_blocks) + 1)

df["cycle_id"] = np.nan
df["cycle_position"] = np.nan

for i, row in wet_blocks.iterrows():
    cycle_id = row["cycle_id"]
    cycle_start = row["wet_start"]

    if i < len(wet_blocks) - 1:
        cycle_end = wet_blocks.loc[i + 1, "wet_start"] - pd.Timedelta(days=1)
    else:
        cycle_end = df["date"].max()

    mask = (df["date"] >= cycle_start) & (df["date"] <= cycle_end)

    df.loc[mask, "cycle_id"] = cycle_id
    df.loc[mask, "cycle_position"] = (
        df.loc[mask, "date"] - cycle_start
    ).dt.days

df["hydro_cycle_phase"] = np.where(
    df["is_wet_phase"],
    "wetting_or_wet",
    "drying_or_dry"
)

# ------------------------------------------------------------
# 6. GPP response detection helper functions
# ------------------------------------------------------------

BASELINE_LOOKBACK_DAYS = 45
MIN_BASELINE_N = 7

MIN_SUSTAINED_GPP_DAYS = 7
MIN_SUSTAINED_END_DAYS = 10

MIN_GPP_RESPONSE_DAYS = 30
DEFAULT_GPP_RESPONSE_DAYS = 120
MAX_GPP_RESPONSE_DAYS = 240
POST_WET_BUFFER_DAYS = 90

FIRST_PEAK_SEARCH_DAYS = 60

GPP_RESPONSE_FRAC = 0.25
NOISE_MULTIPLIER = 1.5
GPP_MIN_ABS_INCREASE = 0.0

# GPP-specific bridge rules.
# These are intentionally separate from the hydrological wet/dry bridge rules.
MAX_OBSERVED_GPP_DIP_BRIDGE_DAYS = 30
MAX_GPP_DATA_GAP_BRIDGE_DAYS = 45


def robust_mad(x):
    """Robust estimate of spread."""
    x = pd.Series(x).dropna().astype(float)

    if len(x) < 3:
        return np.nan

    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def find_true_runs(mask):
    """
    Return start and end positions of True runs in a boolean array.
    Positions are relative to the supplied array, not dataframe index values.
    """
    arr = np.asarray(pd.Series(mask).fillna(False).astype(bool))

    if arr.size == 0:
        return []

    padded = np.r_[False, arr, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])

    starts = changes[0::2]
    ends = changes[1::2] - 1

    return list(zip(starts, ends))


def first_sustained_run(mask, min_days):
    """Find first True run lasting at least min_days."""
    runs = find_true_runs(mask)

    valid_runs = [
        (start, end)
        for start, end in runs
        if (end - start + 1) >= min_days
    ]

    if len(valid_runs) == 0:
        return None

    return valid_runs[0]


def peak_in_window(sub, variable, start_date, end_date):
    """Return peak value, date, and n valid observations inside a date window."""
    w = sub[
        (sub["date"] >= start_date) &
        (sub["date"] <= end_date)
    ].dropna(subset=[variable]).copy()

    if w.empty:
        return {
            "peak": np.nan,
            "peak_date": pd.NaT,
            "n_valid": 0
        }

    idx = w[variable].idxmax()

    return {
        "peak": w.loc[idx, variable],
        "peak_date": w.loc[idx, "date"],
        "n_valid": len(w)
    }


def cumulative_in_window(sub, variable, start_date, n_days):
    """Cumulative value over n_days from start_date."""
    end_date = start_date + pd.Timedelta(days=n_days - 1)

    w = sub[
        (sub["date"] >= start_date) &
        (sub["date"] <= end_date)
    ].copy()

    return {
        "sum": w[variable].sum(skipna=True),
        "mean": w[variable].mean(skipna=True),
        "n_valid": w[variable].notna().sum(),
        "missing_fraction": w[variable].isna().mean() if len(w) > 0 else np.nan
    }


def get_gpp_baseline(df_all, sub, wet_start):
    """
    Prefer antecedent GPP baseline before wet onset.
    If unavailable, use the lower 20th percentile within the cycle.
    """
    antecedent_start = wet_start - pd.Timedelta(days=BASELINE_LOOKBACK_DAYS)

    antecedent = df_all[
        (df_all["date"] >= antecedent_start) &
        (df_all["date"] < wet_start)
    ]["GPP_14d"].dropna()

    if len(antecedent) >= MIN_BASELINE_N:
        baseline = antecedent.median()
        noise = robust_mad(antecedent)
        source = "antecedent_45d_median"
    else:
        cycle_gpp = sub["GPP_14d"].dropna()

        if len(cycle_gpp) == 0:
            return np.nan, np.nan, "unavailable"

        baseline = cycle_gpp.quantile(0.20)

        lower_half = cycle_gpp[cycle_gpp <= cycle_gpp.quantile(0.50)]
        noise = robust_mad(lower_half)

        source = "cycle_20th_percentile"

    if not np.isfinite(noise):
        noise = robust_mad(sub["GPP_14d"])

    if not np.isfinite(noise):
        noise = 0.0

    return baseline, noise, source


def bridge_gpp_active_signal(period_df):
    """
    Bridge GPP active/inactive labels across:
    1. short observed GPP dips
    2. GPP data gaps

    This preserves season continuity without interpolating missing GPP values.
    """
    p = period_df.copy()

    p["gpp_active_bridged"] = p["gpp_active_raw"].fillna(False).astype(bool)
    p["gpp_observed_dip_bridge"] = False
    p["gpp_data_gap_bridge"] = False

    p["gpp_active_block"] = (
        p["gpp_active_bridged"] != p["gpp_active_bridged"].shift()
    ).cumsum()

    blocks = (
        p.groupby("gpp_active_block")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_days=("date", "count"),
            is_active=("gpp_active_bridged", "first"),
            gpp_gap_days=("is_gpp_data_gap", "sum"),
            response_gap_days=("is_response_data_gap", "sum"),
            rain_total=("rain_fluxnet_mm_day", "sum"),
            swc_gain=("SWC_shallow_7d", lambda x: x.max() - x.min()),
        )
        .reset_index()
    )

    blocks["gpp_gap_fraction"] = blocks["gpp_gap_days"] / blocks["n_days"]
    blocks["prev_is_active"] = blocks["is_active"].shift(1)
    blocks["next_is_active"] = blocks["is_active"].shift(-1)

    # Observed dips are allowed if they are short and bounded by active GPP.
    observed_dip_blocks = blocks[
        (blocks["is_active"] == False) &
        (blocks["n_days"] <= MAX_OBSERVED_GPP_DIP_BRIDGE_DAYS) &
        (blocks["gpp_gap_fraction"] < 0.5) &
        (blocks["prev_is_active"] == True) &
        (blocks["next_is_active"] == True)
    ]["gpp_active_block"]

    # Data gaps are allowed if they are short to moderate and bounded by active GPP.
    data_gap_blocks = blocks[
        (blocks["is_active"] == False) &
        (blocks["n_days"] <= MAX_GPP_DATA_GAP_BRIDGE_DAYS) &
        (blocks["gpp_gap_fraction"] >= 0.5) &
        (blocks["prev_is_active"] == True) &
        (blocks["next_is_active"] == True)
    ]["gpp_active_block"]

    p.loc[p["gpp_active_block"].isin(observed_dip_blocks), "gpp_active_bridged"] = True
    p.loc[p["gpp_active_block"].isin(observed_dip_blocks), "gpp_observed_dip_bridge"] = True

    p.loc[p["gpp_active_block"].isin(data_gap_blocks), "gpp_active_bridged"] = True
    p.loc[p["gpp_active_block"].isin(data_gap_blocks), "gpp_data_gap_bridge"] = True

    return p


def gap_metrics_for_window(window_df):
    """
    Summarise missingness inside a season or response window.
    """
    if window_df.empty:
        return {
            "gpp_missing_fraction": np.nan,
            "swc_missing_fraction": np.nan,
            "response_data_gap_fraction": np.nan,
            "n_response_data_gap_blocks": 0,
            "max_response_data_gap_days": 0,
            "gpp_data_gap_bridge_days": 0,
            "gpp_observed_dip_bridge_days": 0,
        }

    if "gpp_data_gap_bridge" not in window_df.columns:
        window_df = window_df.copy()
        window_df["gpp_data_gap_bridge"] = False

    if "gpp_observed_dip_bridge" not in window_df.columns:
        window_df = window_df.copy()
        window_df["gpp_observed_dip_bridge"] = False

    response_gap_runs = find_true_runs(window_df["is_response_data_gap"])

    if len(response_gap_runs) == 0:
        max_gap = 0
    else:
        max_gap = max(end - start + 1 for start, end in response_gap_runs)

    return {
        "gpp_missing_fraction": window_df["GPP"].isna().mean(),
        "swc_missing_fraction": (~window_df["has_any_SWC"]).mean(),
        "response_data_gap_fraction": window_df["is_response_data_gap"].mean(),
        "n_response_data_gap_blocks": len(response_gap_runs),
        "max_response_data_gap_days": max_gap,
        "gpp_data_gap_bridge_days": int(window_df["gpp_data_gap_bridge"].sum()),
        "gpp_observed_dip_bridge_days": int(window_df["gpp_observed_dip_bridge"].sum()),
    }


def classify_gpp_season_confidence(has_season, metrics):
    """
    Assign a simple confidence class based on missingness.
    """
    if not has_season:
        return "no_clear_gpp_season"

    missing_frac = metrics.get("gpp_missing_fraction", np.nan)
    max_gap = metrics.get("max_response_data_gap_days", 0)
    response_gap_frac = metrics.get("response_data_gap_fraction", np.nan)

    if pd.isna(missing_frac):
        return "uncertain"

    if missing_frac >= 0.50 or response_gap_frac >= 0.40 or max_gap > 60:
        return "low"

    if missing_frac >= 0.25 or response_gap_frac >= 0.20 or max_gap > 30:
        return "moderate"

    return "high"


def days_between(a, b):
    if pd.notna(a) and pd.notna(b):
        return (a - b).days
    return np.nan



# ------------------------------------------------------------
# 6B. GPP season, break, and pulse helper functions
# ------------------------------------------------------------

# Lower threshold used to keep the declining tail of a productive season
# inside the same broad GPP season. This is not used to define onset.
GPP_TAIL_FRAC = 0.10
GPP_TAIL_NOISE_MULTIPLIER = 0.50

# A true GPP season break requires more than a short dip.
# It should represent a biological reset toward baseline.
TRUE_RESET_MIN_DAYS = 21
TRUE_RESET_GPP_FRAC = 0.15
TRUE_RESET_SWC_FRAC = 0.25
UNCERTAIN_RESET_MIN_DAYS = 21
LONG_RESET_MIN_DAYS = 35

# GPP pulse detection inside broad GPP seasons.
PULSE_LOCAL_WINDOW_DAYS = 5
PULSE_BACKGROUND_DAYS = 21
PULSE_MIN_SEPARATION_DAYS = 14
PULSE_PROMINENCE_FRAC = 0.15
PULSE_NOISE_MULTIPLIER = 1.0
PULSE_RAIN_LOOKBACK_DAYS = 30
PULSE_SWC_LOOKBACK_DAYS = 21
PULSE_RAIN_SUPPORT_MM = 2.0
PULSE_SWC_SUPPORT_CHANGE = 0.5


def get_swc_baseline_and_amplitude(df_all, sub, wet_start, response_end):
    """
    Estimate antecedent SWC baseline and response amplitude.
    This is used only to judge whether a GPP dip represents a true reset.
    """
    antecedent_start = wet_start - pd.Timedelta(days=BASELINE_LOOKBACK_DAYS)

    antecedent = df_all[
        (df_all["date"] >= antecedent_start) &
        (df_all["date"] < wet_start)
    ]["SWC_shallow_7d"].dropna()

    response = sub[
        (sub["date"] >= wet_start) &
        (sub["date"] <= response_end)
    ]["SWC_shallow_7d"].dropna()

    if len(antecedent) >= MIN_BASELINE_N:
        swc_baseline = antecedent.median()
    else:
        cycle_swc = sub["SWC_shallow_7d"].dropna()
        swc_baseline = cycle_swc.quantile(0.20) if len(cycle_swc) > 0 else np.nan

    if len(response) > 0 and pd.notna(swc_baseline):
        swc_peak = response.max()
        swc_amplitude = max(swc_peak - swc_baseline, 0)
    else:
        swc_peak = np.nan
        swc_amplitude = np.nan

    return swc_baseline, swc_peak, swc_amplitude


def classify_gpp_inactive_gap(
    block_df,
    gpp_baseline,
    gpp_threshold,
    gpp_amplitude,
    swc_baseline,
    swc_amplitude,
):
    """
    Decide whether an inactive interval inside a possible GPP season should be:
    1. merged as an internal dip,
    2. bridged as a data gap,
    3. split as a true season break,
    4. split but flagged as uncertain.

    This is the central rule that separates:
    - GPP season: broad productive period
    - GPP pulse: local response inside a season
    - GPP season break: true reset toward baseline
    """
    n_days = len(block_df)

    if n_days == 0:
        return "merge_empty", False

    gpp_gap_fraction = block_df["is_gpp_data_gap"].mean()

    # Missing-data intervals are not ecological evidence of a reset.
    if gpp_gap_fraction >= 0.5:
        if n_days <= MAX_GPP_DATA_GAP_BRIDGE_DAYS:
            return "bridge_data_gap", False
        return "uncertain_break_candidate_long_data_gap", True

    # Very short observed dips are internal season variability.
    if n_days < TRUE_RESET_MIN_DAYS:
        return "merge_short_observed_dip", False

    gpp_obs = block_df["GPP_14d"].dropna()

    if len(gpp_obs) == 0 or pd.isna(gpp_baseline) or pd.isna(gpp_amplitude):
        return "uncertain_break_candidate_missing_gpp", True

    gpp_reset_threshold = gpp_baseline + TRUE_RESET_GPP_FRAC * max(gpp_amplitude, 0)

    gpp_median = gpp_obs.median()
    gpp_q75 = gpp_obs.quantile(0.75)

    # Stronger than a single low point: most of the dip must be near baseline.
    gpp_near_baseline = (
        (gpp_median <= gpp_reset_threshold) and
        (gpp_q75 <= max(gpp_threshold, gpp_reset_threshold))
    )

    swc_obs = block_df["SWC_shallow_7d"].dropna()

    if (
        len(swc_obs) > 0 and
        pd.notna(swc_baseline) and
        pd.notna(swc_amplitude) and
        swc_amplitude > 0
    ):
        swc_reset_threshold = swc_baseline + TRUE_RESET_SWC_FRAC * swc_amplitude
        swc_reset = swc_obs.median() <= swc_reset_threshold
        swc_evidence_available = True
    else:
        swc_reset = False
        swc_evidence_available = False

    # Strong split: GPP returns close to baseline and SWC also resets.
    if n_days >= TRUE_RESET_MIN_DAYS and gpp_near_baseline and swc_reset:
        return "true_gpp_season_break", True

    # Long GPP collapse without usable SWC is not merged automatically.
    # This is the rule intended for ambiguous cases like picture 3.
    if n_days >= UNCERTAIN_RESET_MIN_DAYS and gpp_near_baseline and not swc_evidence_available:
        return "uncertain_break_candidate_gpp_reset_swc_missing", True

    # Very long observed near-baseline GPP intervals are not merged automatically,
    # even if SWC evidence is ambiguous.
    if n_days >= LONG_RESET_MIN_DAYS and gpp_near_baseline:
        return "uncertain_break_candidate_long_gpp_reset", True

    # Otherwise this is treated as an internal dip/tail within one broad season.
    return "merge_no_true_reset", False


def find_broad_gpp_seasons(
    period_df,
    gpp_baseline,
    gpp_threshold,
    gpp_noise,
    gpp_amplitude,
    swc_baseline,
    swc_amplitude,
):
    """
    Detect broad GPP productive seasons inside one hydrological cycle.

    Onset requires sustained GPP above the main detection threshold.
    Continuity uses a lower tail threshold and only splits the season when
    there is evidence of a true biological reset.
    """
    p = period_df.copy()

    for col in [
        "gpp_active_raw",
        "gpp_tail_raw",
        "gpp_season_candidate_raw",
        "gpp_season_candidate_bridged",
        "gpp_within_season_dip_bridge",
        "gpp_data_gap_bridge",
        "gpp_uncertain_break_candidate",
        "gpp_true_season_break",
    ]:
        p[col] = False

    if (
        p.empty or
        pd.isna(gpp_baseline) or
        pd.isna(gpp_threshold) or
        pd.isna(gpp_amplitude)
    ):
        return p, [], []

    tail_threshold = gpp_baseline + max(
        GPP_TAIL_FRAC * max(gpp_amplitude, 0),
        GPP_TAIL_NOISE_MULTIPLIER * max(gpp_noise, 0),
        0
    )

    p["gpp_active_raw"] = (p["GPP_14d"] >= gpp_threshold).fillna(False)
    p["gpp_tail_raw"] = (p["GPP_14d"] >= tail_threshold).fillna(False)

    # Candidate season includes strong activity plus the declining productive tail.
    p["gpp_season_candidate_raw"] = p["gpp_active_raw"] | p["gpp_tail_raw"]
    p["gpp_season_candidate_bridged"] = p["gpp_season_candidate_raw"].copy()

    # One pass of bridge/split decisions across inactive intervals.
    p["candidate_block"] = (
        p["gpp_season_candidate_bridged"] != p["gpp_season_candidate_bridged"].shift()
    ).cumsum()

    blocks = (
        p.groupby("candidate_block")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_days=("date", "count"),
            is_candidate=("gpp_season_candidate_bridged", "first"),
            gpp_gap_days=("is_gpp_data_gap", "sum"),
        )
        .reset_index()
    )

    blocks["prev_is_candidate"] = blocks["is_candidate"].shift(1)
    blocks["next_is_candidate"] = blocks["is_candidate"].shift(-1)

    break_records = []

    for _, block in blocks.iterrows():
        if block["is_candidate"]:
            continue

        # Only evaluate internal gaps between candidate productive periods.
        if not (block["prev_is_candidate"] == True and block["next_is_candidate"] == True):
            continue

        block_mask = p["candidate_block"].eq(block["candidate_block"])
        block_df = p.loc[block_mask].copy()

        decision, split_here = classify_gpp_inactive_gap(
            block_df=block_df,
            gpp_baseline=gpp_baseline,
            gpp_threshold=gpp_threshold,
            gpp_amplitude=gpp_amplitude,
            swc_baseline=swc_baseline,
            swc_amplitude=swc_amplitude,
        )

        break_records.append({
            "break_start": block["start_date"],
            "break_end": block["end_date"],
            "break_duration_days": int(block["n_days"]),
            "break_decision": decision,
            "split_here": bool(split_here),
        })

        if not split_here:
            p.loc[block_mask, "gpp_season_candidate_bridged"] = True

            if decision == "bridge_data_gap":
                p.loc[block_mask, "gpp_data_gap_bridge"] = True
            else:
                p.loc[block_mask, "gpp_within_season_dip_bridge"] = True
        else:
            if "uncertain" in decision:
                p.loc[block_mask, "gpp_uncertain_break_candidate"] = True
            else:
                p.loc[block_mask, "gpp_true_season_break"] = True

    # Recalculate runs after bridging.
    p["gpp_season_block"] = (
        p["gpp_season_candidate_bridged"] != p["gpp_season_candidate_bridged"].shift()
    ).cumsum()

    runs = find_true_runs(p["gpp_season_candidate_bridged"])

    seasons = []

    for run_start, run_end in runs:
        run_df = p.iloc[run_start:run_end + 1].copy()

        active_run = first_sustained_run(
            run_df["gpp_active_raw"],
            MIN_SUSTAINED_GPP_DAYS
        )

        # A broad season needs at least one sustained strong GPP period.
        # Tail-only intervals are not treated as seasons.
        if active_run is None:
            continue

        active_start_rel, active_end_rel = active_run

        # Start at the first sustained strong GPP onset, not at a weak pre-tail.
        season_start_pos = run_start + active_start_rel
        season_end_pos = run_end

        season_df = p.iloc[season_start_pos:season_end_pos + 1].copy()

        seasons.append({
            "season_start": season_df["date"].min(),
            "season_end": season_df["date"].max(),
            "season_duration_days": len(season_df),
            "tail_threshold": tail_threshold,
            "n_active_days_raw": int(season_df["gpp_active_raw"].sum()),
            "n_tail_days_raw": int(season_df["gpp_tail_raw"].sum()),
            "gpp_data_gap_bridge_days": int(season_df["gpp_data_gap_bridge"].sum()),
            "gpp_within_season_dip_bridge_days": int(season_df["gpp_within_season_dip_bridge"].sum()),
            "n_uncertain_break_days_inside": int(season_df["gpp_uncertain_break_candidate"].sum()),
            "n_true_break_days_inside": int(season_df["gpp_true_season_break"].sum()),
        })

    return p, seasons, break_records


def local_gpp_pulse_candidates(
    season_df,
    gpp_baseline,
    gpp_noise,
    gpp_amplitude,
):
    """
    Detect local GPP pulses inside one broad GPP season.
    A pulse is a local productivity peak with enough prominence and optional
    rain/SWC support.
    """
    s = season_df.sort_values("date").copy()

    if s["GPP_7d"].notna().sum() < 5:
        return []

    min_prominence = max(
        PULSE_PROMINENCE_FRAC * max(gpp_amplitude, 0),
        PULSE_NOISE_MULTIPLIER * max(gpp_noise, 0),
        0
    )

    candidates = []

    obs = s.dropna(subset=["GPP_7d"]).copy()

    for idx, row in obs.iterrows():
        peak_date = row["date"]
        peak_value = row["GPP_7d"]

        local = s[
            (s["date"] >= peak_date - pd.Timedelta(days=PULSE_LOCAL_WINDOW_DAYS)) &
            (s["date"] <= peak_date + pd.Timedelta(days=PULSE_LOCAL_WINDOW_DAYS))
        ].dropna(subset=["GPP_7d"])

        if local.empty:
            continue

        if peak_value < local["GPP_7d"].max():
            continue

        background = s[
            (s["date"] >= peak_date - pd.Timedelta(days=PULSE_BACKGROUND_DAYS)) &
            (s["date"] <= peak_date + pd.Timedelta(days=PULSE_BACKGROUND_DAYS))
        ]["GPP_14d"].dropna()

        if len(background) >= 3:
            local_background = background.quantile(0.25)
        else:
            local_background = gpp_baseline

        prominence = peak_value - local_background

        if pd.isna(prominence) or prominence < min_prominence:
            continue

        before_rain = s[
            (s["date"] >= peak_date - pd.Timedelta(days=PULSE_RAIN_LOOKBACK_DAYS)) &
            (s["date"] <= peak_date)
        ]["rain_fluxnet_mm_day"].sum(skipna=True)

        swc_before = s[
            (s["date"] >= peak_date - pd.Timedelta(days=PULSE_SWC_LOOKBACK_DAYS)) &
            (s["date"] <= peak_date)
        ]["SWC_shallow_7d"].dropna()

        if len(swc_before) >= 2:
            swc_change_before = swc_before.iloc[-1] - swc_before.iloc[0]
        else:
            swc_change_before = np.nan

        pulse_supported_by_rain = before_rain >= PULSE_RAIN_SUPPORT_MM
        pulse_supported_by_swc = (
            pd.notna(swc_change_before) and
            swc_change_before >= PULSE_SWC_SUPPORT_CHANGE
        )

        if pulse_supported_by_rain and pulse_supported_by_swc:
            pulse_confidence = "high"
        elif pulse_supported_by_rain or pulse_supported_by_swc:
            pulse_confidence = "moderate"
        else:
            pulse_confidence = "weak_gpp_only"

        candidates.append({
            "pulse_peak_date": peak_date,
            "pulse_peak_GPP_7d": peak_value,
            "pulse_prominence": prominence,
            "pulse_local_background": local_background,
            "rain_before_peak_30d_mm": before_rain,
            "SWC_change_before_peak_21d": swc_change_before,
            "pulse_supported_by_rain": pulse_supported_by_rain,
            "pulse_supported_by_swc": pulse_supported_by_swc,
            "pulse_confidence": pulse_confidence,
        })

    # Keep the strongest peak when two candidates are very close together.
    candidates = sorted(candidates, key=lambda x: x["pulse_peak_GPP_7d"], reverse=True)
    kept = []

    for cand in candidates:
        too_close = any(
            abs((cand["pulse_peak_date"] - k["pulse_peak_date"]).days) < PULSE_MIN_SEPARATION_DAYS
            for k in kept
        )

        if not too_close:
            kept.append(cand)

    kept = sorted(kept, key=lambda x: x["pulse_peak_date"])

    for i, cand in enumerate(kept, start=1):
        cand["pulse_order_within_season"] = i

        if i == 1:
            cand["pulse_type"] = "first_pulse"
        elif cand["pulse_confidence"] in ["high", "moderate"]:
            cand["pulse_type"] = "secondary_supported_pulse"
        else:
            cand["pulse_type"] = "secondary_weak_gpp_peak"

    return kept


def worst_confidence(confidences):
    order = {
        "high": 0,
        "moderate": 1,
        "low": 2,
        "uncertain": 3,
        "no_clear_gpp_season": 4,
    }

    if len(confidences) == 0:
        return "no_clear_gpp_season"

    return sorted(confidences, key=lambda x: order.get(x, 99))[-1]


# ------------------------------------------------------------
# 7. Detect broad GPP seasons and within-season GPP pulses
# ------------------------------------------------------------

df["gpp_response_window"] = False
df["gpp_productive_season"] = False
df["gpp_phase"] = "outside_cycle"
df["gpp_detection_threshold"] = np.nan
df["gpp_tail_threshold"] = np.nan
df["gpp_baseline"] = np.nan
df["gpp_active_raw"] = False
df["gpp_tail_raw"] = False
df["gpp_season_candidate_raw"] = False
df["gpp_season_candidate_bridged"] = False
df["gpp_data_gap_bridge"] = False
df["gpp_within_season_dip_bridge"] = False
df["gpp_uncertain_break_candidate"] = False
df["gpp_true_season_break"] = False
df["gpp_season_id"] = np.nan
df["gpp_pulse_id"] = np.nan
df["is_gpp_pulse_peak"] = False

cycle_summaries = []
gpp_season_summaries = []
gpp_pulse_summaries = []

gpp_season_counter = 0
gpp_pulse_counter = 0

for cycle_id, sub in df.dropna(subset=["cycle_id"]).groupby("cycle_id"):

    sub = sub.sort_values("date").copy()
    cycle_mask = df["cycle_id"].eq(cycle_id)

    wet_sub = sub[sub["is_wet_phase"]].copy()

    if wet_sub.empty:
        continue

    cycle_start = sub["date"].min()
    cycle_end = sub["date"].max()

    wet_start = wet_sub["date"].min()
    wet_end = wet_sub["date"].max()

    # Dynamic GPP response window.
    # The cap is intentionally longer than before because broad dryland
    # productive seasons can include a long declining tail.
    minimum_response_end = wet_start + pd.Timedelta(days=MIN_GPP_RESPONSE_DAYS - 1)
    default_response_end = wet_start + pd.Timedelta(days=DEFAULT_GPP_RESPONSE_DAYS - 1)
    wet_buffer_end = wet_end + pd.Timedelta(days=POST_WET_BUFFER_DAYS)
    hard_cap_end = wet_start + pd.Timedelta(days=MAX_GPP_RESPONSE_DAYS - 1)

    target_response_end = max(default_response_end, wet_buffer_end, minimum_response_end)
    gpp_response_end = min(cycle_end, target_response_end, hard_cap_end)

    response_mask = (
        cycle_mask &
        (df["date"] >= wet_start) &
        (df["date"] <= gpp_response_end)
    )

    df.loc[response_mask, "gpp_response_window"] = True

    response_window = sub[
        (sub["date"] >= wet_start) &
        (sub["date"] <= gpp_response_end)
    ].copy()

    # GPP baseline and response threshold.
    gpp_baseline, gpp_noise, baseline_source = get_gpp_baseline(df, sub, wet_start)

    gpp_main_response_peak = peak_in_window(
        sub=response_window,
        variable="GPP_7d",
        start_date=wet_start,
        end_date=gpp_response_end
    )

    if pd.notna(gpp_baseline) and pd.notna(gpp_main_response_peak["peak"]):
        gpp_amplitude = gpp_main_response_peak["peak"] - gpp_baseline

        gpp_threshold = gpp_baseline + max(
            GPP_RESPONSE_FRAC * max(gpp_amplitude, 0),
            NOISE_MULTIPLIER * gpp_noise,
            GPP_MIN_ABS_INCREASE
        )

        gpp_tail_threshold = gpp_baseline + max(
            GPP_TAIL_FRAC * max(gpp_amplitude, 0),
            GPP_TAIL_NOISE_MULTIPLIER * max(gpp_noise, 0),
            0
        )
    else:
        gpp_amplitude = np.nan
        gpp_threshold = np.nan
        gpp_tail_threshold = np.nan

    swc_baseline, swc_peak, swc_amplitude = get_swc_baseline_and_amplitude(
        df_all=df,
        sub=sub,
        wet_start=wet_start,
        response_end=gpp_response_end
    )

    df.loc[cycle_mask, "gpp_baseline"] = gpp_baseline
    df.loc[cycle_mask, "gpp_detection_threshold"] = gpp_threshold
    df.loc[cycle_mask, "gpp_tail_threshold"] = gpp_tail_threshold

    cycle_gpp_season_ids = []
    cycle_pulse_ids = []
    cycle_break_records = []
    cycle_season_confidences = []

    if (
        not response_window.empty and
        pd.notna(gpp_threshold) and
        pd.notna(gpp_main_response_peak["peak"])
    ):
        annotated_response, detected_seasons, break_records = find_broad_gpp_seasons(
            period_df=response_window,
            gpp_baseline=gpp_baseline,
            gpp_threshold=gpp_threshold,
            gpp_noise=gpp_noise,
            gpp_amplitude=gpp_amplitude,
            swc_baseline=swc_baseline,
            swc_amplitude=swc_amplitude,
        )

        cycle_break_records.extend(break_records)

        # Write candidate, active, bridge, and break information back.
        for col in [
            "gpp_active_raw",
            "gpp_tail_raw",
            "gpp_season_candidate_raw",
            "gpp_season_candidate_bridged",
            "gpp_data_gap_bridge",
            "gpp_within_season_dip_bridge",
            "gpp_uncertain_break_candidate",
            "gpp_true_season_break",
        ]:
            if col in annotated_response.columns:
                df.loc[annotated_response.index, col] = annotated_response[col]

        if len(detected_seasons) > 0:
            df.loc[cycle_mask, "gpp_phase"] = "outside_gpp_season"

        for season in detected_seasons:
            gpp_season_counter += 1
            gpp_season_id = gpp_season_counter
            cycle_gpp_season_ids.append(gpp_season_id)

            season_start = season["season_start"]
            season_end = season["season_end"]

            season_mask = (
                cycle_mask &
                (df["date"] >= season_start) &
                (df["date"] <= season_end)
            )

            df.loc[season_mask, "gpp_productive_season"] = True
            df.loc[season_mask, "gpp_season_id"] = gpp_season_id
            df.loc[season_mask, "gpp_phase"] = "gpp_productive"

            season_df = df.loc[season_mask].copy()

            # Detect within-season GPP pulses.
            pulses = local_gpp_pulse_candidates(
                season_df=season_df,
                gpp_baseline=gpp_baseline,
                gpp_noise=gpp_noise,
                gpp_amplitude=gpp_amplitude,
            )

            for pulse in pulses:
                gpp_pulse_counter += 1
                pulse_id = gpp_pulse_counter
                cycle_pulse_ids.append(pulse_id)

                pulse_date = pulse["pulse_peak_date"]

                pulse_mask = df["date"].eq(pulse_date) & df["gpp_season_id"].eq(gpp_season_id)
                df.loc[pulse_mask, "gpp_pulse_id"] = pulse_id
                df.loc[pulse_mask, "is_gpp_pulse_peak"] = True

                pulse_summary = {
                    "gpp_pulse_id": pulse_id,
                    "gpp_season_id": gpp_season_id,
                    "cycle_id": int(cycle_id),
                    "wet_start": wet_start,
                    "gpp_season_start": season_start,
                    "gpp_season_end": season_end,
                    **pulse,
                }

                pulse_summary["days_from_wet_start_to_pulse_peak"] = days_between(
                    pulse_date,
                    wet_start
                )

                pulse_summary["days_from_season_start_to_pulse_peak"] = days_between(
                    pulse_date,
                    season_start
                )

                gpp_pulse_summaries.append(pulse_summary)

            # First pulse acts as green-up boundary when available.
            season_pulses = [p for p in pulses if p.get("pulse_order_within_season") == 1]

            if len(season_pulses) > 0:
                first_peak_date = season_pulses[0]["pulse_peak_date"]
                first_peak_value = season_pulses[0]["pulse_peak_GPP_7d"]
            else:
                first_peak = peak_in_window(
                    sub=season_df,
                    variable="GPP_7d",
                    start_date=season_start,
                    end_date=min(season_end, season_start + pd.Timedelta(days=FIRST_PEAK_SEARCH_DAYS - 1))
                )
                first_peak_date = first_peak["peak_date"]
                first_peak_value = first_peak["peak"]

            if pd.notna(first_peak_date):
                greenup_mask = (
                    season_mask &
                    (df["date"] >= season_start) &
                    (df["date"] <= first_peak_date)
                )

                browndown_mask = (
                    season_mask &
                    (df["date"] > first_peak_date) &
                    (df["date"] <= season_end)
                )

                df.loc[greenup_mask, "gpp_phase"] = "gpp_greenup"
                df.loc[browndown_mask, "gpp_phase"] = "gpp_browndown"

            # Mark missing days and bridged dips explicitly.
            gpp_gap_mask = season_mask & df["is_gpp_data_gap"]
            df.loc[gpp_gap_mask, "gpp_phase"] = "gpp_productive_gap"

            dip_bridge_mask = season_mask & df["gpp_within_season_dip_bridge"] & (~df["is_gpp_data_gap"])
            df.loc[dip_bridge_mask, "gpp_phase"] = "gpp_within_season_dip"

            season_df = df.loc[season_mask].copy()
            season_gap_metrics = gap_metrics_for_window(season_df)

            season_confidence = classify_gpp_season_confidence(
                has_season=True,
                metrics=season_gap_metrics
            )

            cycle_season_confidences.append(season_confidence)

            season_peak = peak_in_window(
                sub=season_df,
                variable="GPP_7d",
                start_date=season_start,
                end_date=season_end
            )

            season_summary = {
                "gpp_season_id": gpp_season_id,
                "cycle_id": int(cycle_id),
                "wet_start": wet_start,
                "wet_end": wet_end,
                "gpp_season_start": season_start,
                "gpp_season_end": season_end,
                "gpp_season_duration_days": (season_end - season_start).days + 1,
                "gpp_season_confidence": season_confidence,
                "gpp_baseline": gpp_baseline,
                "gpp_detection_threshold": gpp_threshold,
                "gpp_tail_threshold": gpp_tail_threshold,
                "gpp_response_amplitude": gpp_amplitude,
                "swc_baseline": swc_baseline,
                "swc_response_peak": swc_peak,
                "swc_response_amplitude": swc_amplitude,
                "GPP_season_peak": season_peak["peak"],
                "GPP_season_peak_date": season_peak["peak_date"],
                "GPP_first_pulse_peak": first_peak_value,
                "GPP_first_pulse_peak_date": first_peak_date,
                "n_gpp_pulses_in_season": len(pulses),
                "n_active_days_raw": season["n_active_days_raw"],
                "n_tail_days_raw": season["n_tail_days_raw"],
                "gpp_data_gap_bridge_days": season_gap_metrics["gpp_data_gap_bridge_days"],
                "gpp_within_season_dip_bridge_days": int(season_df["gpp_within_season_dip_bridge"].sum()),
                "gpp_missing_fraction": season_gap_metrics["gpp_missing_fraction"],
                "swc_missing_fraction": season_gap_metrics["swc_missing_fraction"],
                "response_data_gap_fraction": season_gap_metrics["response_data_gap_fraction"],
                "n_response_data_gap_blocks": season_gap_metrics["n_response_data_gap_blocks"],
                "max_response_data_gap_days": season_gap_metrics["max_response_data_gap_days"],
                "GPP_cumulative_observed_in_season": season_df["GPP"].sum(skipna=True),
                "GPP_mean_observed_in_season": season_df["GPP"].mean(skipna=True),
                "days_from_wet_start_to_gpp_season_start": days_between(season_start, wet_start),
                "days_from_wet_start_to_gpp_season_peak": days_between(season_peak["peak_date"], wet_start),
                "days_from_wet_start_to_first_pulse": days_between(first_peak_date, wet_start),
            }

            gpp_season_summaries.append(season_summary)

    if len(cycle_gpp_season_ids) == 0:
        df.loc[cycle_mask, "gpp_phase"] = "cycle_no_clear_gpp_season"

    # --------------------------------------------------------
    # Cycle-level metrics
    # --------------------------------------------------------

    full_cycle_gpp_peak = peak_in_window(
        sub=sub,
        variable="GPP_7d",
        start_date=cycle_start,
        end_date=cycle_end
    )

    gpp_peak_30d = peak_in_window(
        sub=sub,
        variable="GPP_7d",
        start_date=wet_start,
        end_date=min(cycle_end, wet_start + pd.Timedelta(days=29))
    )

    gpp_peak_60d = peak_in_window(
        sub=sub,
        variable="GPP_7d",
        start_date=wet_start,
        end_date=min(cycle_end, wet_start + pd.Timedelta(days=59))
    )

    gpp_peak_90d = peak_in_window(
        sub=sub,
        variable="GPP_7d",
        start_date=wet_start,
        end_date=min(cycle_end, wet_start + pd.Timedelta(days=89))
    )

    swc_response_peak = peak_in_window(
        sub=sub,
        variable="SWC_shallow_7d",
        start_date=wet_start,
        end_date=gpp_response_end
    )

    ndvi_response_peak = peak_in_window(
        sub=sub,
        variable="NDVI_interp_7d",
        start_date=wet_start,
        end_date=gpp_response_end
    )

    gpp_cum_30d = cumulative_in_window(sub, "GPP", wet_start, 30)
    gpp_cum_60d = cumulative_in_window(sub, "GPP", wet_start, 60)
    gpp_cum_90d = cumulative_in_window(sub, "GPP", wet_start, 90)

    cycle_seasons = [
        s for s in gpp_season_summaries
        if s["cycle_id"] == int(cycle_id)
    ]

    if len(cycle_seasons) > 0:
        first_cycle_season = sorted(cycle_seasons, key=lambda x: x["gpp_season_start"])[0]
        gpp_productive_start = min(s["gpp_season_start"] for s in cycle_seasons)
        gpp_productive_end = max(s["gpp_season_end"] for s in cycle_seasons)
        gpp_productive_duration_days = sum(s["gpp_season_duration_days"] for s in cycle_seasons)
        gpp_season_confidence = worst_confidence(cycle_season_confidences)
        first_response_peak_date = first_cycle_season["GPP_first_pulse_peak_date"]
        first_response_peak_value = first_cycle_season["GPP_first_pulse_peak"]
        gpp_browndown_duration_days = (
            first_cycle_season["gpp_season_end"] - first_response_peak_date
        ).days if pd.notna(first_response_peak_date) else np.nan
    else:
        gpp_productive_start = pd.NaT
        gpp_productive_end = pd.NaT
        gpp_productive_duration_days = np.nan
        gpp_season_confidence = "no_clear_gpp_season"
        first_response_peak_date = pd.NaT
        first_response_peak_value = np.nan
        gpp_browndown_duration_days = np.nan

    if pd.notna(gpp_productive_start) and pd.notna(first_response_peak_date):
        start_row = df[df["date"].eq(gpp_productive_start)]
        peak_row = df[df["date"].eq(first_response_peak_date)]

        if not start_row.empty and not peak_row.empty:
            start_gpp = start_row["GPP_7d"].iloc[0]
            peak_gpp = peak_row["GPP_7d"].iloc[0]
            days_to_peak = (first_response_peak_date - gpp_productive_start).days

            if pd.notna(start_gpp) and pd.notna(peak_gpp) and days_to_peak > 0:
                greenup_rate = (peak_gpp - start_gpp) / days_to_peak
            else:
                greenup_rate = np.nan
        else:
            greenup_rate = np.nan
    else:
        greenup_rate = np.nan

    if len(cycle_seasons) > 0:
        union_mask = df["gpp_season_id"].isin(cycle_gpp_season_ids)
        union_df = df.loc[union_mask].copy()
        union_gap_metrics = gap_metrics_for_window(union_df)
    else:
        union_gap_metrics = {
            "gpp_missing_fraction": np.nan,
            "swc_missing_fraction": np.nan,
            "response_data_gap_fraction": np.nan,
            "n_response_data_gap_blocks": 0,
            "max_response_data_gap_days": 0,
            "gpp_data_gap_bridge_days": 0,
            "gpp_observed_dip_bridge_days": 0,
        }

    break_statuses = "; ".join(
        sorted(set([b["break_decision"] for b in cycle_break_records]))
    ) if len(cycle_break_records) > 0 else "none"

    n_true_breaks = sum(1 for b in cycle_break_records if b["break_decision"] == "true_gpp_season_break")
    n_uncertain_breaks = sum(1 for b in cycle_break_records if "uncertain" in b["break_decision"])

    summary = {
        "cycle_id": int(cycle_id),

        "cycle_start": cycle_start,
        "cycle_end": cycle_end,
        "cycle_duration_days": (cycle_end - cycle_start).days + 1,

        "wet_start": wet_start,
        "wet_end": wet_end,
        "wet_duration_days": (wet_end - wet_start).days + 1,

        "hydro_data_gap_bridge_days": int(wet_sub["hydro_data_gap_bridge"].sum()),
        "hydro_observed_dry_gap_bridge_days": int(wet_sub["hydro_observed_dry_gap_bridge"].sum()),

        "gpp_response_window_start": wet_start,
        "gpp_response_window_end": gpp_response_end,
        "gpp_response_window_days": (gpp_response_end - wet_start).days + 1,

        "gpp_baseline": gpp_baseline,
        "gpp_baseline_noise_mad": gpp_noise,
        "gpp_baseline_source": baseline_source,
        "gpp_detection_threshold": gpp_threshold,
        "gpp_tail_threshold": gpp_tail_threshold,
        "gpp_response_amplitude": gpp_amplitude,

        "swc_baseline": swc_baseline,
        "swc_response_peak": swc_peak,
        "swc_response_amplitude": swc_amplitude,

        "n_gpp_seasons_in_cycle": len(cycle_gpp_season_ids),
        "n_gpp_pulses_in_cycle": len(cycle_pulse_ids),
        "gpp_season_break_statuses": break_statuses,
        "n_true_gpp_season_breaks": n_true_breaks,
        "n_uncertain_gpp_season_breaks": n_uncertain_breaks,

        "gpp_productive_start": gpp_productive_start,
        "gpp_productive_end": gpp_productive_end,
        "gpp_productive_duration_days": gpp_productive_duration_days,

        "gpp_season_confidence": gpp_season_confidence,
        "gpp_missing_fraction_in_gpp_season": union_gap_metrics["gpp_missing_fraction"],
        "swc_missing_fraction_in_gpp_season": union_gap_metrics["swc_missing_fraction"],
        "response_data_gap_fraction_in_gpp_season": union_gap_metrics["response_data_gap_fraction"],
        "n_response_data_gap_blocks_in_gpp_season": union_gap_metrics["n_response_data_gap_blocks"],
        "max_response_data_gap_days_in_gpp_season": union_gap_metrics["max_response_data_gap_days"],
        "gpp_data_gap_bridge_days": union_gap_metrics["gpp_data_gap_bridge_days"],
        "gpp_observed_dip_bridge_days": int(df.loc[df["gpp_season_id"].isin(cycle_gpp_season_ids), "gpp_within_season_dip_bridge"].sum()) if len(cycle_gpp_season_ids) > 0 else 0,

        "GPP_first_response_peak": first_response_peak_value,
        "GPP_first_response_peak_date": first_response_peak_date,

        "GPP_main_response_peak": gpp_main_response_peak["peak"],
        "GPP_main_response_peak_date": gpp_main_response_peak["peak_date"],

        "GPP_full_cycle_peak": full_cycle_gpp_peak["peak"],
        "GPP_full_cycle_peak_date": full_cycle_gpp_peak["peak_date"],

        "GPP_peak_30d": gpp_peak_30d["peak"],
        "GPP_peak_30d_date": gpp_peak_30d["peak_date"],

        "GPP_peak_60d": gpp_peak_60d["peak"],
        "GPP_peak_60d_date": gpp_peak_60d["peak_date"],

        "GPP_peak_90d": gpp_peak_90d["peak"],
        "GPP_peak_90d_date": gpp_peak_90d["peak_date"],

        "GPP_cumulative_30d": gpp_cum_30d["sum"],
        "GPP_cumulative_60d": gpp_cum_60d["sum"],
        "GPP_cumulative_90d": gpp_cum_90d["sum"],

        "GPP_mean_30d": gpp_cum_30d["mean"],
        "GPP_mean_60d": gpp_cum_60d["mean"],
        "GPP_mean_90d": gpp_cum_90d["mean"],

        "GPP_n_valid_30d": gpp_cum_30d["n_valid"],
        "GPP_n_valid_60d": gpp_cum_60d["n_valid"],
        "GPP_n_valid_90d": gpp_cum_90d["n_valid"],

        "GPP_missing_fraction_30d": gpp_cum_30d["missing_fraction"],
        "GPP_missing_fraction_60d": gpp_cum_60d["missing_fraction"],
        "GPP_missing_fraction_90d": gpp_cum_90d["missing_fraction"],

        "GPP_greenup_rate": greenup_rate,
        "GPP_browndown_duration_days": gpp_browndown_duration_days,

        "SWC_shallow_response_peak": swc_response_peak["peak"],
        "SWC_shallow_response_peak_date": swc_response_peak["peak_date"],

        "NDVI_response_peak": ndvi_response_peak["peak"],
        "NDVI_response_peak_date": ndvi_response_peak["peak_date"],

        "total_rain_mm": sub["rain_fluxnet_mm_day"].sum(skipna=True),
        "wet_phase_rain_mm": wet_sub["rain_fluxnet_mm_day"].sum(skipna=True),

        "mean_SWC_shallow": sub["SWC_shallow"].mean(skipna=True),
        "mean_SWC_middle": sub["SWC_middle"].mean(skipna=True),
        "mean_GPP": sub["GPP"].mean(skipna=True),
        "cumulative_GPP_full_cycle": sub["GPP"].sum(skipna=True),

        "mean_NDVI_obs": sub["NDVI_obs"].mean(skipna=True),
        "n_NDVI_obs": sub["NDVI_obs"].notna().sum(),
    }

    # Relative timing metrics
    summary["GPP_first_peak_minus_wet_start_days"] = days_between(
        summary["GPP_first_response_peak_date"],
        wet_start
    )

    summary["GPP_main_peak_minus_wet_start_days"] = days_between(
        summary["GPP_main_response_peak_date"],
        wet_start
    )

    summary["GPP_full_cycle_peak_minus_wet_start_days"] = days_between(
        summary["GPP_full_cycle_peak_date"],
        wet_start
    )

    summary["SWC_peak_minus_wet_start_days"] = days_between(
        summary["SWC_shallow_response_peak_date"],
        wet_start
    )

    summary["GPP_first_peak_minus_SWC_peak_days"] = days_between(
        summary["GPP_first_response_peak_date"],
        summary["SWC_shallow_response_peak_date"]
    )

    summary["GPP_main_peak_minus_SWC_peak_days"] = days_between(
        summary["GPP_main_response_peak_date"],
        summary["SWC_shallow_response_peak_date"]
    )

    summary["NDVI_peak_minus_GPP_first_peak_days"] = days_between(
        summary["NDVI_response_peak_date"],
        summary["GPP_first_response_peak_date"]
    )

    summary["NDVI_peak_minus_SWC_peak_days"] = days_between(
        summary["NDVI_response_peak_date"],
        summary["SWC_shallow_response_peak_date"]
    )

    cycle_summaries.append(summary)

cycle_summary_df = pd.DataFrame(cycle_summaries)
gpp_season_summary_df = pd.DataFrame(gpp_season_summaries)
gpp_pulse_summary_df = pd.DataFrame(gpp_pulse_summaries)

# ------------------------------------------------------------
# 8. Export
# ------------------------------------------------------------

df.to_csv(OUTPUT_DAILY, index=False)
cycle_summary_df.to_csv(OUTPUT_CYCLES, index=False)
gap_summary_df.to_csv(OUTPUT_GAPS, index=False)
gpp_season_summary_df.to_csv(OUTPUT_GPP_SEASONS, index=False)
gpp_pulse_summary_df.to_csv(OUTPUT_GPP_PULSES, index=False)

print("\nDone.")
print("\nDaily file saved to:")
print(OUTPUT_DAILY)

print("\nCycle summary saved to:")
print(OUTPUT_CYCLES)

print("\nGPP season summary saved to:")
print(OUTPUT_GPP_SEASONS)

print("\nGPP pulse summary saved to:")
print(OUTPUT_GPP_PULSES)

print("\nData-gap summary saved to:")
print(OUTPUT_GAPS)

print("\nDetected cycles, GPP seasons, GPP pulses, and break decisions:")

cols_to_print = [
    "cycle_id",
    "cycle_start",
    "cycle_end",
    "wet_start",
    "wet_end",
    "n_gpp_seasons_in_cycle",
    "n_gpp_pulses_in_cycle",
    "gpp_season_break_statuses",
    "n_true_gpp_season_breaks",
    "n_uncertain_gpp_season_breaks",
    "gpp_productive_start",
    "gpp_productive_end",
    "gpp_season_confidence",
    "gpp_data_gap_bridge_days",
    "gpp_observed_dip_bridge_days",
    "max_response_data_gap_days_in_gpp_season",
    "GPP_first_response_peak_date",
    "GPP_main_response_peak_date",
    "GPP_full_cycle_peak_date",
    "GPP_first_peak_minus_wet_start_days",
    "GPP_main_peak_minus_wet_start_days",
    "GPP_full_cycle_peak_minus_wet_start_days",
    "GPP_first_peak_minus_SWC_peak_days",
    "NDVI_peak_minus_GPP_first_peak_days",
    "GPP_cumulative_90d",
    "GPP_missing_fraction_90d",
    "n_NDVI_obs",
]

existing_cols = [c for c in cols_to_print if c in cycle_summary_df.columns]

print(
    cycle_summary_df[existing_cols]
    .to_string(index=False)
)

if not gpp_season_summary_df.empty:
    print("\nDetected broad GPP seasons:")
    print(
        gpp_season_summary_df[[
            "gpp_season_id",
            "cycle_id",
            "gpp_season_start",
            "gpp_season_end",
            "gpp_season_duration_days",
            "gpp_season_confidence",
            "n_gpp_pulses_in_season",
            "GPP_season_peak_date",
            "GPP_first_pulse_peak_date",
            "gpp_within_season_dip_bridge_days",
            "gpp_data_gap_bridge_days",
        ]].to_string(index=False)
    )

if not gpp_pulse_summary_df.empty:
    print("\nDetected GPP pulses:")
    print(
        gpp_pulse_summary_df[[
            "gpp_pulse_id",
            "gpp_season_id",
            "cycle_id",
            "pulse_order_within_season",
            "pulse_type",
            "pulse_peak_date",
            "pulse_peak_GPP_7d",
            "pulse_prominence",
            "rain_before_peak_30d_mm",
            "SWC_change_before_peak_21d",
            "pulse_confidence",
        ]].to_string(index=False)
    )

# ------------------------------------------------------------
# 9. Visual check
# ------------------------------------------------------------

plt.figure(figsize=(17, 7))

plt.plot(
    df["date"],
    df["SWC_shallow_7d"],
    label="SWC shallow, 7-day mean",
    linewidth=1.5
)

plt.plot(
    df["date"],
    df["GPP_7d"],
    label="GPP, 7-day mean",
    linewidth=1.5
)

plt.plot(
    df["date"],
    df["GPP_14d"],
    label="GPP, 14-day mean",
    linewidth=1.2,
    alpha=0.8
)

plt.scatter(
    df["date"],
    df["NDVI_obs"] * 50,
    s=20,
    label="NDVI observed x50",
    alpha=0.8
)

# Plot response data gaps first so the GPP season overlay remains visible.
gap_label_added = False

for _, row in response_gap_summary.iterrows():
    if row["gap_duration_days"] >= 3:
        plt.axvspan(
            row["gap_start"],
            row["gap_end"],
            color="grey",
            alpha=0.15,
            label="Response data gap" if not gap_label_added else None
        )
        gap_label_added = True

# Hydrological wet-season starts
wet_label_added = False

for _, row in wet_blocks.iterrows():
    plt.axvline(
        row["wet_start"],
        linestyle="--",
        alpha=0.45,
        label="Wet-season onset" if not wet_label_added else None
    )
    wet_label_added = True

# Broad GPP productive seasons
season_span_label_added = False

if not gpp_season_summary_df.empty:
    for _, row in gpp_season_summary_df.iterrows():
        plt.axvspan(
            row["gpp_season_start"],
            row["gpp_season_end"],
            alpha=0.12,
            label="Broad GPP productive season" if not season_span_label_added else None
        )
        season_span_label_added = True


# GPP pulses and main/full-cycle peaks
pulse_label_added = False
main_peak_label_added = False
full_peak_label_added = False

if not gpp_pulse_summary_df.empty:
    for _, row in gpp_pulse_summary_df.iterrows():
        plt.scatter(
            row["pulse_peak_date"],
            row["pulse_peak_GPP_7d"],
            s=70,
            marker="o",
            label="GPP pulse peak" if not pulse_label_added else None
        )
        pulse_label_added = True

for _, row in cycle_summary_df.iterrows():
    if pd.notna(row["GPP_main_response_peak_date"]) and pd.notna(row["GPP_main_response_peak"]):
        plt.scatter(
            row["GPP_main_response_peak_date"],
            row["GPP_main_response_peak"],
            s=70,
            marker="^",
            label="Main GPP response-window peak" if not main_peak_label_added else None
        )
        main_peak_label_added = True

    if pd.notna(row["GPP_full_cycle_peak_date"]) and pd.notna(row["GPP_full_cycle_peak"]):
        plt.scatter(
            row["GPP_full_cycle_peak_date"],
            row["GPP_full_cycle_peak"],
            s=50,
            marker="x",
            label="Full-cycle GPP peak" if not full_peak_label_added else None
        )
        full_peak_label_added = True

plt.title("Gap-aware seasonal scaffold with GPP seasons, pulses, and break candidates")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
