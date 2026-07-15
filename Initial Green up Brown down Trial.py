# ============================================================
# Kapiti Seasonal: NDVI Green-up and Brown-down within GPP Seasons
#
# Aim:
# Use the existing GPP growing-season scaffold to define the NDVI signature:
#
# 1. NDVI green-up start
# 2. NDVI peak date
# 3. NDVI brown-down end
# 4. NDVI green-up duration
# 5. NDVI brown-down duration
# 6. NDVI amplitude and integral
# 7. Confidence based on observed NDVI support
#
# This script does NOT redefine GPP seasons.
# It adds NDVI phenology labels inside the current scaffold.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

INPUT_DAILY = PROJECT_DIR / "Kapiti_Seasonal_DAILY_WITH_GPP_SEASONS.csv"

OUTPUT_DAILY = PROJECT_DIR / "Kapiti_Seasonal_DAILY_WITH_GPP_AND_NDVI_PHASES.csv"
OUTPUT_NDVI_SUMMARY = PROJECT_DIR / "Kapiti_Seasonal_NDVI_PHENOLOGY_SUMMARY.csv"

# ------------------------------------------------------------
# 2. User-adjustable parameters
# ------------------------------------------------------------

NDVI_COL = "NDVI_interp_7d"
NDVI_OBS_COL = "NDVI_obs"

ANTECEDENT_BASELINE_DAYS = 45
NDVI_THRESHOLD_FRACTION = 0.25

MIN_SUSTAINED_GREENUP_DAYS = 5
MIN_SUSTAINED_BROUNDDOWN_DAYS = 5

PEAK_OBS_SUPPORT_WINDOW_DAYS = 10

HIGH_CONF_MIN_OBS = 8
HIGH_CONF_MAX_OBS_GAP_DAYS = 30

MODERATE_CONF_MIN_OBS = 5
MODERATE_CONF_MAX_OBS_GAP_DAYS = 45

LOW_CONF_MIN_OBS = 2

# Optional visual scaling for NDVI on the same plot as GPP/SWC
NDVI_PLOT_SCALE = 50

# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_DAILY, low_memory=False)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("\nLoaded scaffolded daily dataset:")
print(INPUT_DAILY)
print(f"Rows: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ------------------------------------------------------------
# 4. Ensure numeric columns and fallback NDVI columns
# ------------------------------------------------------------

possible_numeric_cols = [
    "rain_fluxnet_mm_day",
    "rain_7d_sum",
    "rain_14d_sum",
    "SWC_shallow",
    "SWC_middle",
    "SWC_deep",
    "SWC_shallow_7d",
    "SWC_middle_7d",
    "SWC_deep_7d",
    "GPP",
    "GPP_7d",
    "GPP_14d",
    "NDVI",
    "NDVI_obs",
    "NDVI_interp",
    "NDVI_interp_7d",
]

for col in possible_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "NDVI_obs" not in df.columns:
    if "NDVI" in df.columns:
        df["NDVI_obs"] = df["NDVI"]
    else:
        raise ValueError("No NDVI or NDVI_obs column found.")

if "NDVI_interp" not in df.columns:
    df["NDVI_interp"] = df["NDVI_obs"].interpolate(
        method="linear",
        limit=30,
        limit_direction="both"
    )

if "NDVI_interp_7d" not in df.columns:
    df["NDVI_interp_7d"] = (
        df["NDVI_interp"]
        .rolling(7, min_periods=3, center=True)
        .mean()
    )

if "GPP_7d" not in df.columns and "GPP" in df.columns:
    df["GPP_7d"] = (
        df["GPP"]
        .rolling(7, min_periods=3, center=True)
        .mean()
    )

if "SWC_shallow_7d" not in df.columns and "SWC_shallow" in df.columns:
    df["SWC_shallow_7d"] = (
        df["SWC_shallow"]
        .rolling(7, min_periods=3)
        .mean()
    )

# ------------------------------------------------------------
# 5. Reconstruct gpp_season_id if missing
# ------------------------------------------------------------

if "gpp_season_id" not in df.columns:

    print("\n'gpp_season_id' not found.")
    print("Reconstructing GPP season IDs from existing scaffold columns...")

    if "gpp_productive_season" in df.columns:
        gpp_active = df["gpp_productive_season"].fillna(False).astype(bool)

    elif "gpp_phase" in df.columns:
        active_phases = [
            "gpp_greenup",
            "gpp_browndown",
            "gpp_productive",
            "gpp_productive_gap",
        ]

        gpp_active = df["gpp_phase"].isin(active_phases)

    else:
        raise ValueError(
            "Cannot reconstruct gpp_season_id because neither "
            "'gpp_productive_season' nor usable 'gpp_phase' labels were found."
        )

    gpp_block = (gpp_active != gpp_active.shift()).cumsum()

    df["gpp_season_id"] = np.nan

    active_blocks = (
        pd.DataFrame({
            "gpp_block": gpp_block,
            "gpp_active": gpp_active,
            "date": df["date"]
        })
        .groupby("gpp_block")
        .agg(
            is_active=("gpp_active", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_days=("date", "count")
        )
        .reset_index()
    )

    active_blocks = active_blocks[active_blocks["is_active"] == True].copy()
    active_blocks["gpp_season_id_new"] = np.arange(1, len(active_blocks) + 1)

    for _, row in active_blocks.iterrows():
        mask = gpp_block.eq(row["gpp_block"])
        df.loc[mask, "gpp_season_id"] = row["gpp_season_id_new"]

    print(f"Reconstructed {df['gpp_season_id'].nunique()} GPP seasons.")

else:
    print("\n'gpp_season_id' found in dataset.")# ------------------------------------------------------------
# 6. Helper functions
# ------------------------------------------------------------

def safe_sum(series):
    if series.notna().sum() == 0:
        return np.nan

    return series.sum(skipna=True)


def safe_mean(series):
    if series.notna().sum() == 0:
        return np.nan

    return series.mean(skipna=True)


def days_between(a, b):
    if pd.notna(a) and pd.notna(b):
        return (a - b).days

    return np.nan


def find_true_runs(mask):
    """
    Return start and end positions of True runs.
    Positions are relative to the supplied array.
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
    runs = find_true_runs(mask)

    valid_runs = [
        (start, end)
        for start, end in runs
        if (end - start + 1) >= min_days
    ]

    if len(valid_runs) == 0:
        return None

    return valid_runs[0]


def first_sustained_false_run_after_peak(mask, dates, peak_date, min_days):
    """
    Finds the first sustained False run after the NDVI peak.
    Used to define brown-down end.
    """
    temp = pd.DataFrame({
        "date": dates,
        "above_threshold": pd.Series(mask).fillna(False).astype(bool)
    }).copy()

    temp = temp[temp["date"] > peak_date].reset_index(drop=True)

    if temp.empty:
        return None

    below = ~temp["above_threshold"]
    runs = find_true_runs(below)

    valid_runs = [
        (start, end)
        for start, end in runs
        if (end - start + 1) >= min_days
    ]

    if len(valid_runs) == 0:
        return None

    start_pos, end_pos = valid_runs[0]

    return {
        "first_below_date": temp.loc[start_pos, "date"],
        "below_run_end_date": temp.loc[end_pos, "date"]
    }


def max_observed_ndvi_gap_days(obs_dates):
    obs_dates = pd.Series(obs_dates).dropna().sort_values()

    if len(obs_dates) < 2:
        return np.nan

    gaps = obs_dates.diff().dt.days.dropna()

    if gaps.empty:
        return np.nan

    return gaps.max()


def peak_supported_by_observation(obs_dates, peak_date, window_days):
    obs_dates = pd.Series(obs_dates).dropna()

    if pd.isna(peak_date) or obs_dates.empty:
        return False

    delta_days = (obs_dates - peak_date).abs().dt.days

    return (delta_days <= window_days).any()


def classify_ndvi_confidence(
    n_obs,
    max_obs_gap_days,
    peak_supported,
    amplitude,
    greenup_start,
    browndown_end
):
    if n_obs < LOW_CONF_MIN_OBS:
        return "unusable"

    if pd.isna(amplitude) or amplitude <= 0:
        return "unusable"

    if pd.isna(greenup_start) or pd.isna(browndown_end):
        return "low"

    if (
        n_obs >= HIGH_CONF_MIN_OBS and
        pd.notna(max_obs_gap_days) and
        max_obs_gap_days <= HIGH_CONF_MAX_OBS_GAP_DAYS and
        peak_supported
    ):
        return "high"

    if (
        n_obs >= MODERATE_CONF_MIN_OBS and
        pd.notna(max_obs_gap_days) and
        max_obs_gap_days <= MODERATE_CONF_MAX_OBS_GAP_DAYS
    ):
        return "moderate"

    return "low"


def calculate_ndvi_phenology_for_season(
    full_df,
    season_sub,
    season_id,
    ndvi_col,
    ndvi_obs_col
):
    season_sub = season_sub.sort_values("date").copy()

    season_start = season_sub["date"].min()
    season_end = season_sub["date"].max()
    season_duration_days = (season_end - season_start).days + 1

    # Observed NDVI support inside the GPP season
    obs_in_season = season_sub.dropna(subset=[ndvi_obs_col]).copy()
    obs_dates = obs_in_season["date"]

    n_obs = len(obs_in_season)
    max_obs_gap = max_observed_ndvi_gap_days(obs_dates)

    # Phenology series
    valid = season_sub[["date", ndvi_col]].dropna().copy()

    if valid.empty:
        return {
            "gpp_season_id": season_id,
            "gpp_season_start": season_start,
            "gpp_season_end": season_end,
            "gpp_season_duration_days": season_duration_days,
            "NDVI_baseline": np.nan,
            "NDVI_baseline_source": "unavailable",
            "NDVI_peak_value": np.nan,
            "NDVI_peak_date": pd.NaT,
            "NDVI_amplitude": np.nan,
            "NDVI_green_threshold": np.nan,
            "NDVI_greenup_start_date": pd.NaT,
            "NDVI_browndown_start_date": pd.NaT,
            "NDVI_browndown_end_date": pd.NaT,
            "NDVI_greenup_duration_days": np.nan,
            "NDVI_browndown_duration_days": np.nan,
            "NDVI_integral_interp_7d": np.nan,
            "NDVI_mean_interp_7d": np.nan,
            "NDVI_n_valid_interp_7d": 0,
            "NDVI_obs_n": n_obs,
            "NDVI_obs_max_gap_days": max_obs_gap,
            "NDVI_peak_supported_by_obs": False,
            "NDVI_confidence": "unusable",
            "NDVI_issue": "no_valid_interpolated_ndvi"
        }

    # Baseline from antecedent period where possible
    antecedent_start = season_start - pd.Timedelta(days=ANTECEDENT_BASELINE_DAYS)

    antecedent = full_df[
        (full_df["date"] >= antecedent_start) &
        (full_df["date"] < season_start)
    ][ndvi_col].dropna()

    if len(antecedent) >= 3:
        baseline = antecedent.median()
        baseline_source = "antecedent_45d_median"
    else:
        baseline = valid[ndvi_col].quantile(0.20)
        baseline_source = "season_20th_percentile"

    # Peak
    idx_peak = valid[ndvi_col].idxmax()
    peak_date = valid.loc[idx_peak, "date"]
    peak_value = valid.loc[idx_peak, ndvi_col]

    amplitude = peak_value - baseline

    if pd.isna(amplitude) or amplitude <= 0:
        green_threshold = np.nan
        above_threshold = pd.Series(False, index=season_sub.index)
    else:
        green_threshold = baseline + NDVI_THRESHOLD_FRACTION * amplitude
        above_threshold = season_sub[ndvi_col] >= green_threshold

    # Green-up start
    greenup_start_date = pd.NaT

    first_run = first_sustained_run(
        above_threshold,
        MIN_SUSTAINED_GREENUP_DAYS
    )

    if first_run is not None:
        start_pos, end_pos = first_run
        greenup_start_date = season_sub.iloc[start_pos]["date"]

    # Brown-down
    browndown_start_date = peak_date
    browndown_end_date = pd.NaT

    if pd.notna(peak_date) and pd.notna(green_threshold):
        below_run = first_sustained_false_run_after_peak(
            mask=above_threshold,
            dates=season_sub["date"],
            peak_date=peak_date,
            min_days=MIN_SUSTAINED_BROUNDDOWN_DAYS
        )

        if below_run is not None:
            first_below_date = below_run["first_below_date"]
            browndown_end_date = first_below_date - pd.Timedelta(days=1)

            if browndown_end_date < peak_date:
                browndown_end_date = first_below_date
        else:
            # If it never falls below threshold, NDVI remains elevated until season end
            browndown_end_date = season_end

    # Durations
    greenup_duration = days_between(peak_date, greenup_start_date)
    browndown_duration = days_between(browndown_end_date, peak_date)

    # Magnitude
    ndvi_integral = safe_sum(season_sub[ndvi_col])
    ndvi_mean = safe_mean(season_sub[ndvi_col])
    ndvi_n_valid = season_sub[ndvi_col].notna().sum()

    # Observation support
    peak_supported = peak_supported_by_observation(
        obs_dates=obs_dates,
        peak_date=peak_date,
        window_days=PEAK_OBS_SUPPORT_WINDOW_DAYS
    )

    confidence = classify_ndvi_confidence(
        n_obs=n_obs,
        max_obs_gap_days=max_obs_gap,
        peak_supported=peak_supported,
        amplitude=amplitude,
        greenup_start=greenup_start_date,
        browndown_end=browndown_end_date
    )

    # Issue flag
    if confidence == "unusable":
        issue = "insufficient_observed_ndvi_or_no_amplitude"
    elif confidence == "low":
        issue = "low_observed_ndvi_support"
    elif confidence == "moderate":
        issue = "moderate_observed_ndvi_support"
    else:
        issue = "good_observed_ndvi_support"

    return {
        "gpp_season_id": season_id,
        "gpp_season_start": season_start,
        "gpp_season_end": season_end,
        "gpp_season_duration_days": season_duration_days,

        "NDVI_baseline": baseline,
        "NDVI_baseline_source": baseline_source,
        "NDVI_peak_value": peak_value,
        "NDVI_peak_date": peak_date,
        "NDVI_amplitude": amplitude,
        "NDVI_green_threshold": green_threshold,

        "NDVI_greenup_start_date": greenup_start_date,
        "NDVI_browndown_start_date": browndown_start_date,
        "NDVI_browndown_end_date": browndown_end_date,

        "NDVI_greenup_duration_days": greenup_duration,
        "NDVI_browndown_duration_days": browndown_duration,

        "NDVI_integral_interp_7d": ndvi_integral,
        "NDVI_mean_interp_7d": ndvi_mean,
        "NDVI_n_valid_interp_7d": ndvi_n_valid,

        "NDVI_obs_n": n_obs,
        "NDVI_obs_max_gap_days": max_obs_gap,
        "NDVI_peak_supported_by_obs": peak_supported,

        "NDVI_confidence": confidence,
        "NDVI_issue": issue,
    }

# ------------------------------------------------------------
# 7. Initialise daily NDVI phase columns
# ------------------------------------------------------------

df["ndvi_phase"] = "outside_gpp_season"
df["ndvi_phase"] = df["ndvi_phase"].astype("object")

df["ndvi_season_confidence"] = None
df["ndvi_season_confidence"] = df["ndvi_season_confidence"].astype("object")

df["ndvi_green_threshold"] = np.nan
df["ndvi_baseline"] = np.nan
df["ndvi_amplitude"] = np.nan

# ------------------------------------------------------------
# 8. Calculate NDVI phenology inside each GPP season
# ------------------------------------------------------------

season_df = df.dropna(subset=["gpp_season_id"]).copy()

if season_df.empty:
    raise ValueError("No GPP seasons found after reconstruction.")

ndvi_rows = []

# Safety: make sure text columns can store string labels
df["ndvi_phase"] = df["ndvi_phase"].astype("object")
df["ndvi_season_confidence"] = df["ndvi_season_confidence"].astype("object")

for season_id, sub in season_df.groupby("gpp_season_id"):

    sub = sub.sort_values("date").copy()

    result = calculate_ndvi_phenology_for_season(
        full_df=df,
        season_sub=sub,
        season_id=season_id,
        ndvi_col=NDVI_COL,
        ndvi_obs_col=NDVI_OBS_COL
    )

    ndvi_rows.append(result)

    season_mask = df["gpp_season_id"].eq(season_id)

    # Store season-level NDVI diagnostics on daily rows
    df.loc[season_mask, "ndvi_season_confidence"] = result["NDVI_confidence"]
    df.loc[season_mask, "ndvi_green_threshold"] = result["NDVI_green_threshold"]
    df.loc[season_mask, "ndvi_baseline"] = result["NDVI_baseline"]
    df.loc[season_mask, "ndvi_amplitude"] = result["NDVI_amplitude"]

    greenup_start = result["NDVI_greenup_start_date"]
    peak_date = result["NDVI_peak_date"]
    browndown_end = result["NDVI_browndown_end_date"]
    confidence = result["NDVI_confidence"]

    # Assign daily NDVI phases
    if confidence == "unusable":
        df.loc[season_mask, "ndvi_phase"] = "ndvi_unusable"

    else:
        df.loc[season_mask, "ndvi_phase"] = "ndvi_pre_greenup"

        if pd.notna(greenup_start) and pd.notna(peak_date):
            greenup_mask = (
                season_mask &
                (df["date"] >= greenup_start) &
                (df["date"] < peak_date)
            )

            peak_window_mask = (
                season_mask &
                (df["date"] >= peak_date - pd.Timedelta(days=3)) &
                (df["date"] <= peak_date + pd.Timedelta(days=3))
            )

            df.loc[greenup_mask, "ndvi_phase"] = "ndvi_greenup"
            df.loc[peak_window_mask, "ndvi_phase"] = "ndvi_peak_window"

        if pd.notna(peak_date) and pd.notna(browndown_end):
            browndown_mask = (
                season_mask &
                (df["date"] > peak_date + pd.Timedelta(days=3)) &
                (df["date"] <= browndown_end)
            )

            post_browndown_mask = (
                season_mask &
                (df["date"] > browndown_end)
            )

            df.loc[browndown_mask, "ndvi_phase"] = "ndvi_browndown"
            df.loc[post_browndown_mask, "ndvi_phase"] = "ndvi_post_browndown"

# Summary dataframe
ndvi_summary_df = pd.DataFrame(ndvi_rows)

# ------------------------------------------------------------
# 9. Add timing comparisons with GPP and SWC if available
# ------------------------------------------------------------

extra_rows = []

for _, row in ndvi_summary_df.iterrows():

    season_id = row["gpp_season_id"]

    sub = df[df["gpp_season_id"].eq(season_id)].copy()

    # GPP peak
    if "GPP_7d" in sub.columns and sub["GPP_7d"].notna().sum() > 0:
        gpp_idx = sub["GPP_7d"].idxmax()
        gpp_peak_date = sub.loc[gpp_idx, "date"]
        gpp_peak_value = sub.loc[gpp_idx, "GPP_7d"]
    else:
        gpp_peak_date = pd.NaT
        gpp_peak_value = np.nan

    # SWC peak
    if "SWC_shallow_7d" in sub.columns and sub["SWC_shallow_7d"].notna().sum() > 0:
        swc_idx = sub["SWC_shallow_7d"].idxmax()
        swc_peak_date = sub.loc[swc_idx, "date"]
        swc_peak_value = sub.loc[swc_idx, "SWC_shallow_7d"]
    else:
        swc_peak_date = pd.NaT
        swc_peak_value = np.nan

    extra_rows.append({
        "gpp_season_id": season_id,
        "GPP_7d_peak_date": gpp_peak_date,
        "GPP_7d_peak_value": gpp_peak_value,
        "SWC_shallow_7d_peak_date": swc_peak_date,
        "SWC_shallow_7d_peak_value": swc_peak_value,
        "NDVI_peak_minus_GPP_peak_days": days_between(
            row["NDVI_peak_date"],
            gpp_peak_date
        ),
        "NDVI_peak_minus_SWC_peak_days": days_between(
            row["NDVI_peak_date"],
            swc_peak_date
        ),
        "NDVI_greenup_start_minus_GPP_season_start_days": days_between(
            row["NDVI_greenup_start_date"],
            row["gpp_season_start"]
        ),
        "NDVI_browndown_end_minus_GPP_season_end_days": days_between(
            row["NDVI_browndown_end_date"],
            row["gpp_season_end"]
        ),
    })

extra_df = pd.DataFrame(extra_rows)

ndvi_summary_df = ndvi_summary_df.merge(
    extra_df,
    on="gpp_season_id",
    how="left"
)

# ------------------------------------------------------------
# 10. Export
# ------------------------------------------------------------

df.to_csv(OUTPUT_DAILY, index=False)
ndvi_summary_df.to_csv(OUTPUT_NDVI_SUMMARY, index=False)

print("\nSaved daily file with NDVI phases:")
print(OUTPUT_DAILY)

print("\nSaved NDVI phenology summary:")
print(OUTPUT_NDVI_SUMMARY)

print("\nNDVI phenology summary:")
print_cols = [
    "gpp_season_id",
    "gpp_season_start",
    "gpp_season_end",
    "NDVI_greenup_start_date",
    "NDVI_peak_date",
    "NDVI_browndown_end_date",
    "NDVI_greenup_duration_days",
    "NDVI_browndown_duration_days",
    "NDVI_amplitude",
    "NDVI_obs_n",
    "NDVI_obs_max_gap_days",
    "NDVI_peak_supported_by_obs",
    "NDVI_confidence",
    "NDVI_peak_minus_GPP_peak_days",
    "NDVI_peak_minus_SWC_peak_days",
]

print_cols = [c for c in print_cols if c in ndvi_summary_df.columns]

print(
    ndvi_summary_df[print_cols]
    .to_string(index=False)
)

# ------------------------------------------------------------
# 11. Visual check: NDVI phases on time series
# ------------------------------------------------------------

plt.figure(figsize=(17, 7))

if "SWC_shallow_7d" in df.columns:
    plt.plot(
        df["date"],
        df["SWC_shallow_7d"],
        label="SWC shallow, 7-day mean",
        linewidth=1.4
    )

if "GPP_7d" in df.columns:
    plt.plot(
        df["date"],
        df["GPP_7d"],
        label="GPP, 7-day mean",
        linewidth=1.4
    )

plt.plot(
    df["date"],
    df[NDVI_COL] * NDVI_PLOT_SCALE,
    label=f"{NDVI_COL} x{NDVI_PLOT_SCALE}",
    linewidth=1.2,
    alpha=0.8
)

plt.scatter(
    df["date"],
    df[NDVI_OBS_COL] * NDVI_PLOT_SCALE,
    s=22,
    alpha=0.9,
    label=f"{NDVI_OBS_COL} x{NDVI_PLOT_SCALE}"
)

# Shade GPP seasons
gpp_label_added = False

for season_id, sub in df.dropna(subset=["gpp_season_id"]).groupby("gpp_season_id"):
    start = sub["date"].min()
    end = sub["date"].max()

    plt.axvspan(
        start,
        end,
        alpha=0.08,
        label="GPP growing season" if not gpp_label_added else None
    )

    gpp_label_added = True

# NDVI phase markers
greenup_label_added = False
peak_label_added = False
browndown_label_added = False

for _, row in ndvi_summary_df.iterrows():

    if pd.notna(row["NDVI_greenup_start_date"]):
        plt.axvline(
            row["NDVI_greenup_start_date"],
            linestyle=":",
            alpha=0.55,
            label="NDVI green-up start" if not greenup_label_added else None
        )
        greenup_label_added = True

    if pd.notna(row["NDVI_peak_date"]):
        plt.axvline(
            row["NDVI_peak_date"],
            linestyle="-.",
            alpha=0.55,
            label="NDVI peak" if not peak_label_added else None
        )
        peak_label_added = True

    if pd.notna(row["NDVI_browndown_end_date"]):
        plt.axvline(
            row["NDVI_browndown_end_date"],
            linestyle="--",
            alpha=0.55,
            label="NDVI brown-down end" if not browndown_label_added else None
        )
        browndown_label_added = True

plt.title("NDVI green-up and brown-down inside GPP-defined growing seasons")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. Visual check: NDVI phase scatter
# ------------------------------------------------------------

if "SWC_shallow_7d" in df.columns:

    plt.figure(figsize=(8, 6))

    phases_to_plot = [
        "ndvi_greenup",
        "ndvi_peak_window",
        "ndvi_browndown",
        "ndvi_post_browndown",
    ]

    for phase in phases_to_plot:
        sub = df[df["ndvi_phase"] == phase].dropna(
            subset=["SWC_shallow_7d", NDVI_COL]
        )

        if sub.empty:
            continue

        plt.scatter(
            sub["SWC_shallow_7d"],
            sub[NDVI_COL],
            s=22,
            alpha=0.65,
            label=phase
        )

    plt.title("NDVI versus shallow SWC by NDVI phase")
    plt.xlabel("SWC shallow, 7-day mean")
    plt.ylabel(NDVI_COL)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 13. Plot: GPP and NDVI interpolated, with NDVI green-up/brown-down phases
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\Richa\Documents\Python_Projects\Kapiti_Seasonal")

INPUT_FILE = PROJECT_DIR / "Kapiti_Seasonal_DAILY_WITH_GPP_AND_NDVI_PHASES.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ------------------------------------------------------------
# Make sure columns are numeric
# ------------------------------------------------------------

for col in ["GPP_7d", "NDVI_interp_7d"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

NDVI_SCALE = 50

plt.figure(figsize=(17, 7))

# ------------------------------------------------------------
# Plot GPP
# ------------------------------------------------------------

plt.plot(
    df["date"],
    df["GPP_7d"],
    color="black",
    linewidth=1.5,
    label="GPP, 7-day mean"
)

# ------------------------------------------------------------
# Plot NDVI interpolated as background line
# ------------------------------------------------------------

plt.plot(
    df["date"],
    df["NDVI_interp_7d"] * NDVI_SCALE,
    color="grey",
    linewidth=1.2,
    alpha=0.45,
    label=f"NDVI interpolated, 7-day mean x{NDVI_SCALE}"
)

# ------------------------------------------------------------
# Plot NDVI green-up phase in green
# ------------------------------------------------------------

greenup = df[df["ndvi_phase"] == "ndvi_greenup"].copy()

if not greenup.empty:
    for _, block in greenup.groupby(
        (greenup["date"].diff().dt.days.ne(1)).cumsum()
    ):
        plt.plot(
            block["date"],
            block["NDVI_interp_7d"] * NDVI_SCALE,
            color="green",
            linewidth=2.2,
            label="NDVI green-up" if _ == greenup.groupby(
                (greenup["date"].diff().dt.days.ne(1)).cumsum()
            ).ngroup().min() else None
        )

# ------------------------------------------------------------
# Plot NDVI brown-down phase in orange
# ------------------------------------------------------------

browndown = df[df["ndvi_phase"] == "ndvi_browndown"].copy()

if not browndown.empty:
    for _, block in browndown.groupby(
        (browndown["date"].diff().dt.days.ne(1)).cumsum()
    ):
        plt.plot(
            block["date"],
            block["NDVI_interp_7d"] * NDVI_SCALE,
            color="orange",
            linewidth=2.2,
            label="NDVI brown-down" if _ == browndown.groupby(
                (browndown["date"].diff().dt.days.ne(1)).cumsum()
            ).ngroup().min() else None
        )

# ------------------------------------------------------------
# Plot formatting
# ------------------------------------------------------------

plt.title("GPP and NDVI green-up / brown-down phases")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
