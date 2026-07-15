# Kapiti Seasonal Pre-Analysis

This folder contains the scripts used to inspect, merge, clean and restructure the source datasets for the Kapiti seasonal analysis.

The workflow consists of three stages:

```text
1. Explore and document source files
                  ↓
2. Aggregate and merge all datasets by date
                  ↓
3. Create an analysis-ready daily dataset
```

## 1. Dataset Column Explorer and Data Dictionary

This script scans all CSV, XLSX and XLS files in the Kapiti project directory and documents their structure before any datasets are merged.

It:

* Records each file’s dimensions, columns and memory use.
* Summarises column data types, missingness, unique values and example values.
* Calculates numeric minima, maxima, means and medians.
* Estimates variable meanings from column names.
* Groups variables by seasonal-analysis relevance, including rainfall, soil moisture, GPP, NDVI, temperature and carbon fluxes.
* Creates a matrix showing which columns occur in each source file.
* Saves the first ten rows of each dataset as preview files.

### Main output folder

```text
Kapiti_Seasonal_column_exploration_outputs
```

This stage provides the initial dataset inventory and data dictionary.

## 2. Mega Merge All CSV Files

This script reads the footprint, meteorology, NDVI and annual FLUXNET datasets and combines them into one daily dataset.

It:

* Cleans column names and standardises missing-value codes.
* Parses source-specific timestamp formats:

  * Footprint: `yyyy/mm/day/HH/MM`
  * Meteorology: `Year/DoY/Hour`
  * FLUXNET: `TIMESTAMP_START`
* Converts appropriate columns to numeric values.
* Aggregates each dataset to daily resolution:

  * Rainfall variables are summed.
  * Other numeric variables are averaged.
* Prefixes variables with their source, for example:

```text
meteorology__Tair
fluxnet_2023__SWC_1_1_1
ndvi__Weighted
```

* Outer-merges all datasets by date.
* Adds year, month, day of year and nominal rainfall-season variables.
* Produces diagnostics describing which files were successfully processed.

### Main outputs

```text
Kapiti_Seasonal_MEGA_MERGED_DAILY.csv
Kapiti_Seasonal_merge_diagnostics.csv
```

This stage performs the raw integration of all source datasets.

## 3. Create Analysis-Ready Daily Dataset

This script converts the mega-merged file into a cleaner dataset for seasonal and ecohydrological analysis.

It:

* Combines annual FLUXNET columns into continuous variables across years.
* Extracts core variables including:

  * Rainfall
  * Soil moisture
  * NEE
  * GPP
  * NDVI
  * Temperature
  * Vapour pressure deficit
  * Radiation
* Adds selected meteorology and footprint variables.
* Creates simplified analysis aliases such as:

```text
SWC_shallow
SWC_middle
SWC_deep
GPP
NDVI
Tair
VPD
Rg
```

* Uses alternative data sources where the preferred variable is missing.
* Calculates rolling variables, including:

  * 7-, 14- and 30-day rainfall totals
  * 7-day soil-moisture means
  * 7-day GPP means
  * Three-observation NDVI means
* Assigns nominal long-rain, short-rain and dry or transition periods.
* Creates a simple wetting and drying phase classification from changes in shallow soil moisture.
* Reports missingness for the main analysis variables.

### Main outputs

```text
Kapiti_Seasonal_ANALYSIS_READY_DAILY.csv
Kapiti_Seasonal_analysis_ready_diagnostics.csv
```

This stage performs the final variable selection, harmonisation and feature engineering.

## Recommended execution order

Run the scripts in the following order:

1. Dataset Column Explorer and Data Dictionary
2. Mega Merge All CSV Files
3. Create Analysis-Ready Daily Dataset

The first script identifies and documents the available data, the second combines the source datasets at daily resolution, and the third restructures the merged data for seasonal, hysteresis and ecohydrological analyses.
