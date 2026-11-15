# Dangerous humid heat increasingly overlaps recovery after tropical cyclone landfalls: code, result tables and figures

This repository contains the analysis scripts, result tables, derived intermediate files, and figures supporting the study **“Dangerous humid heat increasingly overlaps recovery after tropical cyclone landfalls.”** It also includes a verification script for checking the headline results against the bundled tables.

Release date: 15 November 2026.

## Repository layout

| Path | Contents |
| --- | --- |
| `verify.py` | Recomputes 48 headline checks from the bundled result tables and compares them with the reported values. Two additional checks require the separate 114 MB intermediates archive described below. |
| `run.py` | Runs a stage or script from the repository root, regardless of the current working directory. |
| `requirements.txt` | Package versions used for the reported analyses (Python 3.13). |
| `pilot/` | Analysis scripts and the tables they write (`.csv`, small `.pkl`, `.npz`); `pilot/style.py` contains the figure style. |
| `pilot/legacy/` | Earlier versions and exploratory scripts, retained for the record; these are not required to reproduce the reported study results. |
| `data/` | Small intermediate inputs, including CMIP6 slopes and human-induced changes at cells and counties, ERA5 point lists, station lists, and the EAGLE-I data-quality index. Large raw inputs are downloaded by the scripts listed below. |
| `figs/` | Figures 1–5 (PNG and PDF) and the Supplementary Figures. |

All scripts use paths relative to the repository root (`pilot/...`, `data/...`, `figs/...`). Run them from the root directory or through `run.py`, which changes to the repository root before execution.

## Quick checks (no downloads required)

```bash
pip install -r requirements.txt
python3 run.py verify
python3 run.py figures
```

The verification command runs **48 checks** using the bundled result tables. It prints `PASS` or `FAIL` for each check and exits with status 0 if all checks pass.

The figure command redraws Figures 1–5 from the bundled tables.

### Additional outage verification

A separate intermediates archive (114 MB), containing:

`pilot/us_v2_recs_main.pkl`

provides the county-by-hour records needed for two additional verification checks. After downloading and unpacking this archive into the repository root, `verify.py` runs these two additional checks.

The archive also enables the Stage 5 outage analyses marked with an asterisk (`*`) to be rerun without downloading the corresponding intermediate records. The archive is not required for the 48 standard verification checks or for redrawing Figures 1–5 from the bundled tables.

## Full rebuild from public data

The analysis stages are designed to run in sequence. Each stage below identifies the scripts, inputs, and outputs. Downloads require internet access to the listed public sources; approximate data sizes are indicated where available.

### Stage 0. Public inputs

| Source | Script | Output |
| --- | --- | --- |
| IBTrACS v04r01, JTWC subset, as shipped with huracanpy 1.5.0 | `pilot/landfalls.py` | `pilot/landfalls.csv` (1,568 landfalls) |
| GHS-POP R2023A (jeodpp.jrc.ec.europa.eu) | `pilot/ghsl.py` | `data/ghsl/` (0.5 GB) |
| ERA5 hourly 2 m temperature, dewpoint, and surface pressure, 0.25° (ARCO-ERA5, `storage.googleapis.com/gcp-public-data-arco-era5`; store `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`; time index 0 is 1 January 1900) | `pilot/era5q_extract.py` | `data/era5q/YYYYMMDD.npz` (15,209 days, 0.7 GB) |
| ERA5 10 m wind gust, same store | `pilot/era5_gust_footprint.py` | `pilot/era5_gust_cells.parquet` (bundled) |
| ERA5 1° hourly (same bucket, 1959–2022 store) and ERA5 wet-bulb temperature (Zenodo 10418938) | `pilot/era5h_arco.py`, `pilot/era5_precip.py`, `pilot/fetch_wbt.py`, `pilot/era5h_tw.py` | `data/era5h/`, `data/wbt/` |
| NASA POWER daily and hourly (`power.larc.nasa.gov`) | `pilot/fetch_power.py`, `pilot/fetch_neigh.py`, `pilot/fetch_hourly2.py`, `pilot/fetch_hourly_cells.py`, `pilot/fetch_hourly_trend.py`, `pilot/power_td_fetch.py` | `pilot/power_*.csv`, `pilot/hourly_cells/`, `data/power_td/` |
| GSDM-WBT stations (Zenodo 7014332) | `pilot/gsdm_fetch.py` | `data/gsdm/` |
| CMIP6 monthly tasmax, hurs, tas (Pangeo catalogue, `storage.googleapis.com/cmip6`) | `pilot/cmip6_extract.py`, `pilot/cmip6_extract_attr.py`, `pilot/cmip6_cell_deltas.py` (cells); `pilot/cmip6_extract_county.py`, `pilot/cmip6_county_deltas.py` (counties) | `data/cmip6/`, `data/cmip6cty/` (2.3 GB); derived `slopes26.npz` and `humandelta15.npz` are bundled |
| EAGLE-I outage records (Figshare, DOI: 10.6084/m9.figshare.24237376.v4) | `pilot/eaglei_filter.py`, `pilot/eaglei_daily.py` | `data/eaglei/win_YYYY.parquet`, `daily_YYYY.parquet` (0.4 GB) |
| HDNTL daily night lights (Zenodo 17079409) | `pilot/fetch_hdntl_all.py` | `data/hdntl/` (6 GB) |
| NWS heat products and forecast zones (`mesonet.agron.iastate.edu`) | `pilot/us_heat_warnings.py` | `data/nws/`, `pilot/us_heat_warnings.csv` |
| HRRR forecasts (`storage.googleapis.com/high-resolution-rapid-refresh`) | `pilot/hrrr_extract.py` | `data/hrrr/` |
| HHS emPOWER county data (services2.arcgis.com REST service) | `pilot/us_empower.py` | `data/empower/`, table rows in Supplementary Table 21 |
| GISTEMP v4 global mean temperature | Bundled | `pilot/gmst.csv` |

### Stage 1. NASA POWER exposure records

Independent check and record list reused by the ERA5 analysis.

`pilot/power_more.py`, `pilot/exposure.py`, `pilot/reconcile.py` produce `pilot/recon_records.pkl` (bundled) and `pilot/recon_*.csv`.

### Stage 2. Main global analysis (ERA5 0.25°)

Supports Figures 1–3 and Supplementary Tables 2, 5, 8, 9, 13, 15, 16, and 22.

```bash
python3 pilot/era5_rebuild.py
WLS=3.3 ATT=0 PROJOUT=pilot/era5_projections33.csv python3 pilot/era5_rebuild.py
python3 pilot/era5_extra.py
python3 pilot/era5_gust_eval.py
python3 pilot/era5_vs_power.py
python3 pilot/station_arbiter.py
```

The scripts perform the exposure, regression, attribution, projection, sensitivity, ratio, country, decomposition, night-time, recurrence, gust-footprint, dataset-comparison, and station-check analyses.

### Stage 3. Supporting global analyses

Supports Supplementary Tables 1, 3, 4, 6, 7, 10–12, 14, 17, and 18; Supplementary Figures 1–9; and Supplementary Note 1.

- Regressions and calibration: `stations.py`, `stations_bg.py`, `harden.py`, `robust.py`, `hourly_check.py`, `calib_sens.py`.
- NASA POWER and ERA5 1° exposure and decomposition: `era5.py`, `era5_night.py`, `era5_decomp.py`, `decomp_met.py`, `decomp_met2.py`, `country_exposure.py`, `corridor_cells.py`, `corridor_exposure.py`, `spatial_exposure.py`, `exposure_struct.py`.
- Projections and attribution on NASA POWER: `cmip6_proj.py`, `proj33.py`, `cmip6_eval.py`, `hindcast.py`, `attrib3.py`, `attrib_split.py`, `attrib_var.py`.
- Rainfall, recovery capacity, and screening analyses: `rain_check.py`, `capacity.py`, `ml_screen.py`, `ml_screen_lag.py`, `ext2526.py`.
- Night-light outage analyses: `hdntl_outage2.py`, `outage_reported.py`, `outage_heat.py`, `outage_package2.py`, `outage_selection.py`.

### Stage 4. County outage records and heat (United States, 2015–2024)

```bash
python3 pilot/us_outage_v2.py 4 main
python3 pilot/us_outage_v2.py 0 gap0
THR=0.20 python3 pilot/us_outage_v2.py 4 thr20
CM=cell python3 pilot/us_outage_v2.py 4 cell
```

The main run produces `pilot/us_v2_storms_main.csv`, `us_v2_curves_main.pkl`, and `us_v2_recs_main.pkl` (approximately 7–10 minutes).

The alternative runs test no gap filling, a 20% county threshold, and CMIP6 change at the nearest landfall cell.

Required inputs include `data/eaglei/`, `data/era5q/`, `data/cmip6cty/slopes26.npz`, `data/cmip6cty/humandelta15.npz` (the latter two are bundled), and `data/geo/`.

### Stage 5. Outage results

Supports Figures 4 and 5 and Supplementary Tables 19–21 and 23.

| Script | Result |
| --- | --- |
| `us_v2_summary.py main` * | Totals, human shares, warming ratios, and restoration factors (Supplementary Table 19) |
| `us_v2_peaks.py`, `us_v2_extra.py` * | Hazardous daily peaks, strata, and per-storm medians (Supplementary Tables 19–21) |
| `us_continuity_bounds.py` * | Share of exposure after 24, 48, and 72 hours without power |
| `us_overlap_null.py`, `us_restoration_heat.py`, `us_restoration_heat_rob.py` | Pre-landfall heat swap; restoration speed against heat |
| `us_control_15min.py`, `us_outage_control.py` | Non-storm controls |
| `us_heat_warnings.py`, `us_empower.py` | Heat products in effect; electricity-dependent Medicare beneficiaries |
| `us_missing_check.py` * | Sensitivity to possibly unreported outage intervals |
| `us_v3_new.py` * | Restoration milestones, time to first hazardous hour, restoration stage, 72-hour cap, intensity-weighted exposure |
| `us_v3_robust.py` * | Two bounded quantities for repeated exposure; whole-curve restoration metric; timing robustness (Figure 5b,c) |
| `us_v5_transfer.py` * | Restoration transfer test (Supplementary Table 23) |
| `hrrr_eval.py`, `hrrr_eval2.py`, `hrrr_storm_level.py` | Forecast test and storm-level evaluation |

`*` These analyses can run from the separate intermediates archive alone, together with the bundled `data/cmip6cty/*.npz` files. The archive is described under **Additional outage verification** above.

### Stage 6. Figures

| Figure | Script | Inputs |
| --- | --- | --- |
| Figures 1–3 | `pilot/figs_v6.py` | `pilot/era5_*.csv`, `pilot/coef_table.csv`, `pilot/cmip6_proj.csv`, `pilot/decomp_final.csv` |
| Figures 4 and 5 | `pilot/fig45.py` | `pilot/us_v2_storms_main.csv`, `us_v2_curves_main.pkl`, `us_v4_bounds.csv`, `us_v4_R.pkl`, `us_v4_lead_robust.csv` |
| Supplementary Figures 1–4 | `pilot/figS.py` | Station, hourly, and rainfall tables in `pilot/` |
| Supplementary Figure 5 | `pilot/figS5.py` | `pilot/ml_*_lag.csv` |
| Supplementary Figure 6 | `pilot/figs_v3.py` | Country exposure and capacity tables in `pilot/` |
| Supplementary Figure 7 | `pilot/figS7.py` | NASA POWER and ERA5 tables in `pilot/` |
| Supplementary Figures 8 and 9 | `pilot/figs_v5.py` | `pilot/hdntl_*.csv`, NASA POWER exposure tables |

## Key result tables

| Files | Contents |
| --- | --- |
| `pilot/era5_annual_lmic.csv`, `era5_annual_country.csv`, `era5_country.csv` | Annual LMIC person-days with HI ≥ 40.6 °C, total and by country |
| `pilot/era5_regression.csv`, `era5_ratios.csv`, `era5_decomposition.csv` | Sensitivity per degree, period ratios, and Shapley decomposition |
| `pilot/era5_attribution.csv`, `era5_projections.csv`, `era5_projections33.csv` | Human share by realisation; warming-level projections by model |
| `pilot/era5_vs_power_daily.csv`, `era5_vs_power_hourly.csv`, `station_arbiter.csv` | ERA5 against NASA POWER at the same cells and days; station check |
| `pilot/us_v2_storms_main.csv` | Per-storm outage and heat totals in the observed, natural-only, and warming-level climates |
| `pilot/us_v4_bounds.csv`, `us_v4_restoration_R.csv`, `us_v4_lead_robust.csv`, `us_v5_transfer.csv` | Repeated-exposure bounds, restoration performance, time to first hazardous hour, and restoration transfer test |
| `pilot/us_v3_events.csv` | One row per county outage: peak, restoration milestones, customer-hours, and exposure |

## Notes and interpretation

- Customers are utility accounts summed over storms, not unique households or people.
- Counterfactual climates hold the observed storms, outages, and weather sequences fixed.
- The heat index follows the NWS regression; `pilot/power_more.py` contains the shared function `hi_c`.
- `python3 run.py figures` redraws Figures 1–5 from the bundled tables using the listed package versions.
- The standard verification run checks 48 headline values using the bundled tables. Two additional checks require the separate 114 MB intermediates archive described above.

## License and data availability

The code in this repository is released under the MIT License. See the `LICENSE` file for the full license text.

External datasets remain subject to the terms and conditions of their original providers. Please consult the cited source for the applicable data-use and redistribution terms.
