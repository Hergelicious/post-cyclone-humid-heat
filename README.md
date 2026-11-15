```markdown
# Dangerous humid heat increasingly overlaps recovery after tropical cyclone landfalls: code and result tables

This release holds the analysis scripts, the result tables behind every number in the paper, the small
intermediate files, and a script that checks the headline numbers. Release of 15 November 2026.

## Layout

| Path | Contents |
| --- | --- |
| `verify.py` | Recomputes 48 headline numbers from the bundled tables and compares them with the published values |
| `run.py` | Runs a stage or a script from the release root, whatever the current directory |
| `requirements.txt` | Package versions used for the published results (Python 3.13) |
| `pilot/` | Analysis scripts and the tables they write (`.csv`, small `.pkl`, `.npz`) |
| `pilot/legacy/` | Earlier versions and exploratory scripts, kept for the record; not needed for any result in the paper |
| `data/` | Small intermediate inputs (CMIP6 slopes and human-induced changes at cells and counties, ERA5 point list, station lists, EAGLE-I data-quality index). Large raw inputs are downloaded by the scripts listed below |
| `manuscript/` | Anonymised manuscript and Supplementary Information (Word) |

All scripts use paths relative to the release root (`pilot/...`, `data/...`). Run them from the root, or through
`run.py`, which changes to the root first.

## Quick check (about one minute, no downloads)

```bash
pip install -r requirements.txt        # into an existing environment; numpy and pandas are enough for this step
python3 run.py verify                  # 48 checks; prints PASS or FAIL for each and exits with 0 if all pass

```

With the separate intermediates archive unpacked into the root (`pilot/us_v2_recs_main.pkl`, 114 MB), `verify.py` adds two
checks computed from the county-by-hour records, and the outage analyses of stage 5 marked * can be rerun without downloads.

## Full rebuild from public data

Stages run in order. Each line gives the script, what it reads and what it writes. Downloads need internet access to the
listed hosts; sizes are approximate.

### Stage 0. Public inputs

| Source | Script | Output |
| --- | --- | --- |
| IBTrACS v04r01, JTWC subset, as shipped with huracanpy 1.5.0 | `pilot/landfalls.py` | `pilot/landfalls.csv` (1,568 landfalls) |
| GHS-POP R2023A (jeodpp.jrc.ec.europa.eu) | `pilot/ghsl.py` | `data/ghsl/` (0.5 GB) |
| ERA5 hourly 2 m temperature, dewpoint, surface pressure, 0.25 deg (ARCO-ERA5, storage.googleapis.com/gcp-public-data-arco-era5, store `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`; time index 0 is 1 January 1900) | `pilot/era5q_extract.py` | `data/era5q/YYYYMMDD.npz` (15,209 days, 0.7 GB) |
| ERA5 10 m wind gust, same store | `pilot/era5_gust_footprint.py` | `pilot/era5_gust_cells.parquet` (bundled) |
| ERA5 1 deg hourly (same bucket, 1959-2022 store) and ERA5 wet-bulb temperature (Zenodo 10418938) | `pilot/era5h_arco.py`, `pilot/era5_precip.py`, `pilot/fetch_wbt.py`, `pilot/era5h_tw.py` | `data/era5h/`, `data/wbt/` |
| NASA POWER daily and hourly (power.larc.nasa.gov) | `pilot/fetch_power.py`, `pilot/fetch_neigh.py`, `pilot/fetch_hourly2.py`, `pilot/fetch_hourly_cells.py`, `pilot/fetch_hourly_trend.py`, `pilot/power_td_fetch.py` | `pilot/power_*.csv`, `pilot/hourly_cells/`, `data/power_td/` |
| GSDM-WBT stations (Zenodo 7014332) | `pilot/gsdm_fetch.py` | `data/gsdm/` |
| CMIP6 monthly tasmax, hurs, tas (Pangeo catalogue, storage.googleapis.com/cmip6) | `pilot/cmip6_extract.py`, `pilot/cmip6_extract_attr.py`, `pilot/cmip6_cell_deltas.py` (cells); `pilot/cmip6_extract_county.py`, `pilot/cmip6_county_deltas.py` (counties) | `data/cmip6/`, `data/cmip6cty/` (2.3 GB); the derived `slopes26.npz` and `humandelta15.npz` are bundled |
| EAGLE-I outage records (figshare, doi:10.6084/m9.figshare.24237376.v4) | `pilot/eaglei_filter.py`, `pilot/eaglei_daily.py` | `data/eaglei/win_YYYY.parquet`, `daily_YYYY.parquet` (0.4 GB) |
| HDNTL daily night lights (Zenodo 17079409) | `pilot/fetch_hdntl_all.py` | `data/hdntl/` (6 GB) |
| NWS heat products and forecast zones (mesonet.agron.iastate.edu) | `pilot/us_heat_warnings.py` | `data/nws/`, `pilot/us_heat_warnings.csv` |
| HRRR forecasts (storage.googleapis.com/high-resolution-rapid-refresh) | `pilot/hrrr_extract.py` | `data/hrrr/` |
| HHS emPOWER county data (services2.arcgis.com REST service) | `pilot/us_empower.py` | `data/empower/`, table rows in Supplementary Table 21 |
| GISTEMP v4 global mean temperature | bundled | `pilot/gmst.csv` |

### Stage 1. NASA POWER exposure records (independent check and the record list reused by the ERA5 analysis)

`pilot/power_more.py`, `pilot/exposure.py`, `pilot/reconcile.py` -> `pilot/recon_records.pkl` (bundled), `pilot/recon_*.csv`.

### Stage 2. Main global analysis (ERA5 0.25 deg): Supplementary Tables 2, 5, 8, 9, 13, 15, 16, 22

```bash
python3 pilot/era5_rebuild.py                                                  # exposure, regression, attribution, projections
WLS=3.3 ATT=0 PROJOUT=pilot/era5_projections33.csv python3 pilot/era5_rebuild.py   # 3.3 C sensitivity
python3 pilot/era5_extra.py                                                    # ratios, countries, decomposition, nights, recurrence
python3 pilot/era5_gust_eval.py                                                # gust-footprint sensitivity
python3 pilot/era5_vs_power.py ; python3 pilot/station_arbiter.py              # dataset comparison and station check

```

### Stage 3. Supporting global analyses (Supplementary Tables 1, 3, 4, 6, 7, 10 to 12, 14, 17, 18; Supplementary Note 1)

`stations.py`, `stations_bg.py`, `harden.py`, `robust.py`, `hourly_check.py`, `calib_sens.py` (regressions and calibration);
`era5.py`, `era5_night.py`, `era5_decomp.py`, `decomp_met.py`, `decomp_met2.py`, `country_exposure.py`, `corridor_cells.py`,
`corridor_exposure.py`, `spatial_exposure.py`, `exposure_struct.py` (NASA POWER and ERA5 1 deg exposure and decomposition);
`cmip6_proj.py`, `proj33.py`, `cmip6_eval.py`, `hindcast.py`, `attrib3.py`, `attrib_split.py`, `attrib_var.py` (projections and
attribution on NASA POWER); `rain_check.py`, `capacity.py` (rain then heat, recovery capacity); `ml_screen.py`,
`ml_screen_lag.py`, `ext2526.py` (screening model and 2025-2026 check); `hdntl_outage2.py`, `outage_reported.py`,
`outage_heat.py`, `outage_package2.py`, `outage_selection.py` (night-light outages).

### Stage 4. County outage records and heat (United States, 2015-2024)

```bash
python3 pilot/us_outage_v2.py 4 main            # main run -> pilot/us_v2_storms_main.csv, us_v2_curves_main.pkl, us_v2_recs_main.pkl (7 to 10 min)
python3 pilot/us_outage_v2.py 0 gap0            # no gap filling
THR=0.20 python3 pilot/us_outage_v2.py 4 thr20  # 20% county threshold
CM=cell python3 pilot/us_outage_v2.py 4 cell    # CMIP6 change at the nearest landfall cell

```

Needs `data/eaglei/`, `data/era5q/`, `data/cmip6cty/slopes26.npz` and `humandelta15.npz` (bundled) and `data/geo/`.

### Stage 5. Outage results: Supplementary Tables 19 to 21 and 23

| Script | Result |
| --- | --- |
| `us_v2_summary.py main` * | Totals, human shares, warming ratios, restoration factors (Supplementary Table 19) |
| `us_v2_peaks.py`, `us_v2_extra.py` * | Hazardous daily peaks, strata, per-storm medians (Supplementary Tables 19 to 21) |
| `us_continuity_bounds.py` * | Share of exposure after 24, 48 and 72 h without power |
| `us_overlap_null.py`, `us_restoration_heat.py`, `us_restoration_heat_rob.py` | Pre-landfall heat swap; restoration speed against heat |
| `us_control_15min.py`, `us_outage_control.py` | Non-storm controls |
| `us_heat_warnings.py`, `us_empower.py` | Heat products in effect; electricity-dependent Medicare beneficiaries |
| `us_missing_check.py` * | Sensitivity to possibly unreported outage intervals |
| `us_v3_new.py` * | Restoration milestones, time to first hazardous hour, restoration stage, 72 h cap, intensity-weighted exposure |
| `us_v3_robust.py` * | Two bounded quantities for repeated exposure; whole-curve restoration metric; timing robustness |
| `us_v5_transfer.py` * | Restoration transfer test (Supplementary Table 23) |
| `hrrr_eval.py`, `hrrr_eval2.py`, `hrrr_storm_level.py` | Forecast test and its storm-level evaluation |

* runs from the intermediates archive alone (plus the bundled `data/cmip6cty/*.npz`).

## Key result tables

| Table | Contents |
| --- | --- |
| `pilot/era5_annual_lmic.csv`, `era5_annual_country.csv`, `era5_country.csv` | Annual LMIC person-days with HI >= 40.6 C, total and by country |
| `pilot/era5_regression.csv`, `era5_ratios.csv`, `era5_decomposition.csv` | Sensitivity per degree, period ratios, Shapley decomposition |
| `pilot/era5_attribution.csv`, `era5_projections.csv`, `era5_projections33.csv` | Human share by realisation; warming-level projections by model |
| `pilot/era5_vs_power_daily.csv`, `era5_vs_power_hourly.csv`, `station_arbiter.csv` | ERA5 against NASA POWER at the same cells and days; station check |
| `pilot/us_v2_storms_main.csv` | Per-storm outage and heat totals in the observed, natural-only and warming-level climates |
| `pilot/us_v4_bounds.csv`, `pilot/us_v4_restoration_R.csv`, `pilot/us_v4_lead_robust.csv`, `pilot/us_v5_transfer.csv` | Repeated-exposure bounds, restoration performance, time to first hazardous hour, restoration transfer test |
| `pilot/us_v3_events.csv` | One row per county outage: peak, restoration milestones, customer-hours, exposure |

