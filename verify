# Dangerous humid heat increasingly overlaps recovery after tropical cyclone landfalls: code and result tables

This repository contains the analysis scripts, result tables, and derived intermediate files supporting the study **“Dangerous humid heat increasingly overlaps recovery after tropical cyclone landfalls.”** It also includes a verification script for checking the headline results against the bundled tables.

Release date: 15 November 2026.

## Repository layout

| Path | Contents |
| --- | --- |
| `verify.py` | Recomputes 48 headline checks from the bundled result tables and compares them with the reported values. Two additional checks require the separate 114 MB intermediates archive described below. |
| `run.py` | Runs a stage or script from the repository root, regardless of the current working directory. |
| `requirements.txt` | Package versions used for the reported analyses (Python 3.13). |
| `pilot/` | Analysis scripts and the tables they write (`.csv`, small `.pkl`, `.npz`). |
| `pilot/legacy/` | Earlier versions and exploratory scripts, retained for the record; these are not required to reproduce the reported study results. |
| `data/` | Small intermediate inputs, including CMIP6 slopes and human-induced changes at cells and counties, ERA5 point lists, station lists, and the EAGLE-I data-quality index. Large raw inputs are downloaded by the scripts listed below. |

All scripts use paths relative to the repository root (`pilot/...`, `data/...`). Run them from the root directory or through `run.py`, which changes to the repository root before execution.

## Quick checks (no downloads required)

```bash
pip install -r requirements.txt
python3 run.py verify
