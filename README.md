# Nighttime Lights and Ethiopia Research Repo

This repository is organized for an end-to-end empirical workflow: data preparation, analysis, manuscript development, and output storage.

## Current Structure

- `data/`
  - `raster/`: raw and clipped nighttime-light rasters
  - `shapefiles/`: administrative boundary files
  - `tabular/`: processed national and regional tabular datasets used by the active scripts
- `scripts/`
  - `analysis/`: main empirical analysis scripts
  - `data_processing/`: extraction and data-construction scripts
  - `common/`: shared project path configuration
  - top-level files in `scripts/`: backwards-compatible wrappers so older commands still work
- `outputs/`
  - `figures/`: generated figures
  - `results/`: generated regression summaries, coefficient tables, and derived outputs
- `notebooks/`: exploratory and verification notebooks
- `manuscript/`: supporting manuscript files, notes, and build artifacts
- `Manuscript_NTL_revised.tex`: main LaTeX manuscript entry file

## Main Entry Points

- National validation: `python scripts/analyze_ntl_gdp.py`
- Regional inequality (Gini): `python scripts/analyze_regional_panel.py`
- Event study: `python scripts/analyze_event_study.py`
- Excluding special city-regions: `python scripts/analyze_regional_panel_exclude_special.py`
- Rebuild panel data: `python extract_ntl_panel.py`

## Notes

- The active code now uses a shared paths module at `scripts/common/paths.py`.
- Legacy script entry points are preserved as wrappers, so older commands should still run.
- The manuscript is being revised against the outputs in `outputs/results/` rather than hand-entered numbers.
