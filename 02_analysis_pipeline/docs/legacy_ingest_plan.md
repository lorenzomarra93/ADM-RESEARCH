# Legacy Data-Analysis Assets

Source folder: `03_data_analysis/03.1_ADM analysis`

## Scripts (archived for provenance)
- Multichannel beds (5.1/7.1/7.1.2): moved to `02_analysis_pipeline/scripts/01_multichannel_analysis/legacy/`
  - `analyze_bed.py`
  - `analyze_surround71.py`
  - `analyze_multichannel_mix.py`
  - `test_spectral.py` (spectral sanity checks)
- ADM/object analysis prototypes: moved to `02_analysis_pipeline/scripts/02_object_adm_analysis/legacy/`
  - `analyze_adm_objects.py`
  - `analyze_adm_objects_integrazione_spettro.py`
  - `analyze_adm_objects_streamlite.py`
- OSC control prototype relocated to parameter-space mapping agents: `03_parameter_space_mapping_pipeline/scripts/legacy/osc_object_control.py`
- Streamlit viewers relocated under `02_analysis_pipeline/scripts/02_object_adm_analysis/apps/`
  - `streamlit_tsf_app.py`
  - `streamlit_tsf_simple.py`
- Bed/Object comparison utility now lives at `02_analysis_pipeline/scripts/02_object_adm_analysis/compare_bed_objects.py`

## Data & Outputs (to redistribute)
- Raw audio testbed: `Sine Zoom Frontale.wav`
- CSV: `bed_features.csv`, `test_spectral_features.csv`, `out_objects/objects_timeline.csv`, etc.
- Plots: `FB_index.png`, `LR_index.png`, `traj_xy.png`, `spread.png`, `height.png`, `cy_speed.png`, PNG bundles under `out_bed`, `out_objects`, `out_compare`.
- JSON: `out_objects/summary.json`, `out_compare/compare_metrics.json`.
- README + requirements: original notes under `README.txt`, package list under `requirements.txt`.

## Redistribution Targets
- Raw audio → `02_analysis_pipeline/data/raw/legacy_sessions/`
- Derived CSV/JSON → `02_analysis_pipeline/outputs/{csv,json}/legacy_runs/`
- Plots → `02_analysis_pipeline/outputs/plots/legacy_runs/`
- Legacy README → `02_analysis_pipeline/docs/legacy_notes.md`
- Requirements snapshot → `02_analysis_pipeline/configs/requirements_legacy.txt`

## Cleaning Goals
1. Promote unified multichannel CLI (`multichannel_motion_analysis.py`) covering 5.1/7.1/7.1.2.
2. Integrate ADM parsing + motion KPIs into `adm_motion_analysis.py` used by `adm_parser_agent` and `feature_extraction_agent`.
3. Keep comparison + visualization utilities modular so `visualization_agent` and `report_agent` can orchestrate them.
4. Document provenance of every migrated dataset inside `shared_resources/data_inventory.md`.
