# Data Inventory

| Dataset | Location | Stage | Notes |
| --- | --- | --- | --- |
| Legacy ADM tests | 02_analysis_pipeline/data/raw | raw | To ingest from `03_data_analysis/03.1_ADM analysis`. |
| Sine Zoom Frontale (legacy bed/object test) | 02_analysis_pipeline/data/raw/legacy_sessions | raw | Formerly `03_data_analysis/03.1_ADM analysis/Sine Zoom Frontale.wav`. |
| OSC object control | 03_parameter_space_mapping_pipeline/data/control_data | raw | Port from `OSC to object/`. |
| Mix03 visuals | 02_analysis_pipeline/outputs/plots | processed | Derived from `out_tp_mix03/`. |
| Legacy bed analyses | 02_analysis_pipeline/outputs/{csv,plots}/legacy_runs/bed_analysis | processed | Migrated from `out_bed/` (FB/LR panels, bed_features.csv). |
| Legacy ADM object analyses | 02_analysis_pipeline/outputs/{csv,plots,json}/legacy_runs/object_analysis | processed | Migrated from `out_objects/` exports. |
| Legacy bed vs object comparisons | 02_analysis_pipeline/outputs/{csv,plots,json}/legacy_runs/compare | processed | Migrated from `out_compare/`. |

Update this table whenever data moves between lifecycle stages.
