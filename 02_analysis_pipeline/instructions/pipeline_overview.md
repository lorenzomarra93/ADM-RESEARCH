# Pipeline Overview

1. Register datasets in `shared_resources/data_inventory.md`.
2. For ADM/object workflows run `scripts/02_object_adm_analysis/parse_adm.py` to convert assets into structured JSON.
3. Run `scripts/02_object_adm_analysis/extract_features.py` to compute descriptors.
4. Merge complementary datasets via `scripts/02_object_adm_analysis/merge_datasets.py`.
5. Execute `scripts/02_object_adm_analysis/run_correlation.py` and `scripts/02_object_adm_analysis/generate_plots.py` for analysis.
6. Produce academic/technical reports with `scripts/02_object_adm_analysis/export_report.py`.
7. When working on fixed multichannel beds, prototype metrics inside `scripts/01_multichannel_analysis/`.

Document deviations from this flow in experiment-specific notes.
