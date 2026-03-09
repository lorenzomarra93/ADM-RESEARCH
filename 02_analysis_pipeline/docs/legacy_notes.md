Dolby Atmos Analysis Toolkit

- analyze_bed.py → analizza beds 7.1.2 (10 canali) per feature spaziali.
- analyze_adm_objects.py → parsifica metadata ADM degli objects e produce timeline normalizzata, KPI per oggetto e metriche globali.
- analyze_bed_zoom.py → variante compatta per zoom rapidi.

**Beds 7.1.2**
- Input: WAV interleaved 7.1.2
- Output: `bed_features.csv` con colonne `x_lr`, `y_fb`, `z_height`, `zone`; PNG (`traj_xy`, `FB_index`, `LR_index`, `spread`, `height`, `cy_speed`, `bed_axis_density`, `bed_zone_density`) + summary su console/JSON opzionale
- Dettagli: vedi commenti in `analyze_bed.py`

**ADM Objects**
- Input: WAV ADM BWF (con chunk axml) oppure file XML ADM standalone
- Output: `objects_timeline.csv`, `objects_summary.csv`, `objects_global_summary.json`, PNG opzionali (`--plots`)
- Coordinate esportate: `x_lr` (left/right), `y_fb` (front/back), `z_height` (quota), con tolleranza ±0.02 per classificare il centro
- PNG generati: conteggio oggetti attivi, XY scatter, istogramma velocità, confronto front/center/rear & left/center/right, densità per zona
- Parametri principali: `--hop-ms`, `--motion-threshold`, `--probe`
- Esempio:
  `python analyze_adm_objects.py --adm path/to/mix.wav --plots --outdir out_objects`

Requisiti Python comuni: numpy, pandas, matplotlib, soundfile, scipy (opzionale per smoothing).
