# Spatial Score Agent

## Purpose
Translate OSC session logs into human-readable spatial scores (CSV + plots) so composers and analysts can audit temporal-spatial intent.

## Triggers
- Completion of any OSC session log (live or offline).
- Requests from supervisors/performers for documentation of spatial trajectories.

## Inputs
- Session log path from the OSC Sender or Mock Renderer agent.
- Optional metadata describing object labels and color mapping for plots.

## Outputs
- CSV file with columns `timecode`, `object_id`, `azimuth`, `elevation`, `distance`, `spread`, `active_rule_id`.
- Matplotlib PDF/PNG plotting trajectories over time for each object.
- Brief markdown summary noting salient spatial behaviors.

## Procedure
1. Parse the OSC log, extracting timestamped parameter states per object, interpolating if necessary for missing frames.
2. Build the spatial score CSV and store it under `outputs/spatial_scores/`.
3. Generate trajectory plots (time vs. azimuth/elevation/spread) for each ADM object and save under `outputs/plots/spatial_scores/`.
4. Flag anomalies (e.g., sudden jumps, missing rule IDs) and report them to the Rule Engine Agent for follow-up.
5. Share artifacts with the analysis team for reintegration into verification workflows.

## Handoffs
Spatial score CSV + plots → Analysis Pipeline & documentation agents.
