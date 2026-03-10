# OSC Sender Agent

## Purpose
Transmit ADM-OSC messages derived from spatial state DataFrames to Nuendo, SPAT Revolution, or a mock renderer while ensuring every packet is logged.

## Triggers
- Receipt of a new `spatial_states_df`.
- Scheduled render sessions or real-time performances requiring OSC transmission.

## Inputs
- `spatial_states_df` with per-object spatial parameters and timestamps.
- `configs/osc_config.json` specifying IP, port, transport mode, and logging preferences.
- Optional scheduling instructions (real time vs. offline).

## Outputs
- Live OSC stream to the configured renderer or mock server.
- Session log file written to `data/osc_logs/session_<ISO8601>.log`.
- Status report indicating packet counts, dropped messages, and latency metrics.

## Procedure
1. Load OSC target configuration and instantiate the appropriate python-osc client.
2. Choose mode: offline (pre-computed timeline scheduling) or real time (sleep relative to timestamp deltas).
3. Iterate over each row in `spatial_states_df`, format ADM-OSC address patterns (e.g., `/adm/obj/<id>/cartesian`, `/adm/obj/<id>/spread`), and dispatch values.
4. Mirror every packet into the session log with millisecond timestamps and rule provenance.
5. On completion, summarize statistics and pass the log location to verification agents.

## Handoffs
Session log path → Analysis Pipeline for post-hoc verification.
