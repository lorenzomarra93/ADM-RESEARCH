# Parameter Space Mapping Pipeline

## Purpose
This pipeline ingests normalized musical descriptors coming from the Analysis Pipeline (or directly from live MIDI/audio streams) and translates them into ADM-OSC messages delivered to Nuendo or SPAT Revolution, turning spatial behavior into an explicit compositional parameter.

## Input Contracts
- Expect a Python `dict` (or JSON-equivalent) whose keys exactly match `trigger_parameter` names defined by the CRE (`tension_score`, `rms_db`, `onset_density`, `spectral_centroid`, `voice_count`, `interval_class`, `form_state`).
- Values must be floats normalized to `[0.0, 1.0]` unless a descriptor specification states otherwise (e.g., categorical `form_state` labels).
- Each descriptor bundle carries `timestamp_s` (seconds since session start) to guarantee deterministic replay and alignment with OSC scheduling.

## Output Contracts
- ADM-OSC messages abiding by the EBU specification. Positional controls use `/adm/obj/<id>/cartesian` and diffusion controls use `/adm/obj/<id>/spread`.
- Every dispatched message is mirrored into `data/osc_logs/session_<ISO8601_timestamp>.log` for verification and downstream agents (e.g., spatial scoring).
- Optional human-facing CSV/plots are produced by dedicated agents but must derive from the same log to ensure traceability.

## Rule Engine Location
`src/cre/rule_engine.py` is the central orchestrator. All sub-agents either populate its inputs (descriptor DataFrames, rule sets) or consume its outputs (spatial states, OSC logs). New experiments should extend this module rather than forking ad-hoc scripts.

## Testing Protocol
1. Author rule sets as JSON and store them under `configs/rules/`.
2. Run the pipeline against `src/cre/mock_osc_server.py` (a lightweight OSC echo server) before touching Nuendo/SPAT to ensure deterministic behavior without DAW dependencies.
3. Unit-test each rule scope (micro/meso/macro) in isolation by feeding canned descriptor dictionaries and asserting OSC/log outputs.
4. Archive session logs and test results alongside case studies so the analysis pipeline can re-ingest them for post-hoc evaluation.
