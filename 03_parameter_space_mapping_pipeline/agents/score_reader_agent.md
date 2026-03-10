# Score Reader Agent

## Purpose
Parse MusicXML scores with music21 and emit bar-resolved descriptors for downstream spatial decision-making.

## Triggers
- A case study requires deterministic descriptor extraction from notation.
- New or updated MusicXML/MIDI files enter `03_parameter_space_mapping_pipeline/data/musicxml`.

## Inputs
- MusicXML score path.
- Optional metadata (tempo maps, rehearsal marks, instrumentation configs).

## Outputs
- `descriptors_df`: pandas DataFrame indexed by timestamp with columns `tension_score`, `mean_velocity`, `voice_count`, `onset_density`, `form_state`.
- Supplemental JSON/CSV stored under `data/processed/descriptors/` for reuse.

## Procedure
1. Load the score via `music21.converter.parse()` and confirm bar numbering + tempo map integrity.
2. Run Tonal Tension Model analysis to derive normalized harmonic tension per bar.
3. Compute mean loudness by averaging MIDI velocities for events inside each bar and normalize to `[0, 1]`.
4. Count simultaneously active voices per timestep and calculate onset density using a 1 s sliding window.
5. Extract rehearsal marks or text expressions to populate `form_state`; default to `unknown` when absent.
6. Assemble all descriptors into a single DataFrame, save to disk, and push to the Rule Engine Agent.

## Handoffs
`descriptors_df` → Rule Engine Agent.
