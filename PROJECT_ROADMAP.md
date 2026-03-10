# Project Roadmap

## Phase 1 · Infrastructure (Q1 FY26)
- Stand up directory structure, documentation, and placeholder scripts.
- Inventory legacy materials (01_research → research_framework, OSC experiments → 03_parameter_space_mapping_pipeline/data/control_data).
- Define metadata schemas for ADM parsing and OSC logging.

## Phase 2 · Analytical Depth (Q2 FY26)
- Implement ADM parsing prototypes and feature extraction pipelines.
- Automate visualization + reporting agents.
- Validate descriptors on existing mixes (see `out_tp_mix03`).

## Phase 3 · Mapping Engine (Q3 FY26) ✅ IMPLEMENTED
The full 7-step CRE (Compositional Rule Engine) pipeline is now operational.

### Pipeline architecture (`03_parameter_space_mapping_pipeline/`)

```
MusicXML score
    │
    ▼  (Step 1-2)
src/cre/rule_engine.py
    ├─ extract_chord_events()      → harmonic snapshots via music21 chordify
    └─ extract_dynamic_events()    → per-part Dynamic markings (pp→ff)
    │
    ▼  (Step 3)
    ├─ compute_tension_score()     → mean pairwise interval dissonance [0–1]
    └─ dynamics_to_velocity()      → normalised dynamic level [0–1]
    │
    ▼  (Step 4)  configs/rules/*.json
    ├─ quartet_harmony_rule.json       tension_score → spread (linear)
    └─ quartet_dynamics_position_rule.json  rms_velocity → position_y (linear)
    │
    ▼  (Step 5)
scripts/generate_spatial_timeline.py  →  outputs/spatial_timeline_<ts>.csv
    │
    ▼  (Step 6)
scripts/send_adm_osc.py               →  /adm/obj/<n>/spread  +  /cartesian
scripts/sync_transport.py             →  /transport/play|stop|locate listener
    │
    ▼  (Step 7)
Nuendo: ADM objects move with the music
```

### Key files
| File | Role |
|------|------|
| `src/cre/rule_engine.py` | Full pipeline engine; can run standalone with `--offline` or `--realtime` |
| `scripts/generate_spatial_timeline.py` | Step 5 — offline timeline pre-calculation |
| `scripts/send_adm_osc.py` | Step 6 — OSC dispatch from CSV |
| `scripts/sync_transport.py` | Step 6 — DAW transport sync (listen for /transport/play) |
| `configs/rules/quartet_harmony_rule.json` | CRE rule: tension → spread |
| `configs/rules/quartet_dynamics_position_rule.json` | CRE rule: dynamics → position |
| `configs/osc_config.json` | OSC host/port settings (send: 8000, listen: 9001) |

### Quick start
```bash
# 1. Generate the spatial timeline (offline, no Nuendo needed)
python scripts/generate_spatial_timeline.py \
    --score data/scores/quartet.musicxml \
    --rules-dir configs/rules/ \
    --output outputs/my_timeline.csv

# 2. Dry-run: inspect OSC messages without sending
python scripts/send_adm_osc.py \
    --timeline outputs/my_timeline.csv \
    --dry-run

# 3. Live: press PLAY in Nuendo, Python follows
python scripts/send_adm_osc.py \
    --timeline outputs/my_timeline.csv \
    --wait-for-nuendo --listen-port 9001

# 4. All-in-one (score → OSC in one command)
python src/cre/rule_engine.py \
    --score data/scores/quartet.musicxml \
    --rules-dir configs/rules/ \
    --realtime
```

### Nuendo setup (OSC)
- **Nuendo → Python**: Enable "OSC Output" in Nuendo, target `127.0.0.1:9001`
- **Python → Nuendo**: Nuendo listens on `127.0.0.1:8000` for ADM-OSC messages

## Phase 4 · Integration & Evaluation (Q4 FY26)
- Run closed-loop tests: analysis → mapping → rendering → analysis.
- Extend rule set: dynamics → elevation, density → rotation, spectral centroid → distance.
- Publish academic papers + thesis chapters referencing shared_resources bibliographies.
- Prepare reproducible packages for examiners and collaborators.
