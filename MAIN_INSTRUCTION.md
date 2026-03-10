# MAIN INSTRUCTION

## Project Identity
This workspace consolidates Lorenzo Marra's PhD research on immersive audio, musical composition, and spatial perception. It treats space as an expressive, structural, and measurable parameter that links analytical evidence with creative practice.

## Strategic Objectives
1. Study spatial audio as a compositional parameter rather than a post-production effect.
2. Build quantitative methodologies for parsing ADM/Dolby Atmos metadata and spatial descriptors.
3. Develop parameter-space mapping strategies that link form, harmony, gesture, and timbre to spatial behaviors.
4. Bridge score-based workflows with electronic, gesture-based, and controller-driven practices.
5. Keep every workflow reproducible, documented, and scalable for academic review and artistic deployment.
6. Support theoretical inquiry, data-driven analysis, and case studies for film, concert, and studio contexts.

## Dual-Lane Architecture
### Analysis Pipeline
- Focus: ingesting, parsing, annotating, and visualizing multichannel/object-based recordings and metadata.
- Key tasks: ADM/BW64 parsing, feature extraction, correlation studies, reporting.
- Outputs: structured datasets, plots, technical reports, metadata summaries.

### Parameter Space Mapping
- Focus: compositional design, control mappings, OSC/ADM-OSC integration, and validation of expressive intent.
- Key tasks: mapping matrices, spatial timelines, controller policies, evaluation strategies, case-study documentation.
- Outputs: mapping specifications, test renders, OSC timelines, artistic dossiers.

### Feedback Loop
Insights from the analysis pipeline feed mapping heuristics, while creative hypotheses return to the analysis lane for verification and documentation.

## Compositional Rule Engine (CRE)
The CRE forms the third architectural pillar by ingesting normalized descriptors from the Analysis Pipeline and deterministically steering the Parameter Space Mapping Pipeline. It ensures that harmonic tension, dynamics, density, and form cues remain traceable as they become OSC gestures and ADM states.

### Rule Anatomy
- `id`: unique string identifier for provenance tracking across analyses and renders.
- `trigger_parameter`: musical descriptor driving the rule; valid values `tension_score`, `rms_db`, `onset_density`, `spectral_centroid`, `voice_count`, `interval_class`, `form_state`.
- `spatial_target`: ADM/OSC parameter controlled by the rule; valid values `azimuth`, `elevation`, `distance`, `spread`, `gain`, `trajectory_velocity`.
- `mapping_function`: mathematical transformation string in the format `linear(input_min→input_max : output_min→output_max)`, `sigmoid`, `exponential`, or `lookup_table:filename.json`.
- `temporal_scope`: declares evaluation horizon — `micro` (<1 s events), `meso` (phrases, 4–30 s), or `macro` (sections >30 s).

### CRE Outputs
1. Real-time OSC stream directed to the target renderer (Nuendo or SPAT Revolution).
2. Logged ADM-OSC session file for post-hoc analysis and verification.
3. Human-readable spatial score (CSV) mapping absolute timecodes to spatial states and active rule IDs.

### Micro/Meso/Macro Table
| Scale | Description | Typical Evidence | CRE Rule Scope |
| --- | --- | --- | --- |
| Micro | Instantaneous gestures, timbral inflections, controller events | Frame-level ADM metadata, OSC traces | `micro` scope; e.g., onset-density spike widens `obj2` spread for a single accent |
| Meso | Motives, sections, sound objects evolving over seconds | Feature trajectories, mapping matrices | `meso` scope; e.g., rising tension_score over a phrase lifts `obj3` elevation arc |
| Macro | Global form, dramaturgy, spatial dramaturgic arcs | Timelines, concert/film scene plans | `macro` scope; e.g., form_state "coda" drives ensemble distance to collapse to center |

## Research Questions Compass
Refer to `RESEARCH_QUESTIONS.md` for the living list of core questions. Every task inside both macro-areas must link to at least one question to justify scope.

## Methodological Principles
- Traceability: each dataset, script, and decision must cite its origin and transformation steps.
- Modularity: agents isolate responsibilities (parsing, visualization, mapping, etc.) to ease delegation and automation.
- Micro/Meso/Macro Awareness: decisions must specify the structural scale under study.
- Cross-Domain Alignment: theoretical claims require analytical evidence; creative claims require implementation plans.
- Documentation First: READMEs and instruction files are authoritative sources for collaborators and future agents.

## Micro / Meso / Macro Lenses
- Micro: instantaneous gestures, timbral inflections, controller events whose evidence lives in frame-level ADM metadata or OSC traces. Pair these observations with `micro`-scope CRE rules listed in the table above.
- Meso: motives and phrases evolving over seconds, tracked via descriptor trajectories and mapping matrices. Align with `meso` rules that mediate phrase-level gestures.
- Macro: form-level dramaturgy spanning sections, validated via annotated timelines and scene plans. Macro CRE rules should declare long-horizon states (e.g., collapsing the ensemble toward the center for codas).

## Quantitative ⇄ Creative Bridge
- Analytical datasets must export normalized descriptors that mapping scripts can consume directly.
- Parameter_space_mapping scripts should log OSC and ADM-OSC data in formats analysis scripts can re-ingest for verification.
- Reporting agents translate numbers into actionable creative briefs and academic writeups.

## Application Domains
- **Electronic & Gesture-Based Music**: emphasize controller calibration, latency tracking, and dense timbral descriptors.
- **Film Music**: align spatial mapping with narrative beats, dialogue windows, and deliverables for dubbing stages.
- **Immersive Concert**: plan for venue-specific constraints, speaker layouts, and rehearsal-friendly documentation.

## Naming & Documentation Conventions
- Use snake_case for folders and files; reserve Title Case for markdown headings.
- Prefix data folders with their lifecycle stage (`raw`, `intermediate`, `processed`).
- Keep agent files focused (one role per file) and include Procedure + Handoffs sections.
- Store configuration defaults in `configs/` and never hardcode paths inside scripts.
- Place TODO and NEXT STEPS blocks where further specification is required.

## Specialized Agents
Markdown agents under each macro-area encapsulate expert workflows. They:
- Define triggers for activation and data prerequisites.
- Establish handoffs to other agents or scripts.
- Capture verification checklists to maintain academic rigor.

## Immediate Next Steps
1. Populate priority datasets inside `02_analysis_pipeline/data/raw` and document provenance in `data_management.md`.
2. Port existing OSC/control experiments into `03_parameter_space_mapping_pipeline/data/control_data` with metadata sheets.
3. Instantiate reporting routines (plots + writeups) so progress is auditable each week.
4. Build CRE minimum viable prototype: harmonic tension → object spread, single string quartet case study.
