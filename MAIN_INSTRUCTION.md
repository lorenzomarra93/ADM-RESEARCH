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

## Research Questions Compass
Refer to `RESEARCH_QUESTIONS.md` for the living list of core questions. Every task inside both macro-areas must link to at least one question to justify scope.

## Methodological Principles
- Traceability: each dataset, script, and decision must cite its origin and transformation steps.
- Modularity: agents isolate responsibilities (parsing, visualization, mapping, etc.) to ease delegation and automation.
- Micro/Meso/Macro Awareness: decisions must specify the structural scale under study.
- Cross-Domain Alignment: theoretical claims require analytical evidence; creative claims require implementation plans.
- Documentation First: READMEs and instruction files are authoritative sources for collaborators and future agents.

## Micro / Meso / Macro Lenses
| Scale | Description | Typical Evidence |
| --- | --- | --- |
| Micro | Instantaneous gestures, timbral inflections, controller events | Frame-level ADM metadata, OSC traces |
| Meso | Motives, sections, sound objects evolving over seconds | Feature trajectories, mapping matrices |
| Macro | Global form, dramaturgy, spatial dramaturgic arcs | Timelines, concert/film scene plans |

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
