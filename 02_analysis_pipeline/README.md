# Analysis Pipeline

## Scope
Provides reproducible workflows for parsing, annotating, and reporting on ADM/Dolby Atmos/BW64 materials.

## Inputs
- Multichannel audio with ADM metadata
- Object/beds descriptors from partners
- Control logs for ground-truth comparison

## Outputs
- Normalized feature tables in `outputs/csv|json`
- Diagnostic visualizations in `outputs/plots`
- Reports for academic documentation in `outputs/reports`

## Relation to Project
Feeds `03_parameter_space_mapping_pipeline` with evidence-backed descriptors and receives creative artifacts for verification.
