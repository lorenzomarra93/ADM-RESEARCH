# Feature Extraction Agent

## Purpose
Compute spatial/timbral descriptors aligned with musical timelines.

## Inputs
- Parsed ADM datasets
- Analysis configuration profiles

## Procedure
1. Execute `scripts/02_object_adm_analysis/extract_features.py` per dataset.
2. Validate alignment with musical bar markers.
3. Export features to `outputs/csv/`.

## Handoffs
Feature tables → `correlation_agent`, `visualization_agent`.
