# ADM Parser Agent

## Purpose
Convert validated ADM/BW64 files into structured datasets for feature extraction.

## Inputs
- Metadata summaries from `metadata_agent`
- Raw audio paths

## Procedure
1. Run `scripts/02_object_adm_analysis/parse_adm.py` with the desired config profile.
2. Save object trajectories to `data/intermediate/`.
3. Log parsing stats.

## Handoffs
Intermediate datasets → `feature_extraction_agent`.
