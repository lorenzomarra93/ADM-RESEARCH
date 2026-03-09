# Metadata Agent

## Purpose
Guarantee that every ADM/BW64 asset includes validated metadata before entering the pipeline.

## Inputs
- Raw ADM/BW64 files
- ITU schema references

## Procedure
1. Validate XML structure.
2. Normalize channel/object naming conventions.
3. Store summary JSON in `data/reference/`.

## Handoffs
Parsed assets → `adm_parser_agent`.

## TODO
Automate schema validation with CI.
