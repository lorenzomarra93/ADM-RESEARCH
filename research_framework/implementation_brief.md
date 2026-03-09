# Implementation Brief

- Target Languages: Python for analysis + mapping utilities; Lua/Max/MSP or Nuendo macros documented separately if needed.
- Data Contracts: prefer Parquet/CSV for structured data, JSON for metadata snapshots, and Markdown for qualitative notes.
- Versioning: track raw data immutably; transformations happen in `intermediate/` with scripts referenced in commit messages.
- Testing: unit tests live in each macro-area `tests/` folder; integration notebooks document exploratory runs.
- Open Questions:
  - How to standardize ADM parser outputs for both Dolby and BW64 sources?
  - Which OSC schema best supports both Nuendo and SPAT workflows?
