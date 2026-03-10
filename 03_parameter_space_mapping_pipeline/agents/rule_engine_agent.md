# Rule Engine Agent

## Purpose
Load Compositional Rule Engine (CRE) JSON rule sets, evaluate descriptors over time, and emit spatial states ready for OSC dispatch.

## Triggers
- New descriptor batches arrive from the Score Reader Agent or live analysis streams.
- A rule set under `configs/rules/` is added or modified.

## Inputs
- `descriptors_df` (timestamped pandas DataFrame with normalized musical descriptors).
- Rule set JSON defining `trigger_parameter`, `spatial_target`, mapping functions, and scope priorities.
- Optional conflict-resolution policies (default: macro > meso > micro).

## Outputs
- `spatial_states_df`: DataFrame keyed by timestamp and object ID containing `azimuth`, `elevation`, `distance`, `spread`, `gain`, `trajectory_velocity`, plus `active_rule_id`.
- Diagnostic logs capturing rule activations and overrides.

## Procedure
1. Validate descriptor columns against the rule set’s `trigger_parameter` requirements; abort if missing fields.
2. Load rules, sort them by scope priority (macro > meso > micro), and instantiate mapping functions (linear, sigmoid, exponential, lookup-table).
3. Iterate over each timestep, evaluate all eligible rules, and resolve conflicts by retaining the highest-priority result per `spatial_target`.
4. Compile object-wise spatial state dictionaries, append `active_rule_id`, and accumulate into `spatial_states_df`.
5. Persist the DataFrame to `data/processed/spatial_states/` and signal completion to the OSC Sender Agent.

## Handoffs
`spatial_states_df` → OSC Sender Agent.
