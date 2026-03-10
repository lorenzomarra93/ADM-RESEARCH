# Mock Renderer Agent

## Purpose
Emulate a DAW/renderer by receiving ADM-OSC packets, printing them for debugging, and writing a verification log that other agents can analyze without launching Nuendo or SPAT Revolution.

## Triggers
- Integration tests that must run headless or in CI.
- Debug sessions where OSC routing needs inspection before hitting production renderers.

## Inputs
- OSC port/IP to bind (default `127.0.0.1:9000`).
- Expected ADM-OSC address patterns for validation.

## Outputs
- Console feed of received OSC messages with timestamps.
- Verification log stored under `data/osc_logs/mock_runs/session_<ISO8601>.log`.
- Optional summary metrics (message counts, malformed packets).

## Procedure
1. Launch `src/cre/mock_osc_server.py`, binding to the configured port.
2. Listen for incoming OSC packets, validate that address patterns match `/adm/obj/<id>/cartesian|spread`.
3. Print each message with a human-readable label for rapid debugging.
4. Append every packet to the verification log for reproducibility.
5. Stop the server gracefully and notify the OSC Sender or Rule Engine agents that the testbed is ready/complete.

## Handoffs
Verification log path → Spatial Score Agent and Analysis Pipeline.
