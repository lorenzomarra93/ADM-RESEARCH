"""generate_spatial_timeline.py — Step 5 of the CRE pipeline.

Pre-calculates the full "partitura spaziale" from a MusicXML score and
saves it as a CSV file (one row per chord event per ADM object).

Usage examples:
    # Single rule (default: quartet_harmony_rule.json)
    python scripts/generate_spatial_timeline.py --score path/to/score.musicxml

    # Multiple rules from a directory
    python scripts/generate_spatial_timeline.py \\
        --score path/to/score.musicxml \\
        --rules-dir configs/rules/ \\
        --output outputs/my_timeline.csv

This script is deliberately "offline-first": no OSC sockets are opened.
Inspect the CSV, tweak the rules, then run send_adm_osc.py or the full
rule_engine.py --realtime to dispatch the timeline to Nuendo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the scripts/ directory or the pipeline root
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_ROOT / "src"))

from cre.rule_engine import (
    build_spatial_timeline,
    extract_chord_events,
    extract_dynamic_events,
    load_all_rules,
    load_rule,
)
from music21 import converter

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_DEFAULT_RULE = _PIPELINE_ROOT / "configs" / "rules" / "quartet_harmony_rule.json"
_DEFAULT_OUTPUT_DIR = _PIPELINE_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRE Step 5 — Generate spatial timeline CSV from a MusicXML score.",
    )
    parser.add_argument("--score", required=True, help="Path to MusicXML file.")
    parser.add_argument(
        "--rule-config",
        default=None,
        help=f"Path to a single rule JSON (default: {_DEFAULT_RULE}).",
    )
    parser.add_argument(
        "--rules-dir",
        default=None,
        help="Directory with multiple rule JSONs (overrides --rule-config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output CSV. Defaults to outputs/spatial_timeline_<timestamp>.csv",
    )
    return parser.parse_args()


def render_timeline(
    score_path: Path,
    rules_dir: Path | None = None,
    rule_config: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Core function: score → spatial timeline CSV.

    Args:
        score_path:   Path to the MusicXML file.
        rules_dir:    Directory containing rule JSON files (multi-rule mode).
        rule_config:  Single rule JSON path (single-rule mode).
        output_path:  Where to save the CSV. Auto-generated if None.

    Returns:
        Path to the written CSV file.
    """
    if not score_path.exists():
        raise FileNotFoundError(f"Score not found: {score_path}")

    # ── Step 1+2: Load and parse score ────────────────────────────────────
    LOGGER.info("Loading score: %s", score_path)
    score_obj = converter.parse(score_path)
    chord_events = extract_chord_events(score_obj)
    dynamic_events = extract_dynamic_events(score_obj)

    if not chord_events:
        raise RuntimeError("No chord events found in score — is this a MusicXML file?")

    LOGGER.info(
        "Parsed %d chord events and %d dynamic events.",
        len(chord_events),
        len(dynamic_events),
    )

    # ── Step 3: Load rules ────────────────────────────────────────────────
    if rules_dir is not None:
        rules = load_all_rules(rules_dir)
    elif rule_config is not None:
        rules = [load_rule(rule_config)]
    else:
        rules = [load_rule(_DEFAULT_RULE)]

    # ── Step 4+5: Build spatial timeline ─────────────────────────────────
    LOGGER.info("Building spatial timeline…")
    timeline_df = build_spatial_timeline(chord_events, dynamic_events, rules)

    # ── Write CSV ─────────────────────────────────────────────────────────
    if output_path is None:
        _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_path = _DEFAULT_OUTPUT_DIR / f"spatial_timeline_{ts_label}.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    timeline_df.to_csv(output_path, index=False)
    LOGGER.info("✅  Spatial timeline saved → %s  (%d rows)", output_path, len(timeline_df))
    return output_path


def main() -> None:
    args = parse_args()
    score_path = Path(args.score).expanduser().resolve()
    rules_dir = Path(args.rules_dir).expanduser().resolve() if args.rules_dir else None
    rule_config = Path(args.rule_config).expanduser().resolve() if args.rule_config else None
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    render_timeline(
        score_path=score_path,
        rules_dir=rules_dir,
        rule_config=rule_config,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
