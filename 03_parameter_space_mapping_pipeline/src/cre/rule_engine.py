"""Compositional Rule Engine MVP for the string quartet case study.

This module ingests a MusicXML score, extracts harmonic tension and dynamics,
maps descriptors to ADM-OSC parameters, and optionally streams them to Nuendo
via python-osc.

Pipeline steps (mirrors the PipelineLesson UI):
    1. MusicXML score              → converter.parse()
    2. music21 parser              → extract_chord_events() + extract_dynamic_events()
    3. Feature extraction          → compute_tension_score() + dynamics_to_velocity()
    4. Rule Engine                 → apply_mapping_to_events()  [multi-rule]
    5. Spatial timeline pre-calc   → build_spatial_timeline()
    6. OSC Scheduler               → dispatch_spread_messages() / OscScheduler
    7. Nuendo ADM objects          → receive /adm/obj/<n>/spread, /cartesian, etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from music21 import chord as m21_chord
from music21 import converter, dynamics as m21_dynamics, interval, note as m21_note, stream
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

# Interval dissonance weights (semitone class → tension 0.0–1.0)
INTERVAL_WEIGHTS: Dict[int, float] = {
    0: 0.0,   # unison
    1: 1.0,   # minor second
    2: 0.7,   # major second
    3: 0.5,   # minor third
    4: 0.4,   # major third
    5: 0.25,  # perfect fourth
    6: 1.0,   # tritone
    7: 0.2,   # perfect fifth
    8: 0.4,   # minor sixth
    9: 0.5,   # major sixth
    10: 0.7,  # minor seventh
    11: 1.0,  # major seventh
}

# Dynamics symbol → normalised velocity [0.0 – 1.0]
# Reference: ppp=0.0 … fff=1.0  (7 levels, uniform spacing)
DYNAMICS_VELOCITY: Dict[str, float] = {
    "pppp": 0.0,
    "ppp":  0.083,
    "pp":   0.25,
    "p":    0.375,
    "mp":   0.5,
    "mf":   0.625,
    "f":    0.75,
    "ff":   0.875,
    "fff":  1.0,
    "ffff": 1.0,
    "sfz":  0.9,
    "sfp":  0.8,
    "fz":   0.85,
}

# Instrument → ADM object index (1-based, as used in /adm/obj/<n>/…)
# Verificato dal Renderer Dolby Atmos: oggetti 1, 2, 3, 4 (prima riga Audio Objects)
OBJECT_INDEX: Dict[str, int] = {
    "violin_1": 1,
    "violin_2": 2,
    "viola":    3,
    "cello":    4,
}

# ──────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────

@dataclass
class TempoSpan:
    """Piecewise-linear conversion between quarter-length offsets and seconds."""
    start_offset: float
    end_offset: Optional[float]
    seconds_per_quarter: float
    cumulative_seconds_at_start: float


@dataclass
class ChordEvent:
    """Container describing a chord occurrence extracted from the score."""
    measure: int
    beat: float
    timestamp_s: float
    pitches: List[str]


@dataclass
class DynamicEvent:
    """A dynamic marking extracted from a single part."""
    measure: int
    beat: float
    timestamp_s: float
    part_name: str          # e.g. "Violin I"
    object_id: str          # e.g. "violin_1"
    dynamic_symbol: str     # e.g. "pp", "ff"
    velocity: float         # normalised 0.0–1.0


@dataclass
class SpatialState:
    """One row of the pre-calculated spatial timeline."""
    timestamp_s: float
    measure: int
    beat: float
    object_id: str
    spread: float           # /adm/obj/<n>/spread  0.0–1.0
    x: float                # /adm/obj/<n>/cartesian  left-right  -1..+1
    y: float                # /adm/obj/<n>/cartesian  front-back  -1..+1
    z: float                # /adm/obj/<n>/cartesian  height      -1..+1
    active_rule_id: str


@dataclass
class MappingSpec:
    """Specification of a CRE mapping function decoded from JSON."""
    mapping_type: str
    params: Dict[str, float]


@dataclass
class Rule:
    """CRE rule descriptor."""
    id: str
    trigger_parameter: str          # "tension_score" | "rms_velocity"
    spatial_target: str             # "spread" | "position_y" | "position_x"
    mapping: MappingSpec
    objects: List[str]
    temporal_scope: str             # "micro" | "meso" | "macro"
    smoothing_ms: int


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRE: MusicXML score → spatial timeline → ADM-OSC",
    )
    parser.add_argument("--score", required=True, help="Path to MusicXML file.")
    parser.add_argument("--rule-config", default=None, help="Path to JSON rule (single rule mode).")
    parser.add_argument("--rules-dir", default=None, help="Directory with multiple rule JSONs (multi-rule mode).")
    parser.add_argument("--osc-config", default=None, help="Path to osc_config.json.")
    parser.add_argument("--log-dir", default=None, help="Override OSC log directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print OSC messages without sending.")
    parser.add_argument("--offline", action="store_true",
                        help="Generate and save timeline CSV only; do NOT open an OSC socket.")
    parser.add_argument("--realtime", action="store_true",
                        help="Run the OSC scheduler locked to wall-clock time (requires Nuendo sync).")
    return parser.parse_args()


# ──────────────────────────────────────────────
# JSON HELPERS
# ──────────────────────────────────────────────

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_mapping_spec(spec_str: str) -> MappingSpec:
    """Decode 'linear(0.0→1.0)' or 'exponential(…)' notation."""
    normalized = spec_str.replace("→", "->").strip()

    linear_pattern = re.compile(
        r"linear\(\s*([-\d\.]+)\s*->\s*([-\d\.]+)\s*:\s*([-\d\.]+)\s*->\s*([-\d\.]+)\s*\)"
    )
    match = linear_pattern.match(normalized)
    if match:
        in_min, in_max, out_min, out_max = map(float, match.groups())
        return MappingSpec(
            mapping_type="linear",
            params={"in_min": in_min, "in_max": in_max, "out_min": out_min, "out_max": out_max},
        )

    exp_pattern = re.compile(
        r"exponential\(\s*([-\d\.]+)\s*->\s*([-\d\.]+)\s*:\s*([-\d\.]+)\s*->\s*([-\d\.]+)\s*,\s*exp=([-\d\.]+)\s*\)"
    )
    match = exp_pattern.match(normalized)
    if match:
        in_min, in_max, out_min, out_max, exp = map(float, match.groups())
        return MappingSpec(
            mapping_type="exponential",
            params={"in_min": in_min, "in_max": in_max, "out_min": out_min, "out_max": out_max, "exp": exp},
        )

    raise ValueError(f"Unsupported mapping function: {spec_str!r}")


def load_rule(rule_path: Path) -> Rule:
    data = load_json(rule_path)
    mapping = parse_mapping_spec(data["mapping_function"])
    return Rule(
        id=data["id"],
        trigger_parameter=data["trigger_parameter"],
        spatial_target=data["spatial_target"],
        mapping=mapping,
        objects=data["objects"],
        temporal_scope=data.get("temporal_scope", "meso"),
        smoothing_ms=int(data.get("smoothing_ms", 0)),
    )


def load_all_rules(rules_dir: Path) -> List[Rule]:
    rules = [load_rule(p) for p in sorted(rules_dir.glob("*.json"))]
    LOGGER.info("Loaded %d rules from %s", len(rules), rules_dir)
    return rules


# ──────────────────────────────────────────────
# TEMPO
# ──────────────────────────────────────────────

def build_tempo_spans(score_obj: stream.Score) -> List[TempoSpan]:
    boundaries = score_obj.metronomeMarkBoundaries()
    spans: List[TempoSpan] = []
    cumulative = 0.0
    if not boundaries:
        spans.append(TempoSpan(0.0, None, 1.0, 0.0))
        return spans
    for start, end, mark in boundaries:
        bpm = mark.number if mark and mark.number else 60.0
        spq = 60.0 / bpm
        spans.append(TempoSpan(
            start_offset=float(start),
            end_offset=float(end) if end is not None else None,
            seconds_per_quarter=spq,
            cumulative_seconds_at_start=cumulative,
        ))
        if end is not None:
            cumulative += (float(end) - float(start)) * spq
    return spans


def offset_to_seconds(offset: float, spans: Sequence[TempoSpan]) -> float:
    for span in spans:
        if span.end_offset is None or offset < span.end_offset + 1e-9:
            delta = offset - span.start_offset
            return span.cumulative_seconds_at_start + delta * span.seconds_per_quarter
    last = spans[-1]
    return last.cumulative_seconds_at_start + (offset - last.start_offset) * last.seconds_per_quarter


# ──────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────

def extract_chord_events(score_obj: stream.Score) -> List[ChordEvent]:
    """Step 2a – harmonic snapshot per chord using chordify."""
    spans = build_tempo_spans(score_obj)
    chords_stream = score_obj.chordify()
    events: List[ChordEvent] = []
    for elem in chords_stream.recurse().getElementsByClass(m21_chord.Chord):
        if not elem.pitches:
            continue
        events.append(ChordEvent(
            measure=int(elem.measureNumber or 0),
            beat=float(elem.beat or 0.0),
            timestamp_s=offset_to_seconds(float(elem.offset), spans),
            pitches=[p.nameWithOctave for p in elem.pitches],
        ))
    LOGGER.info("Extracted %d chord events", len(events))
    return events


def extract_dynamic_events(score_obj: stream.Score) -> List[DynamicEvent]:
    """Step 2b – extract Dynamic markings from each part.

    For each part we walk all Dynamic elements and carry the last seen
    dynamic forward (last-known-value model) so that every chord event
    has an associated dynamic level.
    """
    spans = build_tempo_spans(score_obj)

    # Map part names to canonical object IDs
    part_to_object = _build_part_to_object_map(score_obj)

    events: List[DynamicEvent] = []
    for part in score_obj.parts:
        obj_id = part_to_object.get(part.partName or "", None)
        if obj_id is None:
            # Try to infer from part abbreviation or index
            obj_id = part_to_object.get(part.partAbbreviation or "", f"part_{part.id}")

        for elem in part.recurse().getElementsByClass(m21_dynamics.Dynamic):
            sym = elem.value or elem.volumeScalar  # e.g. "pp", "ff"
            if sym is None:
                continue
            sym = str(sym).strip().lower()
            vel = DYNAMICS_VELOCITY.get(sym, 0.5)
            ts = offset_to_seconds(float(elem.offset), spans)
            events.append(DynamicEvent(
                measure=int(elem.measureNumber or 0),
                beat=float(elem.beat or 0.0),
                timestamp_s=ts,
                part_name=part.partName or "",
                object_id=obj_id,
                dynamic_symbol=sym,
                velocity=vel,
            ))

    LOGGER.info("Extracted %d dynamic events across all parts", len(events))
    return events


def _build_part_to_object_map(score_obj: stream.Score) -> Dict[str, str]:
    """Heuristically match part names to canonical object IDs."""
    mapping: Dict[str, str] = {}
    keywords = [
        (["violin i", "violino i", "vl. i", "vl.i", "violin 1"], "violin_1"),
        (["violin ii", "violino ii", "vl. ii", "vl.ii", "violin 2"], "violin_2"),
        (["viola", "vla", "va"], "viola"),
        (["cello", "violoncello", "vc", "vlc"], "cello"),
    ]
    for part in score_obj.parts:
        name_lower = (part.partName or "").lower()
        for keys, obj_id in keywords:
            if any(k in name_lower for k in keys):
                mapping[part.partName or ""] = obj_id
                break
        else:
            mapping[part.partName or ""] = f"part_{part.id}"
    return mapping


def compute_tension_score(event: ChordEvent) -> float:
    """Step 3a – mean pairwise interval dissonance for a chord.

    event.pitches contains note name strings (e.g. 'D5', 'A4').
    We convert them to music21 Pitch objects before computing intervals.
    """
    from music21 import pitch as m21_pitch  # local import to keep top-level clean
    if len(event.pitches) <= 1:
        return 0.0
    pitch_objs = [m21_pitch.Pitch(p) for p in event.pitches]
    dissonances: List[float] = []
    for i, p in enumerate(pitch_objs):
        for q in pitch_objs[i + 1:]:
            semitones = abs(interval.Interval(p, q).semitones) % 12
            dissonances.append(INTERVAL_WEIGHTS.get(semitones, 0.7))
    return float(np.clip(np.mean(dissonances), 0.0, 1.0))


# ──────────────────────────────────────────────
# MAPPING FUNCTIONS
# ──────────────────────────────────────────────

def apply_mapping(value: float, spec: MappingSpec) -> float:
    p = spec.params
    norm = np.clip((value - p["in_min"]) / max(p["in_max"] - p["in_min"], 1e-9), 0.0, 1.0)
    if spec.mapping_type == "exponential":
        norm = norm ** p.get("exp", 2.0)
    return float(p["out_min"] + norm * (p["out_max"] - p["out_min"]))


def moving_average(values: Sequence[float], timestamps: Sequence[float], window_s: float) -> List[float]:
    smoothed: List[float] = []
    for idx, ts in enumerate(timestamps):
        start = ts - window_s
        acc, cnt = 0.0, 0
        for j in range(idx, -1, -1):
            if timestamps[j] < start:
                break
            acc += values[j]
            cnt += 1
        smoothed.append(acc / cnt if cnt else values[idx])
    return smoothed


def apply_mapping_to_events(
    descriptor_values: Sequence[float],
    timestamps: Sequence[float],
    rule: Rule,
) -> List[float]:
    mapped = [apply_mapping(v, rule.mapping) for v in descriptor_values]
    if rule.smoothing_ms > 0 and len(mapped) > 1:
        mapped = moving_average(mapped, timestamps, rule.smoothing_ms / 1000.0)
    return mapped


# ──────────────────────────────────────────────
# DYNAMICS → POSITION MAPPING
# ──────────────────────────────────────────────
#
# Sistema di coordinate ADM-OSC confermato in Nuendo (Dolby Atmos Renderer):
#
#   X:  -1.0 = sinistra estrema     →  +1.0 = destra estrema
#   Y:  -1.0 = dietro (rear)        →  +1.0 = avanti (front/listener)
#   Z:   0.0 = Main layer (piano ascoltatore / "attorno a me")
#        0.5 = metà altezza tra Main e Top  ("a metà altezza")
#        1.0 = Top layer  (soffitto / overhead / "sopra di me")
#
# Regola compositiva applicata:
#   - X fisso per strumento (posizione sul palco, non cambia)
#   - Y varia con la dinamica: pp → dietro, ff → avanti
#   - Z = 0.0 sempre (Main layer, nessun overhead di default)

# Posizione X fissa per strumento — disposizione quartetto sul palco
_BASE_X: Dict[str, float] = {
    "violin_1": -0.85,   # sinistra estrema
    "violin_2": -0.35,   # centro-sinistra
    "viola":     0.35,   # centro-destra
    "cello":     0.85,   # destra estrema
}


def dynamics_to_cartesian(
    velocity: float,
    object_id: str,
    base_y_soft: float = -0.6,   # Y quando pp  (ben dietro il centro)
    base_y_loud: float = 0.9,    # Y quando ff  (molto avanti)
    z: float = 0.0,              # Z = 0.0 → Main layer (piano ascoltatore)
) -> Tuple[float, float, float]:
    """Map a normalised velocity [0–1] to (x, y, z) ADM-OSC coordinates.

    Coordinate system (confirmed in Nuendo / Dolby Atmos Renderer):
        x: -1.0 (L)    … +1.0 (R)
        y: -1.0 (rear) … +1.0 (front)
        z:  0.0 (Main/listener plane)  …  1.0 (Top/ceiling layer)
            — NOT a continuous physical height axis —
            M = "attorno a me", 0.5 = "a metà altezza", T = "sopra di me"

    Compositional rule:
        - x: fixed per instrument (stage position, from _BASE_X)
        - y: scales with dynamic — pp=rear, ff=front
        - z: fixed at 0.0 (Main layer) unless overridden
    """
    x = _BASE_X.get(object_id, 0.0)
    y = base_y_soft + (base_y_loud - base_y_soft) * velocity
    return float(np.clip(x, -1.0, 1.0)), float(np.clip(y, -1.0, 1.0)), float(np.clip(z, 0.0, 1.0))


# ──────────────────────────────────────────────
# STEP 5 – SPATIAL TIMELINE BUILDER
# ──────────────────────────────────────────────

def build_spatial_timeline(
    chord_events: List[ChordEvent],
    dynamic_events: List[DynamicEvent],
    rules: List[Rule],
) -> pd.DataFrame:
    """Pre-calculate the complete spatial timeline — the "partitura spaziale".

    For every chord event and every instrument object this function:
      1. Applies all CRE rules whose trigger_parameter == "tension_score"
         to compute spread.
      2. Looks up the last active dynamic for that instrument at that timestamp
         and maps it to cartesian position via dynamics_to_cartesian().
      3. Assembles a single row per (event, object).

    Returns:
        DataFrame with columns:
            timestamp_s, measure, beat, object_id, object_index,
            spread, x, y, z, tension_score, rms_velocity,
            dynamic_symbol, active_rule_id
    """
    # ── Tension trajectory (same for all objects, comes from chordify) ──
    tension_scores = [compute_tension_score(e) for e in chord_events]
    timestamps = [e.timestamp_s for e in chord_events]

    # Collect spread rules
    spread_rules = [r for r in rules if r.spatial_target == "spread"]
    if not spread_rules:
        LOGGER.warning("No spread rules found; spread will be 0 for all events.")
    primary_spread_rule = spread_rules[0] if spread_rules else None

    spread_values: List[float] = []
    active_rule_id = "none"
    if primary_spread_rule:
        spread_values = apply_mapping_to_events(tension_scores, timestamps, primary_spread_rule)
        active_rule_id = primary_spread_rule.id
    else:
        spread_values = [0.0] * len(chord_events)

    # ── Build per-instrument dynamic lookup (last-known-value) ──
    # Group dynamic events per object_id, sorted by time
    dyn_by_object: Dict[str, List[DynamicEvent]] = {}
    for de in sorted(dynamic_events, key=lambda d: d.timestamp_s):
        dyn_by_object.setdefault(de.object_id, []).append(de)

    all_objects = list(OBJECT_INDEX.keys())

    rows: List[Dict] = []
    for ev, tension, spread in zip(chord_events, tension_scores, spread_values):
        for obj_id in all_objects:
            # Find the last dynamic event for this object at or before ev.timestamp_s
            dyn_evs = dyn_by_object.get(obj_id, [])
            active_dyn = _last_before(dyn_evs, ev.timestamp_s)
            velocity = active_dyn.velocity if active_dyn else 0.5
            dyn_sym = active_dyn.dynamic_symbol if active_dyn else "mp"

            x, y, z = dynamics_to_cartesian(velocity, obj_id)

            rows.append({
                "timestamp_s":    ev.timestamp_s,
                "measure":        ev.measure,
                "beat":           ev.beat,
                "object_id":      obj_id,
                "object_index":   OBJECT_INDEX.get(obj_id, 0),
                "spread":         round(spread, 4),
                "x":              round(x, 4),
                "y":              round(y, 4),
                "z":              round(z, 4),
                "tension_score":  round(tension, 4),
                "rms_velocity":   round(velocity, 4),
                "dynamic_symbol": dyn_sym,
                "active_rule_id": active_rule_id,
            })

    df = pd.DataFrame(rows)
    df.sort_values(["timestamp_s", "object_index"], inplace=True, ignore_index=True)
    LOGGER.info("Spatial timeline built: %d rows", len(df))
    return df


def _last_before(events: List[DynamicEvent], timestamp: float) -> Optional[DynamicEvent]:
    """Return the latest event whose timestamp ≤ `timestamp`, or None."""
    result = None
    for ev in events:
        if ev.timestamp_s <= timestamp + 1e-6:
            result = ev
        else:
            break
    return result


# ──────────────────────────────────────────────
# STEP 6 – OSC SCHEDULER
# ──────────────────────────────────────────────

class OscScheduler:
    """Real-time OSC scheduler locked to wall-clock time.

    Usage:
        scheduler = OscScheduler(timeline_df, osc_client, dry_run=False)
        scheduler.start(t0_wall=time.perf_counter())
        # …Nuendo plays…
        scheduler.join()
    """

    def __init__(
        self,
        timeline: pd.DataFrame,
        client: Optional[SimpleUDPClient],
        dry_run: bool = False,
        listen_host: str = "127.0.0.1",
        listen_port: int = 9001,
    ) -> None:
        self.timeline = timeline.sort_values("timestamp_s").reset_index(drop=True)
        self.client = client
        self.dry_run = dry_run
        self.listen_host = listen_host
        self.listen_port = listen_port
        self._t0_wall: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Transport listener (receives /transport/play from Nuendo) ──────────

    def _start_osc_listener(self) -> None:
        dispatcher = Dispatcher()
        dispatcher.map("/transport/play", self._on_transport_play)
        dispatcher.map("/transport/stop", self._on_transport_stop)
        try:
            server = BlockingOSCUDPServer((self.listen_host, self.listen_port), dispatcher)
            LOGGER.info("Listening for Nuendo transport on %s:%d", self.listen_host, self.listen_port)
            server.handle_request()   # wait for one START message, then proceed
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OSC listener failed (%s); starting immediately.", exc)
            self._on_transport_play("/transport/play", 1)

    def _on_transport_play(self, address: str, *args) -> None:  # noqa: ANN002
        LOGGER.info("Received %s — starting OSC scheduler at t=0", address)
        self._t0_wall = time.perf_counter()
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()

    def _on_transport_stop(self, address: str, *args) -> None:  # noqa: ANN002
        LOGGER.info("Received %s — stopping OSC scheduler", address)
        self._stop_event.set()

    # ── Scheduler loop ─────────────────────────────────────────────────────

    def start(self, t0_wall: Optional[float] = None) -> None:
        """Start immediately (offline mode) or wait for Nuendo START."""
        if t0_wall is not None:
            self._t0_wall = t0_wall
            self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self._thread.start()
        else:
            # Block until /transport/play arrives
            self._start_osc_listener()

    def join(self, timeout: float = 600.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run_scheduler(self) -> None:
        assert self._t0_wall is not None
        LOGGER.info("Scheduler running — %d OSC events to dispatch", len(self.timeline))
        for _, row in self.timeline.iterrows():
            if self._stop_event.is_set():
                break
            target_time = self._t0_wall + row["timestamp_s"]
            now = time.perf_counter()
            sleep_s = target_time - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._dispatch_row(row)
        LOGGER.info("Scheduler finished.")

    def _dispatch_row(self, row: pd.Series) -> None:
        idx = int(row["object_index"])
        # ADM-OSC range: x -1.0 (L) → +1.0 (R), y -1.0 (back) → +1.0 (front), z 0..1
        # Timeline CSV usa x in [0,1] — convertiamo a [-1, +1] con clamp di sicurezza
        x_norm = max(0.0, min(1.0, float(row["x"])))   # clamp [0,1]
        y_norm = max(0.0, min(1.0, float(row["y"])))   # clamp [0,1]
        z_norm = max(0.0, min(1.0, float(row["z"])))   # clamp [0,1]
        x_adm = x_norm * 2.0 - 1.0                    # → -1..+1
        y_adm = y_norm * 2.0 - 1.0                    # → -1..+1
        z_adm = z_norm                                 # Nuendo accetta 0..1 per z

        # ✅ Formato confermato in Nuendo (ADM-OSC EBU, fluido L↔R su asse Y centrale):
        #   Address : /adm/obj/{n}/xyz
        #   Args    : [x (float -1..+1), y (float -1..+1), z (float 0..1)]
        #   Port    : 8000
        addr = f"/adm/obj/{idx}/xyz"
        args = [x_adm, y_adm, z_adm]

        if self.dry_run or self.client is None:
            LOGGER.info("DRY  %s -> [%.3f, %.3f, %.3f]", addr, *args)
        else:
            self.client.send_message(addr, args)


# ──────────────────────────────────────────────
# LEGACY SINGLE-RULE DISPATCH (backward compat)
# ──────────────────────────────────────────────

def dispatch_spread_messages(
    spread_values: Sequence[float],
    events: Sequence[ChordEvent],
    rule: Rule,
    osc_client: Optional[SimpleUDPClient],
    log_file: Path,
    dry_run: bool,
) -> pd.DataFrame:
    """Legacy single-rule OSC dispatch (kept for backward compatibility)."""
    records: List[Dict] = []
    log_lines: List[str] = []
    for event, spread in zip(events, spread_values):
        ts_ms = int(event.timestamp_s * 1000)
        for index, obj in enumerate(rule.objects, start=1):
            address = f"/adm/obj/{index}/spread"
            payload = float(spread)
            if dry_run:
                LOGGER.info("DRY RUN %s -> %.3f (%s)", address, payload, obj)
            else:
                assert osc_client is not None
                osc_client.send_message(address, payload)
            log_lines.append(f"{ts_ms} | {address} | {payload:.4f} | rule={rule.id} | obj={obj}")
            records.append({
                "timestamp_s": event.timestamp_s,
                "measure": event.measure,
                "beat": event.beat,
                "object_id": obj,
                "spread": payload,
                "active_rule_id": rule.id,
            })
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    return pd.DataFrame.from_records(records)


def create_osc_client(host: str, port: int, dry_run: bool) -> Optional[SimpleUDPClient]:
    if dry_run:
        return None
    return SimpleUDPClient(host, port)


# ──────────────────────────────────────────────
# PATH RESOLUTION
# ──────────────────────────────────────────────

def resolve_default_paths(args: argparse.Namespace, pipeline_root: Path):
    score_path = Path(args.score).expanduser()
    if not score_path.exists():
        raise FileNotFoundError(f"Score not found: {score_path}")
    default_rule = pipeline_root / "configs" / "rules" / "quartet_harmony_rule.json"
    default_osc = pipeline_root / "configs" / "osc_config.json"
    rule_path = Path(args.rule_config).expanduser() if args.rule_config else default_rule
    osc_config_path = Path(args.osc_config).expanduser() if args.osc_config else default_osc
    return score_path, rule_path, osc_config_path


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    """Entry point: score → features → timeline → OSC."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    pipeline_root = project_root / "03_parameter_space_mapping_pipeline"
    score_path, rule_path, osc_config_path = resolve_default_paths(args, pipeline_root)

    # ── Step 1+2: Load score and extract features ──────────────────────────
    LOGGER.info("Step 1/2 — Loading score: %s", score_path)
    score_obj = converter.parse(score_path)
    chord_events = extract_chord_events(score_obj)
    dynamic_events = extract_dynamic_events(score_obj)

    if not chord_events:
        raise RuntimeError("No chords found in score.")

    # ── Step 3: Load rules ─────────────────────────────────────────────────
    if args.rules_dir:
        rules = load_all_rules(Path(args.rules_dir).expanduser())
    else:
        rules = [load_rule(rule_path)]

    # ── Step 4+5: Build spatial timeline ──────────────────────────────────
    LOGGER.info("Step 4/5 — Building spatial timeline…")
    timeline_df = build_spatial_timeline(chord_events, dynamic_events, rules)

    osc_config = load_json(osc_config_path)
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else project_root / osc_config["log_directory"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ts_label = datetime.now(timezone.utc).isoformat().replace(":", "-")

    # Save timeline CSV (the "partitura spaziale")
    csv_path = log_dir / f"spatial_timeline_{ts_label}.csv"
    timeline_df.to_csv(csv_path, index=False)
    LOGGER.info("Spatial timeline saved → %s", csv_path)

    if args.offline:
        LOGGER.info("Offline mode: timeline written, no OSC sent.")
        return

    # ── Step 6: OSC Scheduler ─────────────────────────────────────────────
    LOGGER.info("Step 6 — Starting OSC Scheduler…")
    target = osc_config.get("default_target", {})
    host = target.get("host", "127.0.0.1")
    port = int(target.get("port", 8000))
    listen_port = int(osc_config.get("listen_port", 9001))
    osc_client = create_osc_client(host, port, args.dry_run)

    scheduler = OscScheduler(
        timeline=timeline_df,
        client=osc_client,
        dry_run=args.dry_run,
        listen_host="0.0.0.0",
        listen_port=listen_port,
    )

    if args.realtime:
        LOGGER.info("Realtime mode — waiting for /transport/play from Nuendo on port %d…", listen_port)
        scheduler.start(t0_wall=None)       # blocks until START received
    else:
        LOGGER.info("Immediate playback (no sync wait)")
        scheduler.start(t0_wall=time.perf_counter())

    scheduler.join()
    LOGGER.info("Session complete.")


if __name__ == "__main__":
    main()
