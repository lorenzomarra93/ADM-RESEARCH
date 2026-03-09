#!/usr/bin/env python3
"""
osc_object_control.py

Prototype pipeline that parses a MIDI or MusicXML score, maps musical parameters
to spatial coordinates, and drives Nuendo/Dolby Atmos objects via OSC. Optionally
sends MIDI transport (MMC/start) so Nuendo stays in sync while automation is recorded.

Example:
    python osc_object_control.py \\
        --score BookOfWater.mid --format auto --key D --mode dorian \\
        --mapping quintet_mapping.json --osc-host 127.0.0.1 --osc-port 9021 \\
        --mmc-port "IAC Driver Bus 1" --preroll 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mido
from pythonosc import udp_client

# Optional dependency imported lazily when parsing MusicXML.
MUSIC21_AVAILABLE = False

KEY_TO_PC = {
    "C": 0,
    "G": 7,
    "D": 2,
    "A": 9,
    "E": 4,
    "B": 11,
    "F#": 6,
    "Gb": 6,
    "Db": 1,
    "C#": 1,
    "Ab": 8,
    "Eb": 3,
    "Bb": 10,
    "F": 5,
}

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
NAT_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}

FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
FIFTHS_INDEX = {pc: idx for idx, pc in enumerate(FIFTHS_ORDER)}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class NoteEvent:
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    channel: int
    track: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class SpatialParams:
    radius: float
    angle_rad: float
    x: float
    y: float
    z: float
    spread: float
    gain: float
    tension: float


@dataclass
class OscEvent:
    time_s: float
    address: str
    args: Sequence[float]
    note: Optional[NoteEvent] = None
    label: str = ""


class SpatialRuleEngine:
    """Maps pitch/tension to cylindrical coordinates."""

    def __init__(
        self,
        key: str = "C",
        mode: str = "major",
        pitch_low: int = 36,
        pitch_high: int = 96,
        radius_range: Tuple[float, float] = (0.2, 0.95),
        spread_range: Tuple[float, float] = (0.1, 0.8),
    ) -> None:
        if key.upper() not in KEY_TO_PC:
            raise ValueError(f"Unsupported key '{key}'.")
        self.root_pc = KEY_TO_PC[key.upper()]
        self.mode = mode.lower()
        self.pitch_low = pitch_low
        self.pitch_high = pitch_high
        self.radius_min, self.radius_max = radius_range
        self.spread_min, self.spread_max = spread_range

    def tonal_tension(self, pitch: int) -> float:
        pc = (pitch - self.root_pc) % 12
        if self.mode in {"major", "ionian"}:
            scale = MAJOR_SCALE
        else:
            scale = NAT_MINOR_SCALE
        if pc in {0, 4, 7}:
            return 0.1
        if pc in scale:
            return 0.5
        return 1.0

    def _radius_from_tension(self, tension: float) -> float:
        return clamp(
            self.radius_min + (self.radius_max - self.radius_min) * tension,
            0.0,
            1.0,
        )

    def _angle_from_pitch(self, pitch: int) -> float:
        pc = (pitch - self.root_pc) % 12
        idx = FIFTHS_INDEX.get(pc, 0)
        return (idx / len(FIFTHS_ORDER)) * math.tau

    def _z_from_pitch(self, pitch: int) -> float:
        return clamp(
            ((pitch - self.pitch_low) / (self.pitch_high - self.pitch_low + 1e-6)) * 2.0
            - 1.0,
            -1.0,
            1.0,
        )

    def _spread_from_tension(self, tension: float) -> float:
        return self.spread_min + (self.spread_max - self.spread_min) * tension

    def note_to_spatial(self, note: NoteEvent) -> SpatialParams:
        tension = self.tonal_tension(note.pitch)
        radius = self._radius_from_tension(tension)
        angle = self._angle_from_pitch(note.pitch)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = self._z_from_pitch(note.pitch)
        spread = self._spread_from_tension(tension)
        gain = clamp(note.velocity / 127.0, 0.0, 1.0)
        return SpatialParams(
            radius=radius,
            angle_rad=angle,
            x=x,
            y=y,
            z=z,
            spread=spread,
            gain=gain,
            tension=tension,
        )


def build_tempo_map(mid: mido.MidiFile) -> List[Tuple[int, float, int]]:
    """Returns [(tick, seconds, tempo_microseconds_per_beat), ...]."""
    tempo_map: List[Tuple[int, float, int]] = []
    ticks_per_beat = mid.ticks_per_beat
    current_tempo = 500000  # default 120 bpm
    abs_tick = 0
    abs_time = 0.0
    tempo_map.append((abs_tick, abs_time, current_tempo))
    merged = mido.merge_tracks(mid.tracks)
    for msg in merged:
        abs_tick += msg.time
        abs_time += (msg.time * current_tempo) / (ticks_per_beat * 1_000_000.0)
        if msg.type == "set_tempo":
            current_tempo = msg.tempo
            tempo_map.append((abs_tick, abs_time, current_tempo))
    return tempo_map


def ticks_to_seconds(
    tick: int,
    tempo_map: Sequence[Tuple[int, float, int]],
    ticks_per_beat: int,
) -> float:
    for base_tick, base_time, tempo in reversed(tempo_map):
        if tick >= base_tick:
            delta_ticks = tick - base_tick
            seconds = base_time + (delta_ticks * tempo) / (ticks_per_beat * 1_000_000.0)
            return seconds
    return 0.0


def parse_midi(path: Path) -> List[NoteEvent]:
    mid = mido.MidiFile(path)
    tempo_map = build_tempo_map(mid)
    ticks_per_beat = mid.ticks_per_beat
    events: List[NoteEvent] = []
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        track_name = f"Track{track_idx}"
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "track_name":
                track_name = msg.name or track_name
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            if msg.type in {"note_off"} or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key not in active:
                    continue
                start_tick, velocity = active.pop(key)
                start_s = ticks_to_seconds(start_tick, tempo_map, ticks_per_beat)
                end_s = ticks_to_seconds(abs_tick, tempo_map, ticks_per_beat)
                events.append(
                    NoteEvent(
                        start_s=start_s,
                        end_s=end_s,
                        pitch=msg.note,
                        velocity=velocity,
                        channel=msg.channel,
                        track=track_name,
                    )
                )
    events.sort(key=lambda ev: ev.start_s)
    return events


def parse_musicxml(path: Path) -> List[NoteEvent]:
    global MUSIC21_AVAILABLE
    if not MUSIC21_AVAILABLE:
        try:
            import music21  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "music21 is required to parse MusicXML files. Install it via pip."
            ) from exc
        MUSIC21_AVAILABLE = True
    import music21  # type: ignore

    score = music21.converter.parse(path)
    events: List[NoteEvent] = []
    for part_idx, part in enumerate(score.parts):
        part_name = part.partName or f"Part{part_idx}"
        for note in part.recurse().notes:
            if note.isRest:
                continue
            pitch = note.pitch.midi
            start_s = float(note.offset)
            duration_s = float(note.quarterLength)
            events.append(
                NoteEvent(
                    start_s=start_s,
                    end_s=start_s + duration_s,
                    pitch=pitch,
                    velocity=int(getattr(note, "volume", None) or 100),
                    channel=part_idx % 16,
                    track=part_name,
                )
            )
    events.sort(key=lambda ev: ev.start_s)
    return events


def load_mapping(path: Optional[Path]) -> List[Dict]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("objects", [])


def match_rule(note: NoteEvent, rules: Sequence[Dict]) -> Dict:
    for rule in rules:
        channel = rule.get("channel")
        track_kw = rule.get("track_contains")
        if track_kw and track_kw.lower() in note.track.lower():
            return rule
        if channel is not None and int(channel) == note.channel:
            return rule
    return {}


def adjust_params(params: SpatialParams, rule: Dict) -> SpatialParams:
    angle = params.angle_rad + math.radians(rule.get("azimuth_offset_deg", 0.0))
    radius = params.radius * rule.get("radius_scale", 1.0) + rule.get("radius_bias", 0.0)
    radius = clamp(radius, 0.0, 1.0)
    z = clamp(params.z + rule.get("height_offset", 0.0), -1.0, 1.0)
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    spread = clamp(
        params.spread * rule.get("spread_scale", 1.0) + rule.get("spread_bias", 0.0),
        0.0,
        1.0,
    )
    gain = clamp(params.gain * rule.get("gain_scale", 1.0), 0.0, 1.0)
    return SpatialParams(
        radius=radius,
        angle_rad=angle,
        x=x,
        y=y,
        z=z,
        spread=spread,
        gain=gain,
        tension=params.tension,
    )


def build_osc_events(
    notes: Sequence[NoteEvent],
    rules: Sequence[Dict],
    engine: SpatialRuleEngine,
) -> List[OscEvent]:
    events: List[OscEvent] = []
    for note in notes:
        base = engine.note_to_spatial(note)
        rule = match_rule(note, rules)
        if rule:
            params = adjust_params(base, rule)
            object_id = rule["object_id"]
        else:
            params = base
            object_id = note.channel + 1
        obj_prefix = f"/AfcImage/Object/{object_id}"
        label = f"{note.track} n{note.pitch}"
        events.append(
            OscEvent(
                time_s=note.start_s,
                address=f"{obj_prefix}/Position",
                args=[params.x, params.y, params.z],
                note=note,
                label=label,
            )
        )
        events.append(
            OscEvent(
                time_s=note.start_s,
                address=f"{obj_prefix}/Spread",
                args=[params.spread],
                note=note,
                label=label,
            )
        )
        events.append(
            OscEvent(
                time_s=note.start_s,
                address=f"{obj_prefix}/Gain",
                args=[params.gain],
                note=note,
                label=label,
            )
        )
        events.append(
            OscEvent(
                time_s=note.end_s,
                address=f"{obj_prefix}/Gain",
                args=[0.0],
                note=note,
                label=f"{label} release",
            )
        )
    events.sort(key=lambda ev: ev.time_s)
    return events


async def dispatch_events(
    events: Sequence[OscEvent],
    client: udp_client.SimpleUDPClient,
    start_delay: float,
    mmc_port_name: Optional[str],
) -> None:
    mmc_port = mido.open_output(mmc_port_name) if mmc_port_name else None
    try:
        if mmc_port:
            mmc_port.send(mido.Message("stop"))
        await asyncio.sleep(max(0.0, start_delay))
        if mmc_port:
            mmc_port.send(mido.Message("start"))
        if not events:
            return
        timeline_start = events[0].time_s
        base_time = time.perf_counter()
        for event in events:
            target = (event.time_s - timeline_start)
            now = time.perf_counter() - base_time
            wait = target - now
            if wait > 0:
                await asyncio.sleep(wait)
            client.send_message(event.address, list(event.args))
            print(f"[{event.time_s:7.3f}s] {event.address} {event.args} ({event.label})")
        if mmc_port:
            mmc_port.send(mido.Message("stop"))
    finally:
        if mmc_port:
            mmc_port.close()


def infer_format(path: Path, forced: Optional[str]) -> str:
    if forced and forced != "auto":
        return forced
    ext = path.suffix.lower()
    if ext in {".mid", ".midi"}:
        return "midi"
    if ext in {".xml", ".musicxml"}:
        return "musicxml"
    raise ValueError("Unable to infer file format. Use --format explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map MIDI/MusicXML notes to Atmos object panning via OSC."
    )
    parser.add_argument("--score", required=True, help="Path to MIDI or MusicXML file.")
    parser.add_argument(
        "--format",
        choices=["auto", "midi", "musicxml"],
        default="auto",
        help="Force file format detection.",
    )
    parser.add_argument("--key", default="C", help="Key center (e.g., C, D, Bb).")
    parser.add_argument(
        "--mode",
        default="major",
        help="Mode (major, ionian, dorian, minor...).",
    )
    parser.add_argument("--pitch-low", type=int, default=36)
    parser.add_argument("--pitch-high", type=int, default=96)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9021)
    parser.add_argument(
        "--mapping",
        help="JSON mapping file with object_id rules (track_contains/channel).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the first events instead of sending OSC.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of events to preview when --preview is set.",
    )
    parser.add_argument(
        "--mmc-port",
        help="Optional MIDI output port to send start/stop (for Nuendo transport).",
    )
    parser.add_argument(
        "--preroll",
        type=float,
        default=1.5,
        help="Seconds between transport start and first OSC event.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_path = Path(args.score).expanduser()
    fmt = infer_format(score_path, args.format)
    if fmt == "midi":
        note_events = parse_midi(score_path)
    else:
        note_events = parse_musicxml(score_path)
    if not note_events:
        raise SystemExit("No note events detected in the provided score.")
    rules = load_mapping(Path(args.mapping)) if args.mapping else []
    engine = SpatialRuleEngine(
        key=args.key,
        mode=args.mode,
        pitch_low=args.pitch_low,
        pitch_high=args.pitch_high,
    )
    events = build_osc_events(note_events, rules, engine)
    if args.preview:
        for event in events[: args.limit]:
            print(f"{event.time_s:7.3f}s | {event.address} -> {event.args} ({event.label})")
        print(f"... total events: {len(events)}")
        return
    client = udp_client.SimpleUDPClient(args.osc_host, args.osc_port)
    asyncio.run(
        dispatch_events(
            events=events,
            client=client,
            start_delay=args.preroll,
            mmc_port_name=args.mmc_port,
        )
    )


if __name__ == "__main__":
    main()
