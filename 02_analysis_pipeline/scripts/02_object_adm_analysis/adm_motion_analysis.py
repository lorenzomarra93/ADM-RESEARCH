#!/usr/bin/env python3
"""ADM Object Motion Analysis

Refactored version of the legacy ADM parser that feeds the
`adm_parser_agent`, `feature_extraction_agent`, and `visualization_agent`.
It ingests an ADM-enabled BW64/WAV (or standalone XML), expands object
trajectories on a regular time grid, stores timelines/summary JSON, and
renders diagnostic motion plots.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AXIS_CENTER_EPS = 0.02


@dataclass
class BlockEntry:
    object_id: str
    object_name: str
    track_uid: Optional[str]
    channel_format: Optional[str]
    time_start: float
    time_end: float
    x: float
    y: float
    z: float
    gain_db: float
    spread: float
    width: float
    height: float
    depth: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Parse ADM objects and compute motion metrics")
    ap.add_argument("--adm", required=True, help="Path to ADM WAV/BW64 or ADM XML")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="Timeline hop size in ms")
    ap.add_argument("--motion-threshold", type=float, default=0.01, help="Speed threshold for active segments")
    ap.add_argument("--outdir", default="02_analysis_pipeline/outputs", help="Root output directory")
    ap.add_argument("--base-name", default=None, help="Override base name for artifacts")
    ap.add_argument("--plots", action="store_true", help="Export diagnostic PNG panels")
    ap.add_argument("--json", action="store_true", help="Write summary JSON")
    return ap.parse_args()


def local_name(tag: str) -> str:
    return tag.split('}', 1)[-1]


def parse_adm_time(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    txt = value.strip()
    if not txt:
        return None
    if txt.startswith('PT'):
        hours = minutes = seconds = 0.0
        for number, unit in re.findall(r'(\d+(?:\.\d+)?)([HMS])', txt):
            num = float(number)
            if unit == 'H':
                hours = num
            elif unit == 'M':
                minutes = num
            else:
                seconds = num
        return hours * 3600.0 + minutes * 60.0 + seconds
    if '/' in txt:
        num, den = txt.split('/', 1)
        try:
            return float(num) / float(den)
        except Exception:
            return None
    if ':' in txt:
        parts = [float(p) for p in txt.split(':')]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        hours, minutes, seconds = parts[-3:]
        return hours * 3600.0 + minutes * 60.0 + seconds
    if txt.lower().endswith('s'):
        try:
            return float(txt[:-1])
        except Exception:
            return None
    try:
        return float(txt)
    except Exception:
        return None


def parse_gain(elem: ET.Element) -> float:
    txt = (elem.text or '').strip()
    if not txt:
        return 0.0
    try:
        value = float(txt)
    except Exception:
        return 0.0
    unit = elem.attrib.get('unit', '').lower()
    normalisation = elem.attrib.get('normalisation', '').lower()
    if unit == 'linear' or normalisation == 'linear':
        if value <= 0:
            return -120.0
        return 20.0 * math.log10(value)
    return value


def parse_block_positions(block: ET.Element) -> Tuple[float, float, float, float, float, float, float]:
    cartesian_flag = block.attrib.get('cartesian', '').lower() == 'true'
    coords: Dict[str, float] = {}
    spread = width = height = depth = 0.0
    for child in block:
        name = local_name(child.tag)
        if name == 'cartesian':
            txt = (child.text or '').strip().lower()
            cartesian_flag = txt in {'1', 'true', 'yes'}
        elif name == 'position':
            coord = child.attrib.get('coordinate', '').lower()
            try:
                coords[coord] = float((child.text or '0').strip())
            except Exception:
                coords[coord] = 0.0
        elif name == 'gain':
            spread = spread
        elif name == 'width':
            width = float((child.text or '0').strip() or 0.0)
        elif name == 'height':
            height = float((child.text or '0').strip() or 0.0)
        elif name == 'depth':
            depth = float((child.text or '0').strip() or 0.0)
        elif name == 'objectdivergence':
            try:
                spread = float(child.attrib.get('value', child.attrib.get('horizontal', '0')))
            except Exception:
                spread = 0.0
    if cartesian_flag:
        x = coords.get('x', coords.get('X', 0.0))
        y = coords.get('y', coords.get('Y', 0.0))
        z = coords.get('z', coords.get('Z', 0.0))
    else:
        az = math.radians(coords.get('azimuth', coords.get('Azimuth', 0.0)))
        el = math.radians(coords.get('elevation', coords.get('Elevation', 0.0)))
        dist = coords.get('distance', coords.get('Distance', 1.0))
        x = dist * math.sin(az) * math.cos(el)
        y = dist * math.cos(az) * math.cos(el)
        z = dist * math.sin(el)
    return x, y, z, spread, width, height, depth


def extract_block_entries(channel_elem: ET.Element, default_duration: float) -> List[Dict[str, float]]:
    blocks: List[Dict[str, float]] = []
    min_duration = max(default_duration, 1e-3)
    current_time = 0.0
    for block in channel_elem:
        if local_name(block.tag) != 'audioBlockFormat':
            continue
        start = parse_adm_time(block.attrib.get('rtime'))
        if start is None:
            start = current_time
        duration = parse_adm_time(block.attrib.get('duration'))
        x, y, z, spread, width, height, depth = parse_block_positions(block)
        gain_elem = next((ch for ch in block if local_name(ch.tag) == 'gain'), None)
        gain_db = parse_gain(gain_elem) if gain_elem is not None else 0.0
        blocks.append(
            dict(
                time_start=start,
                time_end=start + duration if duration else None,
                x=x,
                y=y,
                z=z,
                gain_db=gain_db,
                spread=spread,
                width=width,
                height=height,
                depth=depth,
            )
        )
        current_time = blocks[-1]['time_end'] if blocks[-1]['time_end'] is not None else start
    for idx, entry in enumerate(blocks):
        if entry['time_end'] is None:
            entry['time_end'] = blocks[idx + 1]['time_start'] if idx + 1 < len(blocks) else entry['time_start'] + default_duration
        if entry['time_end'] <= entry['time_start']:
            entry['time_end'] = entry['time_start'] + min_duration
    return blocks


def extract_axml_from_wav(path: Path) -> Optional[str]:
    import struct

    with path.open('rb') as f:
        header = f.read(12)
        if len(header) < 12 or header[0:4] not in (b'RIFF', b'BW64', b'RF64'):
            return None
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)
            payload = f.read(chunk_size)
            if chunk_size % 2 == 1:
                f.seek(1, 1)
            chunk_name = chunk_id.decode('ascii', errors='ignore')
            if chunk_name.lower() == 'axml':
                try:
                    return payload.decode('utf-8')
                except UnicodeDecodeError:
                    return payload.decode('utf-16-le', errors='ignore')
    return None


def load_adm_xml(path: Path) -> ET.Element:
    if path.suffix.lower() in {'.xml', '.adm'}:
        return ET.parse(path).getroot()
    axml = extract_axml_from_wav(path)
    if not axml:
        raise ValueError("Unable to locate ADM axml chunk in WAV")
    return ET.fromstring(axml)


def collect_block_entries(root: ET.Element, default_duration: float) -> List[BlockEntry]:
    audio_objects: List[ET.Element] = []
    track_uid_map: Dict[str, ET.Element] = {}
    channel_format_map: Dict[str, ET.Element] = {}
    track_format_map: Dict[str, ET.Element] = {}
    stream_format_map: Dict[str, ET.Element] = {}
    for elem in root.iter():
        name = local_name(elem.tag)
        if name == 'audioObject':
            audio_objects.append(elem)
        elif name == 'audioTrackUID':
            key = (
                elem.attrib.get('audioTrackUID')
                or elem.attrib.get('audioTrackUIDID')
                or elem.attrib.get('UID')
            )
            if key:
                track_uid_map[key] = elem
        elif name == 'audioChannelFormat':
            key = elem.attrib.get('audioChannelFormatID') or elem.attrib.get('audioChannelFormatIDRef')
            if key:
                channel_format_map[key] = elem
        elif name == 'audioTrackFormat':
            key = elem.attrib.get('audioTrackFormatID')
            if key:
                track_format_map[key] = elem
        elif name == 'audioStreamFormat':
            key = elem.attrib.get('audioStreamFormatID')
            if key:
                stream_format_map[key] = elem
    entries: List[BlockEntry] = []
    for obj in audio_objects:
        object_id = obj.attrib.get('audioObjectID') or obj.attrib.get('audioObjectIDRef') or obj.attrib.get('ID')
        if not object_id:
            continue
        object_name = obj.attrib.get('audioObjectName') or obj.attrib.get('audioObjectLabel') or object_id
        track_refs: List[str] = []
        for child in obj:
            if local_name(child.tag) == 'audioTrackUIDRef':
                ref = child.attrib.get('audioTrackUIDRef') or (child.text or '').strip()
                if ref:
                    track_refs.append(ref)
        if not track_refs:
            continue
        for track_ref in track_refs:
            track_elem = track_uid_map.get(track_ref)
            if track_elem is None:
                continue
            channel_ref = track_elem.attrib.get('audioChannelFormatIDRef')
            if not channel_ref:
                track_fmt_ref = None
                for child in track_elem:
                    if local_name(child.tag) == 'audioTrackFormatIDRef':
                        track_fmt_ref = child.attrib.get('audioTrackFormatIDRef') or (child.text or '').strip()
                        break
                stream_ref = None
                if track_fmt_ref:
                    track_fmt_elem = track_format_map.get(track_fmt_ref)
                    if track_fmt_elem is not None:
                        for child in track_fmt_elem:
                            if local_name(child.tag) == 'audioStreamFormatIDRef':
                                stream_ref = child.attrib.get('audioStreamFormatIDRef') or (child.text or '').strip()
                                break
                if stream_ref:
                    stream_elem = stream_format_map.get(stream_ref)
                    if stream_elem is not None:
                        for child in stream_elem:
                            if local_name(child.tag) == 'audioChannelFormatIDRef':
                                channel_ref = child.attrib.get('audioChannelFormatIDRef') or (child.text or '').strip()
                                break
            channel_elem = channel_format_map.get(channel_ref)
            if channel_elem is None:
                continue
            blocks = extract_block_entries(channel_elem, default_duration)
            for block in blocks:
                entries.append(
                    BlockEntry(
                        object_id=object_id,
                        object_name=object_name,
                        track_uid=track_ref,
                        channel_format=channel_ref,
                        time_start=block['time_start'],
                        time_end=block['time_end'],
                        x=block['x'],
                        y=block['y'],
                        z=block['z'],
                        gain_db=block['gain_db'],
                        spread=block['spread'],
                        width=block['width'],
                        height=block['height'],
                        depth=block['depth'],
                    )
                )
    return entries


def build_timeline(entries: List[BlockEntry], hop_s: float, threshold: float) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    start = min(e.time_start for e in entries)
    end = max(e.time_end for e in entries)
    time_grid = np.arange(start, end + hop_s, hop_s)
    rows = []
    for obj_id, group in pd.DataFrame([e.__dict__ for e in entries]).groupby('object_id'):
        df = resample_object(group, time_grid, hop_s)
        if df.empty:
            continue
        df['object_id'] = obj_id
        df['object_name'] = group['object_name'].iloc[0]
        df['zone'] = df.apply(classify_zone, axis=1)
        df['is_moving'] = df['speed'] >= threshold
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    timeline = pd.concat(rows, ignore_index=True)
    timeline['active_objects'] = timeline.groupby('time_s')['object_id'].transform('count')
    return timeline


def resample_object(blocks: pd.DataFrame, time_grid: np.ndarray, hop_s: float) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame()
    times = blocks['time_start'].to_numpy()
    values = {
        'x': blocks['x'].to_numpy(),
        'y': blocks['y'].to_numpy(),
        'z': blocks['z'].to_numpy(),
        'gain_db': blocks['gain_db'].to_numpy(),
        'spread': blocks['spread'].fillna(0.0).to_numpy(),
        'width': blocks['width'].fillna(0.0).to_numpy(),
        'height': blocks['height'].fillna(0.0).to_numpy(),
        'depth': blocks['depth'].fillna(0.0).to_numpy(),
    }
    active = np.zeros_like(time_grid, dtype=bool)
    for row in blocks.itertuples():
        active |= ((time_grid >= row.time_start) & (time_grid < row.time_end))
    if not np.any(active):
        return pd.DataFrame()
    interp = {}
    for key, arr in values.items():
        if len(times) == 1:
            interp[key] = np.full_like(time_grid, arr[0], dtype=float)
        else:
            interp[key] = np.interp(time_grid, times, arr, left=arr[0], right=arr[-1])
    data = {
        'time_s': time_grid,
        'x_lr': interp['x'],
        'y_fb': interp['y'],
        'z_height': interp['z'],
        'gain_db': interp['gain_db'],
        'spread': interp['spread'],
        'width': interp['width'],
        'height': interp['height'],
        'depth': interp['depth'],
        'active': active,
    }
    df = pd.DataFrame(data)
    df = df[df['active']].copy()
    df['velocity_x'] = df['x_lr'].diff().fillna(0.0) / hop_s
    df['velocity_y'] = df['y_fb'].diff().fillna(0.0) / hop_s
    df['velocity_z'] = df['z_height'].diff().fillna(0.0) / hop_s
    df['speed'] = np.sqrt(df['velocity_x'] ** 2 + df['velocity_y'] ** 2 + df['velocity_z'] ** 2)
    df['accel'] = df['speed'].diff().fillna(0.0) / hop_s
    return df


def axis_bucket(value: float, neg: str, pos: str) -> str:
    if value <= -AXIS_CENTER_EPS:
        return neg
    if value >= AXIS_CENTER_EPS:
        return pos
    return 'center'


def classify_zone(row: pd.Series) -> str:
    fb = axis_bucket(row['y_fb'], 'rear', 'front')
    lr = axis_bucket(row['x_lr'], 'left', 'right')
    height = 'high' if row['z_height'] >= 0.3 else ('low' if row['z_height'] <= -0.3 else 'mid')
    return f"{fb}_{lr}_{height}"


def summarize_timeline(timeline: pd.DataFrame) -> Dict[str, object]:
    if timeline.empty:
        return {}
    summary = {
        'total_duration': float(timeline['time_s'].max() - timeline['time_s'].min()),
        'frames': int(len(timeline)),
        'object_count': int(timeline['object_id'].nunique()),
        'active_objects_mean': float(timeline.groupby('time_s')['object_id'].nunique().mean()),
        'avg_speed': float(timeline['speed'].mean()),
        'max_speed': float(timeline['speed'].max()),
    }
    zone_counts = timeline['zone'].value_counts(normalize=True).to_dict()
    summary['zone_distribution'] = zone_counts
    return summary


def render_plots(timeline: pd.DataFrame, outdir: Path, base_name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].scatter(timeline['x_lr'], timeline['y_fb'], s=3, alpha=0.5)
    axes[0].set_title('Objects XY trajectory')
    axes[0].set_xlabel('LR')
    axes[0].set_ylabel('FB')
    axes[1].plot(timeline['time_s'], timeline['speed'])
    axes[1].set_title('Speed per object')
    axes[1].set_xlabel('Time (s)')
    axes[2].plot(timeline['time_s'], timeline['z_height'])
    axes[2].set_title('Height over time')
    axes[3].plot(timeline['time_s'], timeline['spread'])
    axes[3].set_title('Spread')
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / f"{base_name}_objects_panel.png", dpi=150)
    plt.close(fig)


def save_outputs(
    timeline: pd.DataFrame,
    summary: Dict[str, object],
    out_root: Path,
    base_name: str,
    plots: bool,
    json_flag: bool,
) -> None:
    csv_dir = out_root / 'csv'
    json_dir = out_root / 'json'
    plots_dir = out_root / 'plots'
    csv_dir.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(csv_dir / f"{base_name}_objects_timeline.csv", index=False)
    if json_flag:
        json_dir.mkdir(parents=True, exist_ok=True)
        with (json_dir / f"{base_name}_objects_summary.json").open('w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
    if plots:
        render_plots(timeline, plots_dir, base_name)


def main() -> None:
    args = parse_args()
    adm_path = Path(args.adm)
    hop_s = args.hop_ms / 1000.0
    root = load_adm_xml(adm_path)
    entries = collect_block_entries(root, hop_s)
    timeline = build_timeline(entries, hop_s, args.motion_threshold)
    if timeline.empty:
        raise RuntimeError("Timeline is empty; check ADM metadata")
    summary = summarize_timeline(timeline) if args.json else {}
    base_name = args.base_name or adm_path.stem
    out_root = Path(args.outdir)
    save_outputs(timeline, summary, out_root, base_name, plots=args.plots, json_flag=args.json)
    print(f"Saved ADM timeline ({len(timeline)} rows) to {out_root}")


if __name__ == '__main__':
    main()
