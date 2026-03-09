"""Analyze ADM objects extracting positional timelines, per-object statistics, and global metrics.

Usage example:
    python analyze_adm_objects.py --adm PATH/mix_objects.wav --hop-ms 10 \
        --outdir out_objects --json-summary out_objects/summary.json \
        --csv-timeline out_objects/objects_timeline.csv

The script parses ADM metadata (either from an ADM BWF WAV or a standalone XML),
expands audioBlockFormat entries into a regular time grid, computes motion and
spatial KPIs, and generates optional plots. Timeline CSV columns include
`x_lr`, `y_fb`, `z_height` to clarify axes (left/right, front/back, elevation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for CLI usage
import matplotlib.pyplot as plt

AXIS_CENTER_EPS = 0.02  # tolerance for considering an object on an axis center


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


def local_name(tag: str) -> str:
    return tag.split('}', 1)[-1]


def parse_adm_time(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    txt = value.strip()
    if not txt:
        return None
    if txt.startswith('PT'):
        # ISO 8601 duration such as PT1H2M3.5S
        hours = minutes = seconds = 0.0
        match = re.findall(r'(\d+(?:\.\d+)?)([HMS])', txt)
        for number, unit in match:
            num = float(number)
            if unit == 'H':
                hours = num
            elif unit == 'M':
                minutes = num
            elif unit == 'S':
                seconds = num
        return hours * 3600.0 + minutes * 60.0 + seconds
    if '/' in txt:
        num, den = txt.split('/', 1)
        try:
            return float(num) / float(den)
        except Exception:
            return None
    if ':' in txt:
        parts = txt.split(':')
        parts = [float(p) for p in parts]
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
            continue
        if name == 'position':
            coord = child.attrib.get('coordinate', '').lower()
            try:
                coords[coord] = float((child.text or '0').strip())
            except Exception:
                coords[coord] = 0.0
        elif name == 'gain':
            spread = spread  # handled elsewhere if needed
        elif name == 'width':
            try:
                width = float((child.text or '0').strip())
            except Exception:
                width = 0.0
        elif name == 'height':
            try:
                height = float((child.text or '0').strip())
            except Exception:
                height = 0.0
        elif name == 'depth':
            try:
                depth = float((child.text or '0').strip())
            except Exception:
                depth = 0.0
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
    current_time = 0.0
    min_duration = max(default_duration, 1e-3)
    for block in channel_elem:
        if local_name(block.tag) != 'audioBlockFormat':
            continue
        start = parse_adm_time(block.attrib.get('rtime'))
        if start is None:
            start = current_time
        duration = parse_adm_time(block.attrib.get('duration'))
        x, y, z, spread, width, height, depth = parse_block_positions(block)
        gain_db = 0.0
        gain_elem = next((ch for ch in block if local_name(ch.tag) == 'gain'), None)
        if gain_elem is not None:
            gain_db = parse_gain(gain_elem)
        blocks.append(dict(
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
        ))
        current_time = blocks[-1]['time_end'] if blocks[-1]['time_end'] is not None else start
    for idx, entry in enumerate(blocks):
        if entry['time_end'] is None:
            if idx + 1 < len(blocks):
                entry['time_end'] = blocks[idx + 1]['time_start']
            else:
                entry['time_end'] = entry['time_start'] + default_duration
        if entry['time_end'] <= entry['time_start']:
            entry['time_end'] = entry['time_start'] + min_duration
    return blocks


def extract_axml_from_wav(path: str) -> Optional[str]:
    with open(path, 'rb') as f:
        header = f.read(12)
        if len(header) < 12:
            return None
        riff_id = header[0:4]
        if riff_id not in (b'RIFF', b'BW64', b'RF64'):
            return None
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)
            chunk_name = chunk_id.decode('ascii', errors='ignore')
            payload = f.read(chunk_size)
            if chunk_size % 2 == 1:
                f.seek(1, os.SEEK_CUR)
            if chunk_name.lower() == 'axml':
                try:
                    return payload.decode('utf-8')
                except UnicodeDecodeError:
                    return payload.decode('utf-16-le', errors='ignore')
    return None


def load_adm_xml(path: str) -> ET.Element:
    if path.lower().endswith(('.xml', '.adm')):
        tree = ET.parse(path)
        return tree.getroot()
    axml = extract_axml_from_wav(path)
    if not axml:
        raise ValueError("Impossibile trovare chunk ADM axml nel WAV.")
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
                ref = child.attrib.get('audioTrackUIDRef') or child.text
                if ref:
                    track_refs.append(ref.strip())
        if not track_refs:
            continue
        for track_ref in track_refs:
            track_elem = track_uid_map.get(track_ref)
            if track_elem is None:
                continue
            channel_ref: Optional[str] = track_elem.attrib.get('audioChannelFormatIDRef')
            # If channel format not referenced directly, follow Track→TrackFormat→Stream→Channel chain.
            if not channel_ref:
                track_fmt_ref = None
                for child in track_elem:
                    name = local_name(child.tag)
                    if name == 'audioTrackFormatIDRef':
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
                entries.append(BlockEntry(
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
                ))
    return entries


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


def axis_bucket(value: float, neg_label: str, pos_label: str, eps: float = AXIS_CENTER_EPS) -> str:
    if value <= -eps:
        return neg_label
    if value >= eps:
        return pos_label
    return 'center'


def classify_zone(row: pd.Series) -> str:
    fb = axis_bucket(row['y_fb'], 'rear', 'front')
    lr = axis_bucket(row['x_lr'], 'left', 'right')
    height = 'high' if row['z_height'] >= 0.3 else ('low' if row['z_height'] <= -0.3 else 'mid')
    return f"{fb}_{lr}_{height}"


def build_timeline(df_blocks: pd.DataFrame, hop_s: float) -> pd.DataFrame:
    if df_blocks.empty:
        return pd.DataFrame(columns=['time_s', 'object_id'])
    start = df_blocks['time_start'].min()
    end = df_blocks['time_end'].max()
    time_grid = np.arange(start, end + hop_s / 2.0, hop_s)
    rows = []
    for object_id, group in df_blocks.groupby('object_id'):
        object_name = group['object_name'].iloc[0]
        sampled = resample_object(group.sort_values('time_start'), time_grid, hop_s)
        if sampled.empty:
            continue
        sampled['object_id'] = object_id
        sampled['object_name'] = object_name
        rows.append(sampled)
    if not rows:
        return pd.DataFrame(columns=['time_s', 'object_id'])
    timeline = pd.concat(rows, ignore_index=True)
    timeline.sort_values(['time_s', 'object_id'], inplace=True)
    return timeline


def compute_object_summary(timeline: pd.DataFrame, hop_s: float, motion_threshold: float) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame()
    timeline['is_moving'] = timeline['speed'].abs() >= motion_threshold
    grouped = timeline.groupby('object_id')
    summary = grouped.agg(
        object_name=('object_name', 'first'),
        total_active_frames=('time_s', 'count'),
        total_active_s=('time_s', lambda s: len(s) * hop_s),
        mean_gain_db=('gain_db', 'mean'),
        mean_speed=('speed', 'mean'),
        max_speed=('speed', 'max'),
        mean_accel=('accel', 'mean'),
        movement_pct=('is_moving', 'mean'),
        x_min=('x_lr', 'min'),
        x_max=('x_lr', 'max'),
        y_min=('y_fb', 'min'),
        y_max=('y_fb', 'max'),
        z_min=('z_height', 'min'),
        z_max=('z_height', 'max'),
        lr_mean=('x_lr', 'mean'),
        fb_mean=('y_fb', 'mean'),
        height_mean_pos=('z_height', 'mean'),
        spread_mean=('spread', 'mean'),
        width_mean=('width', 'mean'),
        height_mean=('height', 'mean'),
        depth_mean=('depth', 'mean'),
    ).reset_index()
    summary['movement_pct'] = summary['movement_pct'] * 100.0
    summary['range_x'] = summary['x_max'] - summary['x_min']
    summary['range_y'] = summary['y_max'] - summary['y_min']
    summary['range_z'] = summary['z_max'] - summary['z_min']
    def dominant_axis(sub_df: pd.DataFrame) -> str:
        vx = sub_df['velocity_x'].to_numpy()
        vy = sub_df['velocity_y'].to_numpy()
        vz = sub_df['velocity_z'].to_numpy()
        var_x = np.var(vx)
        var_y = np.var(vy)
        var_z = np.var(vz)
        axis = np.argmax([var_x, var_y, var_z])
        return ['LR', 'FB', 'Height'][axis]
    dominant = {obj_id: dominant_axis(group) for obj_id, group in grouped}
    summary['dominant_axis'] = summary['object_id'].map(dominant)
    return summary


def compute_global_metrics(timeline: pd.DataFrame, hop_s: float, motion_threshold: float) -> Dict[str, object]:
    if timeline.empty:
        return {}
    timeline = timeline.copy()
    timeline['is_moving'] = timeline['speed'].abs() >= motion_threshold
    timeline['zone'] = timeline.apply(classify_zone, axis=1)
    fb_vals = timeline['y_fb'].to_numpy()
    lr_vals = timeline['x_lr'].to_numpy()
    height_vals = timeline['z_height'].to_numpy()
    active_counts = timeline.groupby('time_s')['object_id'].nunique()
    moving_counts = timeline[timeline['is_moving']].groupby('time_s')['object_id'].nunique()
    total_time = len(active_counts) * hop_s
    total_object_time = len(timeline) * hop_s
    zone_density = timeline.groupby('zone')['object_id'].count() * hop_s
    zone_density = (zone_density / total_object_time).to_dict()
    def axis_fractions(values: np.ndarray, neg_label: str, pos_label: str) -> Dict[str, float]:
        neg = float((values < -AXIS_CENTER_EPS).sum()) * hop_s
        pos = float((values > AXIS_CENTER_EPS).sum()) * hop_s
        center = float(values.size * hop_s - neg - pos)
        if total_object_time:
            return {
                neg_label: neg / total_object_time,
                'center': center / total_object_time,
                pos_label: pos / total_object_time,
            }
        return {neg_label: 0.0, 'center': 0.0, pos_label: 0.0}

    fb_density = axis_fractions(fb_vals, 'rear', 'front')
    lr_density = axis_fractions(lr_vals, 'left', 'right')
    height_density = axis_fractions(height_vals, 'low', 'high')
    global_summary = {
        'timeline_start_s': float(timeline['time_s'].min()),
        'timeline_end_s': float(timeline['time_s'].max()),
        'total_duration_s': float(total_time),
        'active_object_count_mean': float(active_counts.mean()),
        'active_object_count_max': int(active_counts.max()),
        'pct_frames_with_motion': float(len(moving_counts) / len(active_counts) * 100.0),
        'zone_density_time_fraction': zone_density,
        'front_back_time_fraction': fb_density,
        'left_right_time_fraction': lr_density,
        'height_time_fraction': height_density,
        'speed_percentiles': {
            'p50': float(timeline['speed'].quantile(0.5)),
            'p90': float(timeline['speed'].quantile(0.9)),
            'p99': float(timeline['speed'].quantile(0.99)),
        },
        'spread_mean': float(timeline['spread'].mean()),
    }
    return global_summary


def save_plots(timeline: pd.DataFrame, summary: Dict[str, object], outdir: str, prefix: str = "") -> None:
    if timeline.empty:
        return
    os.makedirs(outdir, exist_ok=True)
    timeline = timeline.copy()
    if 'zone' not in timeline.columns:
        timeline['zone'] = timeline.apply(classify_zone, axis=1)
    time_vals = timeline['time_s']
    active_counts = timeline.groupby('time_s')['object_id'].nunique()
    plt.figure(figsize=(8, 4))
    plt.plot(active_counts.index, active_counts.values)
    plt.xlabel('Time (s)')
    plt.ylabel('Active objects')
    plt.title('Active Object Count Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}objects_active_count.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(timeline['x_lr'], timeline['y_fb'], c=time_vals, s=8, cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='Time (s)')
    plt.xlabel('X (LR)')
    plt.ylabel('Y (FB)')
    plt.title('Object positions XY')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}objects_xy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(timeline['speed'], bins=40, color='#3366cc', alpha=0.8)
    plt.xlabel('Speed (units/s)')
    plt.ylabel('Frames')
    plt.title('Speed distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}objects_speed_hist.png"), dpi=150)
    plt.close()

    fb_counts = [
        int((timeline['y_fb'] > AXIS_CENTER_EPS).sum()),
        int((np.abs(timeline['y_fb']) <= AXIS_CENTER_EPS).sum()),
        int((timeline['y_fb'] < -AXIS_CENTER_EPS).sum()),
    ]
    lr_counts = [
        int((timeline['x_lr'] < -AXIS_CENTER_EPS).sum()),
        int((np.abs(timeline['x_lr']) <= AXIS_CENTER_EPS).sum()),
        int((timeline['x_lr'] > AXIS_CENTER_EPS).sum()),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(['front', 'center', 'rear'], fb_counts, color='#cc8844')
    ax[0].set_title('Front vs Rear occupancy')
    ax[0].set_ylabel('Frame count')
    ax[1].bar(['left', 'center', 'right'], lr_counts, color='#4488cc')
    ax[1].set_title('Left vs Right occupancy')
    for axis in ax:
        axis.set_xlabel('Zone')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}objects_axis_density.png"), dpi=150)
    plt.close(fig)

    zone_counts = timeline.groupby('zone')['object_id'].count().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    plt.bar(zone_counts.index, zone_counts.values, color='#cc6633')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Frame count')
    plt.title('Zone occupancy')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}objects_zone_density.png"), dpi=150)
    plt.close()


def parse_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze ADM objects and extract spatial KPIs.")
    ap.add_argument('--adm', required=True, help='Percorso WAV ADM BWF o file XML con metadata ADM.')
    ap.add_argument('--hop-ms', type=float, default=10.0, help='Passo temporale per la normalizzazione (ms).')
    ap.add_argument('--default-block-ms', type=float, default=10.0, help='Durata di fallback per block senza duration (ms).')
    ap.add_argument('--motion-threshold', type=float, default=0.01, help='Soglia velocità per definire un oggetto in movimento.')
    ap.add_argument('--outdir', default='out', help='Cartella output.')
    ap.add_argument('--csv-timeline', default=None, help='Path CSV per timeline normalizzata.')
    ap.add_argument('--csv-objects', default=None, help='Path CSV per metriche per oggetto.')
    ap.add_argument('--json-summary', default=None, help='Path JSON per metriche globali.')
    ap.add_argument('--plots', action='store_true', help='Genera PNG di supporto (active count, XY, speed, assi front/back & left/right, zone).')
    ap.add_argument('--probe', action='store_true', help='Mostra info su oggetti e termina.')
    return ap.parse_args()


def main() -> None:
    args = parse_arguments()
    hop_s = args.hop_ms / 1000.0
    default_block_s = args.default_block_ms / 1000.0
    root = load_adm_xml(args.adm)
    entries = collect_block_entries(root, default_block_s)
    if not entries:
        raise RuntimeError('Nessun audioObject con block format trovato nell\'ADM.')
    df_blocks = pd.DataFrame([entry.__dict__ for entry in entries])
    if args.probe:
        info = df_blocks.groupby('object_id')['object_name'].first()
        print('=== ADM OBJECTS ===')
        for oid, name in info.items():
            count = len(df_blocks[df_blocks['object_id'] == oid])
            print(f"{oid:20s} | {name:30s} | blocks={count}")
        print('====================')
        return
    timeline = build_timeline(df_blocks, hop_s)
    if timeline.empty:
        raise RuntimeError('Timeline vuota dopo la normalizzazione temporale.')
    object_summary = compute_object_summary(timeline.copy(), hop_s, args.motion_threshold)
    global_summary = compute_global_metrics(timeline.copy(), hop_s, args.motion_threshold)
    os.makedirs(args.outdir, exist_ok=True)
    timeline_path = args.csv_timeline or os.path.join(args.outdir, 'objects_timeline.csv')
    objects_path = args.csv_objects or os.path.join(args.outdir, 'objects_summary.csv')
    json_path = args.json_summary or os.path.join(args.outdir, 'objects_global_summary.json')
    timeline.drop(columns=['active'], errors='ignore').to_csv(timeline_path, index=False)
    object_summary.to_csv(objects_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(global_summary, f, indent=2)
    if args.plots:
        save_plots(timeline.copy(), global_summary, args.outdir)
    print('=== ADM OBJECT SUMMARY ===')
    print(f"Timeline CSV : {timeline_path}")
    print(f"Objects CSV  : {objects_path}")
    print(f"Global JSON  : {json_path}")
    print(f"Objects found: {object_summary.shape[0]}")
    print(f"Duration (s) : {global_summary.get('total_duration_s', 0.0):.3f}")
    print(f"Active count mean/max: {global_summary.get('active_object_count_mean', 0.0):.2f} / {global_summary.get('active_object_count_max', 0)}")
    print('===========================')


if __name__ == '__main__':
    main()
