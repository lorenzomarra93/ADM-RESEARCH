#!/usr/bin/env python3
"""Multichannel Motion Analysis
================================

Clean CLI that extracts spatial descriptors from fixed-loudspeaker mixes
(5.1 / 7.1 / 7.1.2) and stores CSV + panel plots ready for the
`metadata_agent`, `feature_extraction_agent`, and `visualization_agent` workflows.

Example::

    python 02_analysis_pipeline/scripts/01_multichannel_analysis/multichannel_motion_analysis.py \
        --wav path/to/mix.wav --layout auto --outdir 02_analysis_pipeline/outputs --plots

The script computes per-frame centroids, spreads, velocities, and summary KPIs.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

AXIS_CENTER_EPS = 0.02
EPS = 1e-12

DEFAULT_LAYOUTS: Dict[str, Tuple[str, ...]] = {
    "51": ("L", "R", "C", "LFE", "Ls", "Rs"),
    "71": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"),
    "712": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf"),
}

ITU_ANGLES: Dict[str, Tuple[float, float]] = {
    "L": (-30.0, 0.0),
    "R": (30.0, 0.0),
    "C": (0.0, 0.0),
    "Ls": (-90.0, 0.0),
    "Rs": (90.0, 0.0),
    "Lrs": (-150.0, 0.0),
    "Rrs": (150.0, 0.0),
    "Ltf": (-30.0, 30.0),
    "Rtf": (30.0, 30.0),
    "Ltb": (-150.0, 30.0),
    "Rtb": (150.0, 30.0),
    "LFE": (0.0, 0.0),
}


@dataclass
class FrameMetrics:
    time_s: float
    x_lr: float
    y_fb: float
    z_height: float
    spread: float
    energy_db: float
    cy_speed: float
    zone: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract spatial descriptors from multichannel WAVs")
    ap.add_argument("--wav", required=True, help="Path to the multichannel WAV file")
    ap.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "51", "71", "712", "custom"],
        help="Predefined layout shortcut",
    )
    ap.add_argument(
        "--order",
        default=None,
        help="Comma-separated channel order when --layout custom (e.g. L,R,C,LFE,Ls,Rs)",
    )
    ap.add_argument("--win-ms", type=float, default=20.0, help="Analysis window in ms")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="Hop size in ms")
    ap.add_argument("--outdir", default="02_analysis_pipeline/outputs", help="Output root directory")
    ap.add_argument("--base-name", default=None, help="Optional base name for artifacts")
    ap.add_argument("--plots", action="store_true", help="Save diagnostic PNG panels")
    ap.add_argument("--json", action="store_true", help="Save KPI summary as JSON")
    ap.add_argument("--itu", action="store_true", help="Use ITU azimuth/elevation vectors")
    return ap.parse_args()


def resolve_order(layout: str, order_arg: str | None, num_channels: int) -> Tuple[str, ...]:
    if order_arg:
        labels = tuple(s.strip() for s in order_arg.split(",") if s.strip())
        if len(labels) != num_channels:
            raise ValueError("Custom order length mismatch vs channel count")
        return labels
    if layout == "auto":
        mapping = {6: "51", 8: "71", 10: "712"}
        if num_channels not in mapping:
            raise ValueError(f"Cannot infer layout for {num_channels} channels; pass --order")
        layout = mapping[num_channels]
    if layout == "custom":
        raise ValueError("--layout custom requires --order")
    labels = DEFAULT_LAYOUTS[layout]
    if len(labels) != num_channels:
        raise ValueError(f"Layout {layout} expects {len(labels)} channels; file has {num_channels}")
    return labels


def unit_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.sin(az) * math.cos(el)
    y = math.cos(az) * math.cos(el)
    z = math.sin(el)
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return np.array([x / norm, y / norm, z / norm], dtype=float)


def build_vectors(order: Sequence[str], use_itu: bool) -> Tuple[np.ndarray, List[str]]:
    vectors = []
    labels = []
    fallback = {
        "L": (-1.0, 1.0, 0.0),
        "R": (1.0, 1.0, 0.0),
        "C": (0.0, 1.0, 0.0),
        "Ls": (-1.0, 0.0, 0.0),
        "Rs": (1.0, 0.0, 0.0),
        "Lrs": (-1.0, -1.0, 0.0),
        "Rrs": (1.0, -1.0, 0.0),
        "Ltf": (-1.0, 1.0, 1.0),
        "Rtf": (1.0, 1.0, 1.0),
        "Ltb": (-1.0, -1.0, 1.0),
        "Rtb": (1.0, -1.0, 1.0),
    }
    for label in order:
        if label.upper() == "LFE":
            continue
        if use_itu:
            if label not in ITU_ANGLES:
                raise KeyError(f"No ITU definition for {label}")
            vectors.append(unit_from_az_el(*ITU_ANGLES[label]))
        else:
            vec = fallback.get(label)
            if vec is None:
                raise KeyError(f"No fallback vector for {label}; specify --order with supported labels")
            vectors.append(unit_from_az_el(*vec))
        labels.append(label)
    return np.vstack(vectors), labels


def frames_iter(n_samples: int, win: int, hop: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += hop


def axis_zone(x: float, y: float, z: float) -> str:
    def bucket(val: float) -> str:
        if val < -AXIS_CENTER_EPS:
            return "neg"
        if val > AXIS_CENTER_EPS:
            return "pos"
        return "center"

    fb = bucket(y)
    lr = bucket(x)
    ht = bucket(z)
    return f"{fb}_{lr}_{ht}"


def compute_metrics(
    audio: np.ndarray,
    fs: int,
    order: Tuple[str, ...],
    use_itu: bool,
    win_ms: float,
    hop_ms: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    win = int(round(fs * win_ms / 1000.0))
    hop = int(round(fs * hop_ms / 1000.0))
    if win <= 0 or hop <= 0:
        raise ValueError("win_ms and hop_ms must be > 0")
    coords, labels = build_vectors(order, use_itu)
    label_to_idx = {lab: idx for idx, lab in enumerate(order)}
    active_idx = [label_to_idx[lab] for lab in labels]
    X = audio[:, active_idx].T
    n_dir, total = X.shape
    rows: List[FrameMetrics] = []
    prev_y = None
    dt = hop / fs
    for start, end in frames_iter(total, win, hop):
        seg = X[:, start:end]
        E = np.sqrt(np.mean(seg * seg, axis=-1) + EPS)
        weights = E / (np.sum(E) + EPS)
        centroid = (weights[:, None] * coords).sum(axis=0)
        spread = float(np.sqrt(np.sum(weights * np.linalg.norm(coords - centroid, axis=1) ** 2)))
        energy_db = 20.0 * math.log10(np.sum(E) + EPS)
        y_val = float(centroid[1])
        cy_speed = 0.0 if prev_y is None else (y_val - prev_y) / dt
        prev_y = y_val
        time_s = start / fs
        rows.append(
            FrameMetrics(
                time_s=time_s,
                x_lr=float(centroid[0]),
                y_fb=y_val,
                z_height=float(centroid[2]),
                spread=spread,
                energy_db=energy_db,
                cy_speed=cy_speed,
                zone=axis_zone(*centroid),
            )
        )
    df = pd.DataFrame([r.__dict__ for r in rows])
    summary = {
        "duration_s": float(len(audio) / fs),
        "frames": len(df),
        "fb_mean": float(df["y_fb"].mean()),
        "lr_mean": float(df["x_lr"].mean()),
        "spread_mean": float(df["spread"].mean()),
        "spread_max": float(df["spread"].max()),
        "cy_speed_peak": float(df["cy_speed"].abs().max()),
    }
    return df, summary


def save_outputs(
    df: pd.DataFrame,
    summary: Dict[str, float],
    csv_dir: Path,
    json_dir: Path,
    plots_dir: Path,
    base_name: str,
    plots: bool,
    json_flag: bool,
) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{base_name}_motion.csv"
    df.to_csv(csv_path, index=False)
    if json_flag:
        json_dir.mkdir(parents=True, exist_ok=True)
        with (json_dir / f"{base_name}_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    if plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        render_panel(df, plots_dir / f"{base_name}_panel.png")


def render_panel(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].scatter(df["x_lr"], df["y_fb"], s=4, alpha=0.6)
    axes[0].set_title("XY Trajectory")
    axes[0].set_xlabel("LR")
    axes[0].set_ylabel("FB")
    axes[1].plot(df["time_s"], df["y_fb"], label="FB")
    axes[1].set_title("Front/Back Index")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[2].plot(df["time_s"], df["x_lr"], color="C1", label="LR")
    axes[2].set_title("Left/Right Index")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[3].plot(df["time_s"], df["spread"], color="C2", label="Spread")
    axes[3].set_title("Spread")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    audio, fs = sf.read(wav_path)
    if audio.ndim != 2:
        raise ValueError("Expected interleaved multichannel WAV")
    order = resolve_order(args.layout, args.order, audio.shape[1])
    df, summary = compute_metrics(audio, fs, order, args.itu, args.win_ms, args.hop_ms)
    base_name = args.base_name or wav_path.stem
    base_out = Path(args.outdir)
    save_outputs(
        df,
        summary,
        csv_dir=base_out / "csv",
        json_dir=base_out / "json",
        plots_dir=base_out / "plots",
        base_name=base_name,
        plots=args.plots,
        json_flag=args.json,
    )
    print(f"Saved descriptors to {base_out}")


if __name__ == "__main__":
    main()
