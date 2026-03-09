#!/usr/bin/env python3
"""
analyze_multichannel_mix.py

Prototype tool to inspect channel-based film mixes (5.1, 7.1, 7.1.2 or custom order).
It extracts time-series spatial metrics, per-channel statistics and saves optional plots
so that ADM/Dolby Atmos object analyses can be compared with legacy multichannel beds.

Example:
    python analyze_multichannel_mix.py --wav path/to/mix.wav --layout auto \
        --itu --outdir out_mix --plots --json-summary out_mix/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

AXIS_CENTER_EPS = 0.02
EPS = 1e-12

# Default layouts reflect cinema practice (SMPTE/ITU)
DEFAULT_LAYOUTS: Dict[str, Tuple[str, ...]] = {
    "51": ("L", "R", "C", "LFE", "Ls", "Rs"),
    "71": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs"),
    "712": ("L", "R", "C", "LFE", "Ls", "Rs", "Lrs", "Rrs", "Ltf", "Rtf"),
}

# Azimuth/elevation pairs (degrees) for loudspeaker labels
ITU_ANGLES: Dict[str, Tuple[float, float]] = {
    "L": (-30, 0),
    "R": (30, 0),
    "C": (0, 0),
    "Ls": (-110, 0),
    "Rs": (110, 0),
    "Lrs": (-150, 0),
    "Rrs": (150, 0),
    "Ltf": (-30, 30),
    "Rtf": (30, 30),
    "Ltb": (-150, 30),
    "Rtb": (150, 30),
    "LFE": (0, 0),  # ignored in centroid computations but kept for summaries
}

LEFT_LABELS = {"L", "Ls", "Lrs", "Ltf", "Ltb"}
RIGHT_LABELS = {"R", "Rs", "Rrs", "Rtf", "Rtb"}
FRONT_LABELS = {"L", "R", "C", "Ltf", "Rtf"}
REAR_LABELS = {"Lrs", "Rrs"}
SURROUND_LABELS = {"Ls", "Rs", "Lrs", "Rrs"}
HEIGHT_LABELS = {"Ltf", "Rtf", "Ltb", "Rtb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a 5.1/7.1 WAV to extract channel statistics and spatial indexes."
    )
    parser.add_argument("--wav", required=True, help="Path to the multichannel WAV file.")
    parser.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "51", "71", "712", "custom"],
        help="Speaker layout shortcut. Use 'auto' to infer by channel count or 'custom' with --order.",
    )
    parser.add_argument(
        "--order",
        default=None,
        help="Custom comma separated channel labels (e.g. L,R,C,LFE,Ls,Rs). Overrides --layout.",
    )
    parser.add_argument("--win-ms", type=float, default=20.0, help="Analysis window (ms).")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Analysis hop (ms).")
    parser.add_argument(
        "--smooth",
        default="ma:5",
        help="Smoothing mode for cy before derivatives. Formats: none | ma:N | sg:W:P.",
    )
    parser.add_argument("--outdir", default="out_multichannel", help="Output directory.")
    parser.add_argument("--plots", action="store_true", help="Generate PNG figures in outdir/plots.")
    parser.add_argument("--itu", action="store_true", help="Use ITU azimuth/elevation vectors.")
    parser.add_argument("--save-features", default=None, help="Optional CSV path for frame metrics.")
    parser.add_argument("--save-channel-summary", default=None, help="Optional CSV path for per-channel stats.")
    parser.add_argument("--json-summary", default=None, help="Optional JSON output path.")
    parser.add_argument("--probe", action="store_true", help="Print WAV info and exit.")
    return parser.parse_args()


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, EPS))


def resolve_order(
    layout: str,
    order_arg: str | None,
    num_channels: int,
) -> Tuple[str, ...]:
    if order_arg:
        labels = tuple(s.strip() for s in order_arg.split(",") if s.strip())
        if len(labels) != num_channels:
            raise ValueError("Custom --order does not match WAV channel count.")
        return labels
    if layout == "auto":
        if num_channels == 6:
            return DEFAULT_LAYOUTS["51"]
        if num_channels == 8:
            return DEFAULT_LAYOUTS["71"]
        if num_channels == 10:
            return DEFAULT_LAYOUTS["712"]
        raise ValueError(
            f"Cannot infer layout for {num_channels} channels. Specify --order explicitly."
        )
    if layout == "custom":
        raise ValueError("--layout custom requires --order.")
    labels = DEFAULT_LAYOUTS[layout]
    if len(labels) != num_channels:
        raise ValueError(f"Layout {layout} expects {len(labels)} channels, found {num_channels}.")
    return labels


def unit_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.sin(az) * math.cos(el)
    y = math.cos(az) * math.cos(el)
    z = math.sin(el)
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return np.array([x / norm, y / norm, z / norm], dtype=float)


def build_vectors(order: Sequence[str], use_itu: bool) -> Tuple[np.ndarray, List[str], List[int]]:
    coords = []
    labels = []
    original_indices = []
    for idx, label in enumerate(order):
        if label.upper() == "LFE":
            continue
        if use_itu:
            if label not in ITU_ANGLES:
                raise KeyError(f"No ITU angles for label '{label}'.")
            coords.append(unit_from_az_el(*ITU_ANGLES[label]))
        else:
            approx = {
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
            if label not in approx:
                raise KeyError(f"No fallback coordinates for '{label}'.")
            v = np.array(approx[label], dtype=float)
            coords.append(v / np.linalg.norm(v))
        labels.append(label)
        original_indices.append(idx)
    return np.vstack(coords), labels, original_indices


def frames_iter(n_samples: int, win: int, hop: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += hop


def parse_smooth_mode(mode_str: str) -> Tuple[str, Tuple[int, int] | int | None]:
    mode_str = (mode_str or "none").strip().lower()
    if mode_str == "none":
        return "none", None
    if mode_str.startswith("ma:"):
        return "ma", int(mode_str.split(":")[1])
    if mode_str.startswith("sg:"):
        parts = mode_str.split(":")
        if len(parts) != 3:
            raise ValueError("Savitzky–Golay needs sg:window:polyorder.")
        return "sg", (int(parts[1]), int(parts[2]))
    raise ValueError("Invalid --smooth. Use none | ma:N | sg:W:P.")


def smooth_series(x: np.ndarray, mode: Tuple[str, Tuple[int, int] | int | None]) -> np.ndarray:
    kind, param = mode
    if kind == "none":
        return x
    if kind == "ma":
        N = max(1, int(param or 1))
        if N <= 1:
            return x
        kernel = np.ones(N) / N
        return np.convolve(x, kernel, mode="same")
    if kind == "sg":
        try:
            from scipy.signal import savgol_filter
        except Exception:
            return smooth_series(x, ("ma", 7))
        window, poly = param  # type: ignore[misc]
        window = max(poly * 2 + 1, window)
        if window % 2 == 0:
            window += 1
        return savgol_filter(x, window, poly, mode="interp")
    return x


def frame_rms(frame: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frame * frame, axis=0) + EPS)


def spectral_stats(signal: np.ndarray, fs: int) -> Tuple[float, float]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    energy = spectrum.sum() + EPS
    centroid = float(np.dot(freqs, spectrum) / energy)
    bandwidth = float(np.sqrt(np.dot(((freqs - centroid) ** 2), spectrum) / energy))
    return centroid, bandwidth


def classify_zone(cx: float, cy: float) -> str:
    def bucket(value: float, neg: str, pos: str) -> str:
        if value <= -AXIS_CENTER_EPS:
            return neg
        if value >= AXIS_CENTER_EPS:
            return pos
        return "center"

    fb = bucket(cy, "rear", "front")
    lr = bucket(cx, "left", "right")
    return f"{fb}_{lr}"


def compute_frame_metrics(
    data: np.ndarray,
    fs: int,
    labels: Sequence[str],
    use_itu: bool,
    win_ms: float,
    hop_ms: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], float]:
    coords, dir_labels, dir_indices = build_vectors(labels, use_itu)
    win = max(1, int(fs * win_ms / 1000.0))
    hop = max(1, int(fs * hop_ms / 1000.0))
    dt = hop / fs
    n_samples = data.shape[0]
    channel_series: List[np.ndarray] = []
    rows = []
    for start, end in frames_iter(n_samples, win, hop):
        frame = data[start:end, :]
        rms = frame_rms(frame)
        channel_series.append(rms)
        dir_rms = rms[dir_indices]
        weight_sum = float(dir_rms.sum()) + EPS
        weights = dir_rms / weight_sum
        centroid = (weights[:, None] * coords).sum(axis=0)
        spread = float(np.sqrt(np.sum(((coords - centroid) ** 2).sum(axis=1) * weights)))
        left_share = float(sum(weights[i] for i, lab in enumerate(dir_labels) if lab in LEFT_LABELS))
        right_share = float(sum(weights[i] for i, lab in enumerate(dir_labels) if lab in RIGHT_LABELS))
        front_share = float(sum(weights[i] for i, lab in enumerate(dir_labels) if lab in FRONT_LABELS))
        rear_share = float(sum(weights[i] for i, lab in enumerate(dir_labels) if lab in REAR_LABELS))
        surround_share = float(sum(weights[i] for i, lab in enumerate(dir_labels) if lab in SURROUND_LABELS))
        height_share = float(sum(weights[i] for i, lab in enumerate(dir_labels) if lab in HEIGHT_LABELS))
        rows.append(
            {
                "cx": centroid[0],
                "cy": centroid[1],
                "cz": centroid[2],
                "theta": math.atan2(centroid[1], centroid[0]),
                "spread": spread,
                "LR_balance": left_share - right_share,
                "FB_balance": front_share - rear_share,
                "surround_share": surround_share,
                "height_share": height_share,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Audio shorter than analysis window.")
    df.insert(0, "time_s", np.arange(len(df)) * dt)
    channel_matrix = np.vstack(channel_series)
    # attach per-channel rms series
    for idx, label in enumerate(labels):
        df[f"rms_{label}"] = channel_matrix[:, idx]
    df["zone"] = df.apply(lambda row: classify_zone(row["cx"], row["cy"]), axis=1)
    return df, {label: channel_matrix[:, i] for i, label in enumerate(labels)}, dt


def channel_summary_table(
    data: np.ndarray,
    fs: int,
    labels: Sequence[str],
) -> pd.DataFrame:
    total_energy = float(np.sum(data ** 2)) + EPS
    rows = []
    for idx, label in enumerate(labels):
        sig = data[:, idx]
        rms = float(np.sqrt(np.mean(sig * sig) + EPS))
        peak = float(np.max(np.abs(sig)) + EPS)
        centroid, bandwidth = spectral_stats(sig, fs)
        energy_share = float(np.sum(sig * sig) / total_energy)
        rows.append(
            {
                "channel": label,
                "rms": rms,
                "rms_db": dbfs(rms),
                "peak": peak,
                "peak_db": dbfs(peak),
                "crest_db": dbfs(peak) - dbfs(rms),
                "spectral_centroid_hz": centroid,
                "spectral_bandwidth_hz": bandwidth,
                "energy_share": energy_share,
            }
        )
    return pd.DataFrame(rows)


def fractions_from_series(series: np.ndarray, dt: float, neg: str, pos: str) -> Dict[str, float]:
    total = len(series) * dt
    if total <= 0:
        return {neg: 0.0, "center": 0.0, pos: 0.0}
    neg_time = float(np.sum(series < -AXIS_CENTER_EPS)) * dt
    pos_time = float(np.sum(series > AXIS_CENTER_EPS)) * dt
    center_time = total - neg_time - pos_time
    return {neg: neg_time / total, "center": center_time / total, pos: pos_time / total}


def save_plots(
    df: pd.DataFrame,
    channel_series: Dict[str, np.ndarray],
    dt: float,
    outdir: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    t = df["time_s"].to_numpy()

    # Trajectory subplot
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(df["cx"], df["cy"])
    ax[0, 0].set_title("XY Trajectory")
    ax[0, 0].set_xlabel("cx (LR)")
    ax[0, 0].set_ylabel("cy (FB)")
    ax[0, 0].axis("equal")
    ax[0, 1].plot(t, df["LR_balance"])
    ax[0, 1].axhline(0, ls="--", lw=0.8, color="grey")
    ax[0, 1].set_title("Left-Right Balance")
    ax[0, 1].set_xlabel("Time (s)")
    ax[1, 0].plot(t, df["FB_balance"])
    ax[1, 0].axhline(0, ls="--", lw=0.8, color="grey")
    ax[1, 0].set_title("Front-Back Balance")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 1].plot(t, df["spread"])
    ax[1, 1].set_title("Spatial Spread")
    ax[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "panel_spatial.png"), dpi=160)
    plt.close(fig)

    # Channel heatmap
    channels = list(channel_series.keys())
    heatmap = np.vstack([channel_series[ch] for ch in channels])
    extent = [0, t[-1], 0, len(channels)]
    plt.figure(figsize=(10, 4))
    plt.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    plt.yticks(np.arange(len(channels)) + 0.5, channels)
    plt.xlabel("Time (s)")
    plt.title("Per-channel RMS Heatmap")
    plt.colorbar(label="RMS")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "channel_heatmap.png"), dpi=160)
    plt.close()

    # Zone occupancy bar plot
    zone_counts = df["zone"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(zone_counts.index, zone_counts.values, color="#cc6633")
    plt.ylabel("Frame count")
    plt.xticks(rotation=30, ha="right")
    plt.title("Zone Occupancy")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "zone_density.png"), dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.probe:
        info = sf.info(args.wav)
        print(info)
        return
    data, fs = sf.read(args.wav)
    if data.ndim != 2:
        raise ValueError("Expected an interleaved multichannel WAV.")
    order = resolve_order(args.layout, args.order, data.shape[1])
    df, channel_series, dt = compute_frame_metrics(
        data,
        fs,
        order,
        use_itu=args.itu,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
    )
    sm_mode = parse_smooth_mode(args.smooth)
    df["cy_smooth"] = smooth_series(df["cy"].to_numpy(), sm_mode)
    df["cy_speed"] = np.gradient(df["cy_smooth"].to_numpy(), dt)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    feature_path = args.save_features or os.path.join(outdir, "multichannel_features.csv")
    df.to_csv(feature_path, index=False)

    channel_df = channel_summary_table(data, fs, order)
    channel_summary_path = args.save_channel_summary or os.path.join(
        outdir, "channel_summary.csv"
    )
    channel_df.to_csv(channel_summary_path, index=False)

    fb_fraction = fractions_from_series(df["cy"].to_numpy(), dt, "rear", "front")
    lr_fraction = fractions_from_series(df["cx"].to_numpy(), dt, "left", "right")
    summary = {
        "file": os.path.abspath(args.wav),
        "sample_rate": fs,
        "layout": order,
        "win_ms": args.win_ms,
        "hop_ms": args.hop_ms,
        "smooth": args.smooth,
        "itu_vectors": bool(args.itu),
        "front_back_time_fraction": fb_fraction,
        "left_right_time_fraction": lr_fraction,
        "zone_fractions": (df["zone"].value_counts(normalize=True).to_dict()),
        "mean_spread": float(df["spread"].mean()),
        "median_height_share": float(df["height_share"].median()),
        "mean_surround_share": float(df["surround_share"].mean()),
    }
    summary_path = args.json_summary or os.path.join(outdir, "mix_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.plots:
        save_plots(df, channel_series, dt, os.path.join(outdir, "plots"))

    print("=== Multichannel Analysis ===")
    print(f"Features CSV : {feature_path}")
    print(f"Channel stats: {channel_summary_path}")
    print(f"Summary JSON : {summary_path}")
    print(f"Frames       : {len(df)} (dt={dt:.4f}s), smooth={args.smooth}, ITU={args.itu}")
    print(f"Front/Back   : {fb_fraction}")
    print(f"Left/Right   : {lr_fraction}")
    print(f"Zone fractions: {summary['zone_fractions']}")


if __name__ == "__main__":
    main()
