#!/usr/bin/env python3
"""Quick spectral probe for validation datasets.

Wraps the original `test_spectral.py` prototype into a CLI that stores
MFCC-based descriptors inside `02_analysis_pipeline/outputs/csv` and
an accompanying PNG useful for QA sessions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract spectral descriptors for sanity checks")
    ap.add_argument("--audio", required=True, help="Path to mono/multichannel WAV")
    ap.add_argument("--outdir", default="02_analysis_pipeline/outputs", help="Output root directory")
    ap.add_argument("--prefix", default=None, help="Optional prefix for saved files")
    ap.add_argument("--hop", type=float, default=0.01, help="Hop size in seconds")
    ap.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y[:, 0]
    hop_length = int(sr * args.hop)
    n_fft = 2048
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=args.n_mfcc, hop_length=hop_length, n_fft=n_fft)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)
    data = {
        "time_s": frame_times,
        "spec_centroid": centroid,
        "spec_bandwidth": bandwidth,
        "spec_rolloff": rolloff,
        "zcr": zcr,
        "rms": rms,
    }
    for idx in range(args.n_mfcc):
        data[f"mfcc_{idx+1:02d}"] = mfcc[idx]
    df = pd.DataFrame(data)
    out_root = Path(args.outdir)
    csv_dir = out_root / "csv"
    plot_dir = out_root / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or audio_path.stem
    df.to_csv(csv_dir / f"{prefix}_spectral_features.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length, ax=axes[0])
    axes[0].set_title("Mel Spectrogram")
    plt.colorbar(ax=axes[0], format="%+2.0f dB")
    librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=hop_length, ax=axes[1])
    axes[1].set_title("MFCC")
    plt.colorbar(ax=axes[1])
    axes[2].plot(frame_times, centroid, label="Centroid")
    axes[2].plot(frame_times, bandwidth / 10, label="Bandwidth/10")
    axes[2].plot(frame_times, rms * 1000, label="RMS*1000")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_spectral_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved spectral descriptors to {csv_dir}")


if __name__ == "__main__":
    main()
