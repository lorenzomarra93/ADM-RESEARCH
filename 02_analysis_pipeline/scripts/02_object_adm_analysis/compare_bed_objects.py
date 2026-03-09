"""Compare spatial metrics between a 7.1.2 bed analysis CSV and an ADM objects timeline CSV."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter
    _HAVE_SAVGOL = True
except Exception:
    _HAVE_SAVGOL = False

@dataclass
class AlignedSeries:
    df: pd.DataFrame
    dt: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare bed vs object spatial timelines")
    ap.add_argument("--bed", required=True, help="CSV con bed_features (time_s, FB, LR, height, spread, ...)")
    ap.add_argument("--obj", required=True, help="CSV con timeline objects (time_s, obj_cx/obj_cy...)" )
    ap.add_argument("--out", required=True, help="Cartella di output")
    ap.add_argument("--lag-max-ms", type=float, default=250.0, help="Lag massimo per cross-correlation")
    ap.add_argument("--smooth-ms", type=float, default=80.0, help="Finestra smoothing (ms); 0 per disattivare")
    ap.add_argument("--tolerance-ms", type=float, default=5.0, help="Tolleranza merge_asof (ms)")
    ap.add_argument("--epsilon", type=float, default=0.02, help="Soglia velocità per segmentazione FB")
    ap.add_argument("--min-seg-ms", type=float, default=300.0, help="Durata minima segmento (ms)")
    return ap.parse_args()


def load_csv(path: str, rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise ValueError(f"Il file {path} non contiene la colonna time_s")
    df = df.copy()
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df = df.dropna(subset=["time_s"]).sort_values("time_s")
    if rename_map:
        for key, target in rename_map.items():
            if key in df.columns:
                df[target] = df[key]
    return df


def apply_smoothing(series: pd.Series, dt: float, smooth_ms: float) -> pd.Series:
    if smooth_ms <= 0:
        return series
    window = max(3, int(round(smooth_ms / 1000.0 / dt)))
    if window % 2 == 0:
        window += 1
    if window <= 3:
        return series.rolling(window, center=True, min_periods=1).mean()
    if _HAVE_SAVGOL:
        try:
            return pd.Series(savgol_filter(series.to_numpy(), window_length=window, polyorder=min(3, window-1)), index=series.index)
        except Exception:
            pass
    return series.rolling(window, center=True, min_periods=1).mean()


def load_align(bed_path: str, obj_path: str, smooth_ms: float, tolerance_ms: float) -> AlignedSeries:
    bed = load_csv(bed_path)
    obj = load_csv(obj_path, rename_map={
        "obj_cx": "obj_cx",
        "obj_cy": "obj_cy",
        "obj_cz": "obj_cz",
        "cx": "obj_cx",
        "cy": "obj_cy",
        "cz": "obj_cz",
        "x_lr": "obj_cx",
        "y_fb": "obj_cy",
        "z_height": "obj_cz",
        "spread": "obj_spread",
        "obj_spread": "obj_spread",
    })
    required_bed = ["FB", "LR", "height", "spread"]
    required_obj = ["obj_cy", "obj_cx", "obj_cz", "obj_spread"]
    for col in required_bed:
        if col not in bed.columns:
            raise ValueError(f"Colonna {col} mancante in bed CSV")
    for col in required_obj:
        if col not in obj.columns:
            raise ValueError(f"Colonna {col} mancante in object CSV")
    tolerance = tolerance_ms / 1000.0
    bed = bed.sort_values("time_s").reset_index(drop=True)
    obj = obj.sort_values("time_s").reset_index(drop=True)
    if not bed["time_s"].is_monotonic_increasing:
        bed = bed.sort_values("time_s")
    if not obj["time_s"].is_monotonic_increasing:
        obj = obj.sort_values("time_s")
    dt = np.median(np.diff(bed["time_s"].to_numpy())) if len(bed) > 1 else 1.0
    for col in ["FB", "LR", "height", "spread"]:
        bed[col] = apply_smoothing(bed[col], dt, smooth_ms)
    for col in ["obj_cy", "obj_cx", "obj_cz", "obj_spread"]:
        obj[col] = apply_smoothing(obj[col], dt, smooth_ms)
    aligned = pd.merge_asof(
        bed,
        obj[["time_s", "obj_cx", "obj_cy", "obj_cz", "obj_spread"]],
        on="time_s",
        direction="nearest",
        tolerance=tolerance,
    ).dropna()
    if aligned.empty:
        raise RuntimeError("Allineamento vuoto. Aumentare la tolleranza o controllare i dati.")
    aligned = aligned.reset_index(drop=True)
    if len(aligned) > 1:
        dt_aligned = np.median(np.diff(aligned["time_s"].to_numpy()))
    else:
        dt_aligned = dt
    aligned.rename(columns={
        "FB": "bed_FB",
        "LR": "bed_LR",
        "height": "bed_height",
        "spread": "bed_spread",
    }, inplace=True)
    return AlignedSeries(df=aligned, dt=float(dt_aligned))


def pearson_metrics(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    from scipy.stats import pearsonr
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def error_metrics(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    diff = x - y
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae}


def estimate_best_lag(series: pd.Series, ref: pd.Series, dt: float, max_lag_ms: float) -> float:
    max_shift = int(round((max_lag_ms / 1000.0) / dt))
    if max_shift <= 0 or len(series) < 2:
        return 0.0
    x = series.to_numpy() - np.mean(series)
    y = ref.to_numpy() - np.mean(ref)
    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(series) + 1, len(series))
    mask = (lags >= -max_shift) & (lags <= max_shift)
    if not np.any(mask):
        return 0.0
    best_lag = lags[mask][np.argmax(corr[mask])]
    return float(best_lag * dt * 1000.0)


def compute_metrics(aligned: AlignedSeries, max_lag_ms: float) -> Dict[str, Dict[str, float]]:
    df = aligned.df
    metrics = {}
    pairs = {
        "FB": (df["bed_FB"], df["obj_cy"]),
        "LR": (df["bed_LR"], df["obj_cx"]),
        "height": (df["bed_height"], df["obj_cz"]),
        "spread": (df["bed_spread"], df["obj_spread"]),
    }
    for key, (a, b) in pairs.items():
        r, p = pearson_metrics(a, b)
        errs = error_metrics(a, b)
        metrics[key] = {"pearson": r, "p_value": p, **errs}
    lag_ms = estimate_best_lag(df["bed_FB"], df["obj_cy"], aligned.dt, max_lag_ms)
    metrics["FB"]["best_lag_ms"] = lag_ms
    return metrics


def segment_motion(series: np.ndarray, time_s: np.ndarray, vel_thresh: float = 0.02, min_dur_s: float = 0.3):
    v = np.gradient(series, time_s)
    moving = np.abs(v) > vel_thresh
    best = None
    start = None
    for i, active in enumerate(moving):
        if active and start is None:
            start = i
        if not active and start is not None:
            dur = time_s[i-1] - time_s[start]
            if dur >= min_dur_s and (best is None or dur > best[1] - best[0]):
                best = (time_s[start], time_s[i-1], start, i)
            start = None
    if start is not None:
        dur = time_s[-1] - time_s[start]
        if dur >= min_dur_s and (best is None or dur > best[1] - best[0]):
            best = (time_s[start], time_s[-1], start, len(time_s))
    return best


def compute_kpis(time_s: np.ndarray, fb: np.ndarray, spread: np.ndarray, idx0: int, idx1: int) -> Dict[str, float]:
    if idx1 <= idx0:
        return {
            "slope": float("nan"),
            "delta_spread": float("nan"),
            "path_len": float("nan"),
            "t_zero_cross": None,
        }
    t_seg = time_s[idx0:idx1]
    fb_seg = fb[idx0:idx1]
    spread_seg = spread[idx0:idx1]
    slope = linregress_segment(t_seg, fb_seg)
    delta_spread = float(spread_seg[-1] - spread_seg[0]) if len(spread_seg) >= 2 else float("nan")
    path_len = float(np.sum(np.abs(np.diff(fb_seg))))
    zero_cross = None
    for i in range(1, len(fb_seg)):
        if fb_seg[i-1] < 0 <= fb_seg[i] or fb_seg[i-1] > 0 >= fb_seg[i]:
            denom = fb_seg[i] - fb_seg[i-1]
            if denom != 0:
                frac = -fb_seg[i-1] / denom
                zero_cross = float(t_seg[i-1] + frac * (t_seg[i] - t_seg[i-1]))
            else:
                zero_cross = float(t_seg[i])
            break
    return {
        "slope": slope,
        "delta_spread": delta_spread,
        "path_len": path_len,
        "t_zero_cross": zero_cross,
    }


def linregress_segment(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    a, _ = np.polyfit(t, y, 1)
    return float(a)


def make_plots(aligned: AlignedSeries, metrics: Dict[str, Dict[str, float]], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    df = aligned.df
    t = df["time_s"].to_numpy()
    def plot_pair(x, y, metric_key, label_x, label_y, filename_prefix):
        plt.figure(figsize=(8, 4))
        plt.plot(t, x, label="bed")
        plt.plot(t, y, label="objects")
        plt.xlabel("Time (s)")
        plt.ylabel(label_x)
        plt.legend()
        meta = metrics.get(metric_key, {})
        r = meta.get("pearson")
        rmse = meta.get("rmse")
        lag = meta.get("best_lag_ms") if metric_key == "FB" else None
        title = f"{label_x} comparison"
        if not pd.isna(r) and not pd.isna(rmse):
            title += f" (r={r:.3f}, RMSE={rmse:.3f}"
            if lag is not None:
                title += f", lag={lag:.1f} ms"
            title += ")"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"compare_{filename_prefix}.png"), dpi=140)
        plt.close()

        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(x, y, s=8, alpha=0.7)
        min_val = np.nanmin([np.nanmin(x), np.nanmin(y)])
        max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
        if not (np.isinf(min_val) or np.isinf(max_val)):
            plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
        plt.xlabel(f"bed {label_x}")
        plt.ylabel(f"obj {label_y}")
        scatter_title = f"Scatter {label_x}"
        if not pd.isna(r) and not pd.isna(rmse):
            scatter_title += f" (r={r:.3f}, RMSE={rmse:.3f})"
        plt.title(scatter_title)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"scatter_{filename_prefix}.png"), dpi=140)
        plt.close()

        plt.figure(figsize=(8, 3.2))
        plt.plot(t, x - y)
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Residual")
        plt.title(f"Residual bed-obj {label_x}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"residual_{filename_prefix}.png"), dpi=140)
        plt.close()

    pairs = [
        ("FB", "bed_FB", "obj_cy", "FB", "cy", "FB"),
        ("LR", "bed_LR", "obj_cx", "LR", "cx", "LR"),
        ("spread", "bed_spread", "obj_spread", "spread", "spread", "spread"),
    ]
    for metric_key, bed_col, obj_col, label_x, label_y, fname in pairs:
        if bed_col in df.columns and obj_col in df.columns:
            plot_pair(df[bed_col], df[obj_col], metric_key, label_x, label_y, fname)



def main() -> None:
    args = parse_args()
    aligned = load_align(args.bed, args.obj, args.smooth_ms, args.tolerance_ms)
    metrics = compute_metrics(aligned, args.lag_max_ms)
    time_arr = aligned.df["time_s"].to_numpy()
    seg = segment_motion(
        aligned.df["bed_FB"].to_numpy(),
        time_arr,
        vel_thresh=args.epsilon,
        min_dur_s=args.min_seg_ms / 1000.0,
    )
    if seg is None:
        idx0, idx1 = 0, len(aligned.df)
        t0, t1 = float(time_arr[0]), float(time_arr[-1])
    else:
        t0, t1, idx0, idx1 = seg
        if idx1 <= idx0:
            idx0, idx1 = 0, len(aligned.df)
            t0, t1 = float(time_arr[0]), float(time_arr[-1])
    bed_base = compute_kpis(time_arr, aligned.df["bed_FB"].to_numpy(), aligned.df["bed_spread"].to_numpy(), idx0, idx1)
    obj_base = compute_kpis(time_arr, aligned.df["obj_cy"].to_numpy(), aligned.df["obj_spread"].to_numpy(), idx0, idx1)
    bed_kpis = {f"bed_{k}": v for k, v in bed_base.items()}
    obj_kpis = {f"obj_{k}": v for k, v in obj_base.items()}
    diff_kpis = {}
    for key in bed_base:
        b_val = bed_base[key]
        o_val = obj_base.get(key, float("nan"))
        if pd.isna(b_val) or pd.isna(o_val):
            diff = float("nan")
        else:
            diff = b_val - o_val
        diff_kpis[f"diff_{key}"] = diff
    df_aligned = aligned.df.copy()
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    df_aligned.to_csv(os.path.join(outdir, "compare_timeseries.csv"), index=False)
    make_plots(aligned, metrics, outdir)
    all_metrics = {
        "metrics": metrics,
        "bed_kpis": bed_kpis,
        "obj_kpis": obj_kpis,
        "diff_kpis": diff_kpis,
        "segment": {
            "start_idx": int(idx0),
            "end_idx": int(idx1),
            "start_time": float(t0),
            "end_time": float(t1),
            "duration_s": float(max(0.0, t1 - t0)),
        },
    }
    with open(os.path.join(outdir, "compare_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("=== COMPARE SUMMARY ===")
    for key, vals in metrics.items():
        print(f"{key}: r={vals.get('pearson'):.3f}, p={vals.get('p_value'):.3g}, rmse={vals.get('rmse'):.4f}, mae={vals.get('mae'):.4f}")
    print(f"FB best lag (ms): {metrics['FB'].get('best_lag_ms', 0.0):.2f}")
    print(f"Segment window: {all_metrics['segment']['start_time']:.3f}s → {all_metrics['segment']['end_time']:.3f}s")
    print(
        f"Bed slope: {bed_kpis['bed_slope']:.4f} | Obj slope: {obj_kpis['obj_slope']:.4f} | Δ = {diff_kpis['diff_slope']:.4f}"
    )
    print("========================")


if __name__ == "__main__":
    main()
