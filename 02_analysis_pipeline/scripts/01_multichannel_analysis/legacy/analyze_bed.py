"""
analyze_bed.py — Analisi energy-based di un WAV 7.1.2 (bed) per estrarre feature spaziali,
KPI e grafici, trattando la distribuzione di energia come "metadata-as-score".

USO (esempio):
  python analyze_bed.py --wav PATH/zoom_712.wav --itu --smooth sg:9:2 \
      --json-summary out/summary.json --panel out/panel.png

ARGOMENTI PRINCIPALI
  --wav PATH               : file WAV interleaved 7.1.2 (10 canali).
  --win-ms 20 --hop-ms 10  : finestra e hop in millisecondi.
  --order L,R,C,LFE,Ls,Rs,Lrs,Rrs,Ltf,Rtf  : ordine canali; default Pro Tools classico.
  --itu                    : usa vettori diffusori con angoli ITU (consigliato).
  --smooth MODE            : smoothing di cy prima della derivata. MODE ∈ {none, ma:N, sg:W:P}
                             - ma:N = moving average su N frame (es. ma:5)
                             - sg:W:P = Savitzky–Golay con window W (dispari) e polyorder P (es. sg:9:2)
  --json-summary PATH      : salva KPI e segmentazione in JSON.
  --panel PATH             : salva pannello 2×2 (traj XY, FB, LR, spread).
  --polar PATH             : salva plot polare (theta(t)) opzionale.
  --probe                  : stampa info canali (conteggio e picchi) e termina.
  --outdir out             : cartella di output per CSV/PNG (default: out)

FEATURE E KPI
  - Serie per frame: cx, cy, cz (centro), theta, FB (=cy), FB_group, LR, height, spread, cy_speed, x_lr, y_fb, z_height, zone.
  - KPI: t_zero_cross, FB_slope (pendenza su transizione), Δspread, LR_bias_mean, path_len, height_mean,
         min/max spread e relativi tempi; segmentazione Anchor/Transition/Focus.
  - Metriche aggiuntive: frazioni temporali front/center/rear, left/center/right, low/center/high e grafici di densità asse/zone.

VETTORI DIFFUSORI (ITU)
  Azimuth (°): L/R ±30, C 0, Ls/Rs ±90, Lrs/Rrs ±150, Ltf/Rtf ±30 con elevation +30.
  Conversione: x=sin(az)*cos(el), y=cos(az)*cos(el), z=sin(el). Normalizzati.
"""
import argparse
import os, json, math
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

AXIS_CENTER_EPS = 0.02  # tolerance to consider the centroid centered on an axis
try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def unit_from_az_el(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.sin(az) * math.cos(el)
    y = math.cos(az) * math.cos(el)
    z = math.sin(el)
    n = math.sqrt(x*x + y*y + z*z) or 1.0
    return np.array([x/n, y/n, z/n], dtype=float)

def build_vectors(order_labels, use_itu=True):
    if use_itu:
        ang = {
            'L':  (-30, 0), 'R':  ( 30, 0), 'C': ( 0, 0),
            'Ls': (-90, 0), 'Rs': ( 90, 0),
            'Lrs':(-150, 0), 'Rrs':(150, 0),
            'Ltf':(-30, 30), 'Rtf':( 30, 30),
            'LFE':(0, 0),
        }
        vecs = []
        dir_labels = []
        for lab in order_labels:
            if lab == 'LFE': continue
            vecs.append(unit_from_az_el(*ang[lab]))
            dir_labels.append(lab)
        return np.vstack(vecs), dir_labels
    else:
        rough = {
            'L':[-1, 1, 0], 'R':[ 1, 1, 0], 'C':[0, 1, 0],
            'Ls':[-1, 0, 0], 'Rs':[ 1, 0, 0],
            'Lrs':[-1,-1, 0], 'Rrs':[ 1,-1,0],
            'Ltf':[-1, 1, 1], 'Rtf':[ 1, 1, 1],
            'LFE':[0,0,0],
        }
        vecs = []
        dir_labels = []
        for lab in order_labels:
            if lab == 'LFE': continue
            vec = np.array(rough[lab], dtype=float)
            vec = vec / np.linalg.norm(vec)
            vecs.append(vec)
            dir_labels.append(lab)
        return np.vstack(vecs), dir_labels

def parse_smooth_mode(mode_str):
    if mode_str is None or mode_str.lower() == 'none':
        return ('none', None)
    try:
        if mode_str.startswith('ma:'):
            N = int(mode_str.split(':')[1])
            return ('ma', N)
        if mode_str.startswith('sg:'):
            _, w, p = mode_str.split(':')
            return ('sg', (int(w), int(p)))
    except Exception:
        pass
    raise ValueError("Formato --smooth non valido. Usa: none | ma:N | sg:W:P (es. sg:9:2)")

def smooth_series(x, mode):
    kind, param = mode
    if kind == 'none': return x
    if kind == 'ma':
        N = max(1, int(param))
        if N <= 1: return x
        ker = np.ones(N)/N
        return np.convolve(x, ker, mode='same')
    if kind == 'sg':
        if not _HAVE_SCIPY:
            N = param[0] if isinstance(param, tuple) else 7
            return smooth_series(x, ('ma', N))
        W, P = param
        if W % 2 == 0: W += 1
        W = max(W, P*2+1)
        try:
            return savgol_filter(x, W, P, mode='interp')
        except Exception:
            return smooth_series(x, ('ma', 7))
    return x

def window_rms(x):
    return np.sqrt(np.mean(x*x, axis=-1) + 1e-18)

def frames_iter(n_samples, win, hop):
    i = 0
    while i + win <= n_samples:
        yield i, i+win
        i += hop

def compute_features(wav_path, order_labels, use_itu, win_ms, hop_ms, outdir):
    os.makedirs(outdir, exist_ok=True)
    x, fs = sf.read(wav_path)
    if x.ndim != 2:
        raise ValueError("Atteso WAV multicanale interleaved.")
    if x.shape[1] < 10:
        raise ValueError(f"Attesi 10 canali (7.1.2). Trovati: {x.shape[1]}")
    label_to_idx = {lab: i for i, lab in enumerate(order_labels)}
    idx_all = [label_to_idx[lab] for lab in order_labels]
    X_all = x[:, idx_all].T
    dir_labels = [lab for lab in order_labels if lab != 'LFE']
    dir_idx = [label_to_idx[lab] for lab in dir_labels]
    X = x[:, dir_idx].T
    n_dir, N = X.shape
    coords, dir_labels_check = build_vectors(order_labels, use_itu=use_itu)
    assert coords.shape[0] == n_dir and len(dir_labels_check) == n_dir
    win = int(fs * win_ms / 1000.0)
    hop = int(fs * hop_ms / 1000.0)
    dt = hop / fs
    rows = []
    all_weights = []
    for s, e in frames_iter(N, win, hop):
        seg = X[:, s:e]
        E = window_rms(seg)
        w = E / (np.sum(E) + 1e-18)
        all_weights.append(w)
        c = (w[:, None] * coords).sum(axis=0)
        spread = np.sqrt((np.sum(((coords - c) ** 2), axis=1) * w).sum())
        lab2col = {lab: i for i, lab in enumerate(dir_labels)}
        left_grp  = ['L', 'Ls', 'Lrs', 'Ltf']
        right_grp = ['R', 'Rs', 'Rrs', 'Rtf']
        front_grp = ['L', 'R', 'C', 'Ltf', 'Rtf']
        rear_grp  = ['Lrs', 'Rrs']
        top_grp   = ['Ltf', 'Rtf']
        def sum_w(labels):
            return sum(w[lab2col[l]] for l in labels if l in lab2col)
        LR = sum_w(left_grp) - sum_w(right_grp)
        FB_group = sum_w(front_grp) - sum_w(rear_grp)
        height = sum_w(top_grp)
        rows.append((c[0], c[1], c[2], FB_group, LR, height, spread))
    features = np.asarray(rows)
    centroid = features[:, :3]
    FB_group = features[:, 3]
    LR = features[:, 4]
    height = features[:, 5]
    spread = features[:, 6]
    norm_weights = np.vstack(all_weights)
    times = np.arange(features.shape[0]) * dt
    df = pd.DataFrame({
        "time_s": times,
        "cx": centroid[:, 0],
        "cy": centroid[:, 1],
        "cz": centroid[:, 2],
        "theta": np.arctan2(centroid[:, 1], centroid[:, 0]),
        "FB": centroid[:, 1],
        "FB_group": FB_group,
        "LR": LR,
        "height": height,
        "spread": spread,
    })
    df["x_lr"] = df["cx"]
    df["y_fb"] = df["cy"]
    df["z_height"] = df["cz"]
    extra = {
        "fs": fs,
        "win": win,
        "hop": hop,
        "dt": dt,
        "dir_labels": dir_labels,
        "coords": coords,
        "weights": norm_weights,
    }
    return df, dt, extra

def contiguous_segments(mask, dt, min_dur_s):
    segs = []
    if mask.size == 0:
        return segs
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if not val and start is not None:
            dur = (i - start) * dt
            if dur >= min_dur_s:
                segs.append((start * dt, i * dt))
            start = None
    if start is not None:
        i = len(mask)
        dur = (i - start) * dt
        if dur >= min_dur_s:
            segs.append((start * dt, i * dt))
    return segs

def zero_cross_time(t, y):
    sign = np.sign(y)
    for i in range(1, len(y)):
        if sign[i-1] < 0 and sign[i] >= 0:
            if y[i] == y[i-1]:
                return float(t[i])
            a = (0 - y[i-1]) / (y[i] - y[i-1])
            return float(t[i-1] + a * (t[i] - t[i-1]))
    return None

def linregress(t, y):
    if len(t) < 2:
        return None
    a, b = np.polyfit(t, y, 1)
    return float(a)

def compute_kpis(df, dt, seg_params):
    cy_sm = df["cy_smooth"].to_numpy()
    t = df["time_s"].to_numpy()
    cy_speed = df["cy_speed"].to_numpy()
    spread = df["spread"].to_numpy()
    FB = df["FB"].to_numpy()
    LR = df["LR"].to_numpy()
    height = df["height"].to_numpy()
    anchor_mask = np.abs(cy_speed) < seg_params["anchor_speed_th"]
    transition_mask = np.abs(cy_speed) >= seg_params["trans_speed_th"]
    thr_spread = np.percentile(spread, seg_params["focus_percentile"])
    focus_mask = spread <= thr_spread
    anchor_segs = contiguous_segments(anchor_mask, dt, seg_params["anchor_min_dur"])
    trans_segs  = contiguous_segments(transition_mask, dt, seg_params["trans_min_dur"])
    focus_segs  = contiguous_segments(focus_mask, dt, seg_params["focus_min_dur"])
    t0_cross = zero_cross_time(t, cy_sm)
    FB_slope = None
    if trans_segs:
        t0, t1 = trans_segs[0]
        sel = (t >= t0) & (t <= t1)
        FB_slope = linregress(t[sel], FB[sel])
        spread_start = float(np.median(spread[sel][:max(1, len(spread[sel])//5)]))
        spread_end   = float(np.median(spread[sel][-max(1, len(spread[sel])//5):]))
        delta_spread = spread_end - spread_start
    else:
        FB_slope = linregress(t, FB)
        delta_spread = float(spread[-1] - spread[0])
    LR_bias_mean = float(np.mean(LR))
    path_len = float(np.sum(np.abs(cy_speed) * dt))
    height_mean = float(np.mean(height))
    spread_min = float(np.min(spread)); t_spread_min = float(t[int(np.argmin(spread))])
    spread_max = float(np.max(spread)); t_spread_max = float(t[int(np.argmax(spread))])
    summary = {
        "t_zero_cross": t0_cross,
        "FB_slope": FB_slope,
        "delta_spread": delta_spread,
        "LR_bias_mean": LR_bias_mean,
        "path_len": path_len,
        "height_mean": height_mean,
        "spread_min": spread_min,
        "t_spread_min": t_spread_min,
        "spread_max": spread_max,
        "t_spread_max": t_spread_max,
        "segments": {
            "anchor": anchor_segs,
            "transition": trans_segs,
            "focus": focus_segs,
        },
        "focus_spread_threshold": float(thr_spread),
    }
    return summary


def axis_bucket(value, neg_label, pos_label, eps=AXIS_CENTER_EPS):
    if value <= -eps:
        return neg_label
    if value >= eps:
        return pos_label
    return "center"


def classify_zone(row):
    fb = axis_bucket(row["y_fb"], "rear", "front")
    lr = axis_bucket(row["x_lr"], "left", "right")
    height = "high" if row["z_height"] >= 0.3 else ("low" if row["z_height"] <= -0.3 else "mid")
    return f"{fb}_{lr}_{height}"


def axis_time_fraction(values, dt, neg_label, pos_label, eps=AXIS_CENTER_EPS):
    total_time = len(values) * dt
    if total_time <= 0:
        return {neg_label: 0.0, "center": 0.0, pos_label: 0.0}
    neg = float(np.sum(values < -eps)) * dt
    pos = float(np.sum(values > eps)) * dt
    center = total_time - neg - pos
    return {
        neg_label: neg / total_time,
        "center": center / total_time,
        pos_label: pos / total_time,
    }

def save_single_plots(df, outdir, prefix=""):
    os.makedirs(outdir, exist_ok=True)
    t = df["time_s"].to_numpy()
    plt.figure()
    plt.plot(df["cx"], df["cy_smooth"])
    plt.axis('equal')
    plt.xlabel("cx (LR)")
    plt.ylabel("cy (FB)")
    plt.title("Traiettoria XY")
    plt.savefig(os.path.join(outdir, f"{prefix}traj_xy.png"), dpi=140)
    plt.close()
    plt.figure()
    plt.plot(t, df["FB"])
    plt.axhline(0, ls="--", lw=0.8)
    plt.title("Front–Back (FB)")
    plt.xlabel("Time (s)"); plt.ylabel("FB")
    plt.savefig(os.path.join(outdir, f"{prefix}FB_index.png"), dpi=140)
    plt.close()
    plt.figure()
    plt.plot(t, df["LR"])
    plt.axhline(0, ls="--", lw=0.8)
    plt.title("Left–Right (LR)")
    plt.xlabel("Time (s)"); plt.ylabel("LR")
    plt.savefig(os.path.join(outdir, f"{prefix}LR_index.png"), dpi=140)
    plt.close()
    plt.figure()
    plt.plot(t, df["spread"])
    plt.title("Spatial Spread")
    plt.xlabel("Time (s)"); plt.ylabel("Spread")
    plt.savefig(os.path.join(outdir, f"{prefix}spread.png"), dpi=140)
    plt.close()
    plt.figure()
    plt.plot(t, df["height"])
    plt.title("Height Ratio")
    plt.xlabel("Time (s)"); plt.ylabel("Height")
    plt.savefig(os.path.join(outdir, f"{prefix}height.png"), dpi=140)
    plt.close()
    plt.figure()
    plt.plot(t, df["cy_speed"])
    plt.title("cy Speed")
    plt.xlabel("Time (s)"); plt.ylabel("cy_speed")
    plt.savefig(os.path.join(outdir, f"{prefix}cy_speed.png"), dpi=140)
    plt.close()
    fb_counts = [
        int((df["y_fb"] > AXIS_CENTER_EPS).sum()),
        int((np.abs(df["y_fb"]) <= AXIS_CENTER_EPS).sum()),
        int((df["y_fb"] < -AXIS_CENTER_EPS).sum()),
    ]
    lr_counts = [
        int((df["x_lr"] < -AXIS_CENTER_EPS).sum()),
        int((np.abs(df["x_lr"]) <= AXIS_CENTER_EPS).sum()),
        int((df["x_lr"] > AXIS_CENTER_EPS).sum()),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(["front", "center", "rear"], fb_counts, color="#cc8844")
    ax[0].set_title("Front vs Rear occupancy")
    ax[0].set_ylabel("Frame count")
    ax[1].bar(["left", "center", "right"], lr_counts, color="#4488cc")
    ax[1].set_title("Left vs Right occupancy")
    for axis in ax:
        axis.set_xlabel("Zone")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}bed_axis_density.png"), dpi=140)
    plt.close(fig)
    df_zone = df
    if "zone" not in df_zone.columns:
        df_zone = df_zone.copy()
        df_zone["zone"] = df_zone.apply(classify_zone, axis=1)
    zone_counts = df_zone["zone"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    plt.bar(zone_counts.index, zone_counts.values, color="#cc6633")
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Frame count")
    plt.title("Zone occupancy")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}bed_zone_density.png"), dpi=140)
    plt.close()

def save_panel(df, path):
    t = df["time_s"].to_numpy()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    ax[0,0].plot(df["cx"], df["cy_smooth"])
    ax[0,0].set_title("Traiettoria XY"); ax[0,0].set_xlabel("cx"); ax[0,0].set_ylabel("cy"); ax[0,0].axis('equal')
    ax[0,1].plot(t, df["FB"]); ax[0,1].axhline(0, ls="--", lw=0.8)
    ax[0,1].set_title("Front–Back"); ax[0,1].set_xlabel("s")
    ax[1,0].plot(t, df["LR"]); ax[1,0].axhline(0, ls="--", lw=0.8)
    ax[1,0].set_title("Left–Right"); ax[1,0].set_xlabel("s")
    ax[1,1].plot(t, df["spread"])
    ax[1,1].set_title("Spread"); ax[1,1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_polar(df, path):
    theta = df["theta"].to_numpy()
    t = df["time_s"].to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(theta, t)
    ax.set_title("Tempo–Angolo (theta(t))")
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Analyze 7.1.2 bed WAV to spatial features/KPIs.")
    ap.add_argument("--wav", required=True, help="Percorso WAV 7.1.2 interleaved (10 canali).")
    ap.add_argument("--win-ms", type=int, default=20)
    ap.add_argument("--hop-ms", type=int, default=10)
    ap.add_argument("--order", default="L,R,C,LFE,Ls,Rs,Lrs,Rrs,Ltf,Rtf",
                    help="Ordine canali (virgole). Default Pro Tools classico.")
    ap.add_argument("--itu", action="store_true", help="Usa vettori diffusori ITU.")
    ap.add_argument("--smooth", default="ma:5", help="Smoothing: none | ma:N | sg:W:P")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--json-summary", default=None)
    ap.add_argument("--panel", default=None, help="Salva pannello 2×2 a questo path.")
    ap.add_argument("--polar", default=None, help="Salva plot polare tempo–angolo a questo path.")
    ap.add_argument("--probe", action="store_true", help="Stampa info canali e termina.")
    args = ap.parse_args()
    order_labels = [s.strip() for s in args.order.split(",")]
    if len(order_labels) != 10:
        raise ValueError("L'ordine deve avere 10 etichette (7.1.2).")
    if args.probe:
        x, fs = sf.read(args.wav)
        peaks = (np.abs(x).max(axis=0)).tolist()
        print(f"[PROBE] fs={fs} Hz, channels={x.shape[1]}, peaks={peaks}")
        return
    df, dt, extra = compute_features(
        args.wav, order_labels, args.itu, args.win_ms, args.hop_ms, args.outdir
    )
    sm_mode = parse_smooth_mode(args.smooth)
    df["cy_smooth"] = smooth_series(df["cy"].to_numpy(), sm_mode)
    df["cy_speed"]  = np.gradient(df["cy_smooth"].to_numpy(), dt)
    df["zone"] = df.apply(classify_zone, axis=1)
    total_time = len(df) * dt
    zone_fraction = {}
    if total_time > 0:
        zone_fraction = (df["zone"].value_counts() * dt / total_time).to_dict()
    fb_fraction = axis_time_fraction(df["y_fb"].to_numpy(), dt, "rear", "front")
    lr_fraction = axis_time_fraction(df["x_lr"].to_numpy(), dt, "left", "right")
    height_fraction = axis_time_fraction(df["z_height"].to_numpy(), dt, "low", "high")
    seg_params = dict(
        anchor_speed_th=0.05,
        trans_speed_th=0.20,
        anchor_min_dur=0.30,
        trans_min_dur=0.20,
        focus_percentile=20,
        focus_min_dur=0.30,
    )
    summary = compute_kpis(df, dt, seg_params)
    summary.update({
        "zone_density_time_fraction": zone_fraction,
        "front_back_time_fraction": fb_fraction,
        "left_right_time_fraction": lr_fraction,
        "height_time_fraction": height_fraction,
    })
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "bed_features.csv")
    df.to_csv(csv_path, index=False)
    save_single_plots(df, args.outdir)
    if args.panel:
        save_panel(df, args.panel)
    if args.polar:
        save_polar(df, args.polar)
    if args.json_summary:
        meta = {
            "file": os.path.abspath(args.wav),
            "fs": extra["fs"],
            "win_ms": args.win_ms,
            "hop_ms": args.hop_ms,
            "order": order_labels,
            "itu": bool(args.itu),
            "smooth": args.smooth,
        }
        meta.update(summary)
        with open(args.json_summary, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    print("=== SUMMARY ===")
    print(f"CSV      : {csv_path}")
    if args.panel: print(f"PANEL    : {args.panel}")
    print(f"Frames   : {len(df)}  | dt={dt:.4f}s  | smooth={args.smooth} | ITU={args.itu}")
    print(f"ZeroCross: {summary['t_zero_cross']}")
    print(f"FB_slope : {summary['FB_slope']:.4f}  | Δspread: {summary['delta_spread']:.4f}")
    print(f"LR_bias  : {summary['LR_bias_mean']:.4f}  | path_len: {summary['path_len']:.4f}")
    print(f"Heightµ  : {summary['height_mean']:.4f}  | spread[min,max]=({summary['spread_min']:.3f}, {summary['spread_max']:.3f})")
    fb_frac = summary.get('front_back_time_fraction', {})
    lr_frac = summary.get('left_right_time_fraction', {})
    height_frac = summary.get('height_time_fraction', {})
    print(f"Front/Center/Rear: {fb_frac}")
    print(f"Left/Center/Right: {lr_frac}")
    print(f"Height zones: {height_frac}")
    print(f"Segments : anchor={summary['segments']['anchor']}  transition={summary['segments']['transition']}  focus={summary['segments']['focus']}")
    print("================")

if __name__ == "__main__":
    main()
