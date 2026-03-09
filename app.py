import io
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Spatial Mix Score", layout="wide")


EXPECTED_COLUMNS = [
    "time",
    "front_ratio",
    "surround_ratio",
    "center_ratio",
    "lfe_ratio",
    "rms_total",
]


@st.cache_data
def load_csv_content(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV from raw bytes and sort by time if present."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "time" in df.columns:
        df = df.sort_values("time")
    return df.reset_index(drop=True)


def ensure_columns(df: pd.DataFrame, label: str) -> None:
    """Warn the user if mandatory columns are missing."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        st.warning(
            f"[{label}] Mancano le colonne richieste: {', '.join(missing)}.\n"
            "Alcuni grafici potrebbero non essere disponibili."
        )


def add_spread_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with spread_index column."""
    result = df.copy()
    eps = 1e-6
    if {"surround_ratio", "front_ratio"}.issubset(result.columns):
        result["spread_index"] = result["surround_ratio"] / (result["front_ratio"] + eps)
    else:
        result["spread_index"] = np.nan
    return result


def filter_by_time(df: pd.DataFrame, time_min: float, time_max: float) -> pd.DataFrame:
    """Subset df in the [time_min, time_max] range."""
    if "time" not in df.columns:
        return df
    mask = (df["time"] >= time_min) & (df["time"] <= time_max)
    return df.loc[mask]


def generate_demo_data(duration: float = 120.0, fps: int = 10) -> pd.DataFrame:
    """Create a synthetic dataset useful when no CSV is available."""
    t = np.linspace(0, duration, int(duration * fps))
    front = 0.5 + 0.3 * np.sin(2 * np.pi * t / 45)
    surround = 0.3 + 0.25 * np.sin(2 * np.pi * t / 30 + np.pi / 4)
    center = np.clip(1.0 - (front + surround), 0.05, 0.4)
    normalize = front + surround + center
    front_ratio = front / normalize
    surround_ratio = surround / normalize
    center_ratio = center / normalize
    lfe_ratio = 0.1 + 0.05 * np.sin(2 * np.pi * t / 60 + np.pi / 3)
    rms_total = 0.4 + 0.3 * np.sin(2 * np.pi * t / 20) + 0.1 * np.random.rand(t.size)
    return pd.DataFrame(
        {
            "time": t,
            "front_ratio": front_ratio,
            "surround_ratio": surround_ratio,
            "center_ratio": center_ratio,
            "lfe_ratio": lfe_ratio,
            "rms_total": rms_total,
        }
    )


def read_uploaded_files(files) -> Dict[str, pd.DataFrame]:
    """Convert uploaded CSV files into dataframes keyed by filename."""
    datasets: Dict[str, pd.DataFrame] = {}
    for uploaded in files:
        try:
            data = load_csv_content(uploaded.getvalue())
            ensure_columns(data, uploaded.name)
            datasets[uploaded.name] = data
        except Exception as exc:
            st.error(f"Errore durante il caricamento di {uploaded.name}: {exc}")
    return datasets


def render_line_chart(df: pd.DataFrame) -> None:
    y_cols = [col for col in ["front_ratio", "surround_ratio", "center_ratio"] if col in df.columns]
    if not y_cols:
        st.info("Colonne front_ratio / surround_ratio / center_ratio non trovate nel dataset selezionato.")
        return
    fig = px.line(
        df,
        x="time",
        y=y_cols,
        labels={"time": "Tempo (s)", "value": "Ratio", "variable": "Indice"},
        title="Distribuzione front / surround / center nel tempo",
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)


def render_3d_chart(df: pd.DataFrame) -> None:
    if not {"spread_index", "rms_total"}.issubset(df.columns):
        st.info("Non posso calcolare la partitura 3D: mancano 'spread_index' o 'rms_total'.")
        return
    color_col = "center_ratio" if "center_ratio" in df.columns else None
    fig = px.line_3d(
        df,
        x="time",
        y="spread_index",
        z="rms_total",
        color=color_col,
        labels={
            "time": "Tempo (s)",
            "spread_index": "Indice di spread (surround/front)",
            "rms_total": "RMS totale",
            "center_ratio": "Center ratio",
        },
        title="Evoluzione congiunta di spread immersivo e intensità",
    )
    st.plotly_chart(fig, use_container_width=True)


def build_animated_3d_figure(
    df: pd.DataFrame,
    max_points: int = 2000,
    max_frames: int = 200,
) -> go.Figure:
    """Create an animated 3D line figure similar to Plotly's Scatter3d example."""
    if len(df) > max_points:
        stride = max(len(df) // max_points, 1)
        anim_df = df.iloc[::stride].reset_index(drop=True)
    else:
        anim_df = df.reset_index(drop=True)
    frame_step = max(len(anim_df) // max_frames, 1)
    has_center = "center_ratio" in anim_df.columns
    line_kwargs = dict(width=5, color="#1f77b4")
    cmin = float(anim_df["center_ratio"].min()) if has_center else None
    cmax = float(anim_df["center_ratio"].max()) if has_center else None

    def make_trace(seg: pd.DataFrame, show_scale: bool) -> go.Scatter3d:
        marker_kwargs = dict(size=3, opacity=0.85)
        if has_center:
            marker_kwargs.update(
                color=seg["center_ratio"],
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(title="Center ratio"),
                showscale=show_scale,
            )
        return go.Scatter3d(
            x=seg["time"],
            y=seg["spread_index"],
            z=seg["rms_total"],
            mode="lines+markers",
            line=line_kwargs,
            marker=marker_kwargs,
            name="Traiettoria" if show_scale else None,
            showlegend=False,
        )

    context_trace = go.Scatter3d(
        x=anim_df["time"],
        y=anim_df["spread_index"],
        z=anim_df["rms_total"],
        mode="lines",
        line=dict(color="rgba(150,150,150,0.3)", width=2),
        name="Traiettoria globale",
        showlegend=False,
    )
    init_count = min(10, len(anim_df))
    base_trace = make_trace(anim_df.iloc[:init_count], True)
    frames = []
    for idx in range(2, len(anim_df) + 1, frame_step):
        seg = anim_df.iloc[:idx]
        frame_trace = make_trace(seg, False)
        frames.append(
            go.Frame(
                data=[frame_trace],
                name=f"{seg['time'].iloc[-1]:.2f}s",
            )
        )
    fig = go.Figure(
        data=[context_trace, base_trace],
        layout=go.Layout(
            scene=dict(
                xaxis_title="Tempo (s)",
                yaxis_title="Spread index",
                zaxis_title="RMS totale",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                            label=frame.name,
                        )
                        for frame in frames
                    ],
                    currentvalue=dict(prefix="Tempo: "),
                )
            ],
        ),
        frames=frames,
    )
    return fig


def render_scatter(df: pd.DataFrame) -> None:
    if not {"surround_ratio", "rms_total"}.issubset(df.columns):
        st.info("Scatter non disponibile: mancano 'surround_ratio' o 'rms_total'.")
        return
    color_col = "time" if "time" in df.columns else None
    fig = px.scatter(
        df,
        x="surround_ratio",
        y="rms_total",
        color=color_col,
        labels={
            "surround_ratio": "Surround ratio",
            "rms_total": "RMS totale",
            "time": "Tempo (s)",
        },
        title="Relazione surround_ratio vs intensità (RMS)",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)


def render_animated_3d_chart(df: pd.DataFrame) -> None:
    if not {"spread_index", "rms_total"}.issubset(df.columns):
        st.info("Animazione 3D non disponibile: mancano 'spread_index' o 'rms_total'.")
        return
    with st.spinner("Costruisco l'animazione 3D..."):
        fig = build_animated_3d_figure(df)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Partitura spaziale del mix 7.1")
    st.write(
        "Visualizza come un mix 7.1 distribuisce energia fra front, surround e center "
        "nel tempo, con metriche quantitative utili per l'analisi del surround cinematografico."
    )

    st.sidebar.header("Impostazioni")
    uploaded_files = st.sidebar.file_uploader(
        "Carica uno o più CSV dell'analisi del mix 7.1", type=["csv"], accept_multiple_files=True
    )
    datasets = read_uploaded_files(uploaded_files) if uploaded_files else {}

    st.sidebar.subheader("CSV locali")
    default_dir = Path.cwd() / "out_tp_mix03"
    local_dir_input = st.sidebar.text_input("Cartella da scandire", value=str(default_dir))
    local_choice = None
    if local_dir_input.strip():
        local_dir = Path(local_dir_input).expanduser()
        if local_dir.is_dir():
            csv_paths = sorted(local_dir.glob("*.csv"))
            if csv_paths:
                labels = [p.name for p in csv_paths]
                local_label = st.sidebar.selectbox("Seleziona un CSV locale", ["(nessuno)"] + labels)
                if local_label != "(nessuno)":
                    local_choice = csv_paths[labels.index(local_label)]
                    try:
                        data = load_csv_content(local_choice.read_bytes())
                        ensure_columns(data, local_choice.name)
                        datasets[f"locale: {local_choice.name}"] = data
                    except Exception as exc:
                        st.error(f"Impossibile leggere {local_choice}: {exc}")
            else:
                st.sidebar.info(f"Nessun CSV trovato in {local_dir}")
        else:
            st.sidebar.info("La cartella specificata non esiste.")

    demo_requested = st.sidebar.checkbox("Aggiungi dataset demo sintetico", value=not datasets)
    if demo_requested:
        datasets["Esempio sintetico"] = generate_demo_data()

    if not datasets:
        st.sidebar.info("Carica almeno un CSV o abilita il dataset demo per iniziare.")
        st.stop()

    dataset_name = st.sidebar.selectbox("Scegli il dataset da esplorare", list(datasets.keys()))
    df = datasets[dataset_name]

    if "time" not in df.columns:
        st.error("Il dataset selezionato non contiene la colonna 'time'. Impossibile proseguire.")
        st.stop()

    df = add_spread_index(df)

    t_min = float(df["time"].min())
    t_max = float(df["time"].max())
    if t_min == t_max:
        time_range = (t_min, t_max)
    else:
        step = max((t_max - t_min) / 100.0, 0.1)
        time_range = st.sidebar.slider(
            "Intervallo temporale da visualizzare (s)",
            min_value=t_min,
            max_value=t_max,
            value=(t_min, t_max),
            step=step,
        )

    df_filtered = filter_by_time(df, time_range[0], time_range[1])
    if df_filtered.empty:
        st.warning("Nessun dato disponibile nell'intervallo temporale selezionato.")
        st.stop()

    st.subheader("Andamento di front, surround e center nel tempo")
    render_line_chart(df_filtered)

    st.subheader("Partitura 3D: spread immersivo vs intensità sonora")
    render_3d_chart(df_filtered)
    if st.checkbox("Mostra animazione 3D (beta)", value=False):
        st.caption("Animazione Plotly che riproduce la traiettoria nel tempo; potrebbe essere pesante su dataset molto lunghi.")
        render_animated_3d_chart(df_filtered)

    st.subheader("Relazione tra surround_ratio e intensità sonora")
    render_scatter(df_filtered)

    st.markdown(
        """
        **Note analitiche**

        - La partitura 2D aiuta a individuare blocchi narrativi più frontali rispetto a sezioni con surround esteso.
        - La traiettoria 3D (tempo–spread–RMS) mette in luce se l'intensità sonora è accoppiata a mix immersivi.
        - Lo scatter surround vs RMS supporta verifiche statistiche: cluster inclinati suggeriscono correlazioni
          tra uso del surround e dinamica; pattern sparsi indicano scelte più episodiche.
        """
    )


if __name__ == "__main__":
    main()
