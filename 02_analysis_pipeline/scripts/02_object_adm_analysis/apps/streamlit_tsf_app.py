"""TSF Timeline Viewer (Streamlit) - Versione Integrata
Carica direttamente file WAV ADM oppure CSV objects_timeline.csv per visualizzazione interattiva.
Se viene caricato un WAV ADM, l'analisi viene eseguita automaticamente.
Visualizza:
- Timeline oggetti attivi + front/rear
- Statistiche front vs rear (occupazione + t-test speed)
- Serie motion (speed, y_fb, z_height) per oggetto
- MFCC heatmap (se presenti)
- Correlazioni semplici timbro↔spazio
- Spettrogramma canale oggetto (subset temporale)
Avvio:
    streamlit run streamlit_tsf_app.py
"""
import io
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

try:
    import librosa, librosa.display
    import soundfile as sf
    from sklearn.preprocessing import MinMaxScaler
    from scipy.spatial.distance import cdist
    from scipy.stats import pearsonr
except Exception:
    librosa = None
    sf = None

# Import delle funzioni di analisi ADM
try:
    from pathlib import Path
    import sys
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    # Import delle funzioni dall'altro script
    import adm_motion_analysis as adm_analyzer
except Exception as e:
    st.error(f"Errore import modulo ADM: {e}")
    adm_analyzer = None

AXIS_CENTER_EPS = 0.02
st.set_page_config(page_title="TSF Timeline Viewer", layout="wide")
st.title("TSF Timeline / Front–Back / Spectrogram Viewer")

@st.cache_data
def analyze_adm_file(wav_path, hop_ms=10.0, motion_threshold=0.01):
    """Esegue l'analisi ADM su un file WAV e restituisce il DataFrame timeline"""
    try:
        # Carica i metadati ADM
        root = adm_analyzer.load_adm_xml(wav_path)
        
        # Estrae le entries
        default_duration = hop_ms / 1000.0
        entries = adm_analyzer.collect_block_entries(root, default_duration)
        
        if not entries:
            return None, "Nessun oggetto ADM trovato nel file"
        
        # Converte in DataFrame
        df_blocks = pd.DataFrame([{
            'object_id': entry.object_id,
            'object_name': entry.object_name,
            'track_uid': entry.track_uid,
            'channel_format': entry.channel_format,
            'time_start': entry.time_start,
            'time_end': entry.time_end,
            'x': entry.x,
            'y': entry.y,
            'z': entry.z,
            'gain_db': entry.gain_db,
            'spread': entry.spread,
            'width': entry.width,
            'height': entry.height,
            'depth': entry.depth,
        } for entry in entries])
        
        # Costruisce timeline
        hop_s = hop_ms / 1000.0
        timeline = adm_analyzer.build_timeline(df_blocks, hop_s)
        
        if timeline.empty:
            return None, "Timeline vuota dopo il processing"
        
        return timeline, None
        
    except Exception as e:
        return None, f"Errore durante l'analisi: {str(e)}"

with st.sidebar:
    st.header("Input dati")
    
    # Opzioni input
    input_mode = st.radio("Modalità input:", ["CSV Timeline", "WAV ADM"])
    
    # Inizializziamo le variabili
    tl_file = None
    wav_file = None
    adm_file = None
    timeline = None
    
    if input_mode == "CSV Timeline":
        tl_file = st.file_uploader("Timeline CSV (objects_timeline.csv)", type=["csv"])
        wav_file = st.file_uploader("Audio multicanale (opzionale)", type=["wav"])
        
        if tl_file is not None:
            timeline = pd.read_csv(tl_file)
            required_cols = {"time_s", "object_id", "y_fb", "z_height"}
            missing = required_cols - set(timeline.columns)
            if missing:
                st.error(f"CSV non valido, mancano colonne: {missing}")
                st.stop()
    
    else:  # WAV ADM mode
        adm_file = st.file_uploader("File WAV ADM", type=["wav"])
        timeline = None
        
        if adm_file is not None:
            if adm_analyzer is None:
                st.error("Modulo di analisi ADM non disponibile")
                st.stop()
            
            # Parametri analisi
            st.subheader("Parametri Analisi")
            hop_ms = st.number_input("Hop size (ms)", 5.0, 100.0, 10.0, step=5.0)
            motion_threshold = st.number_input("Soglia movimento", 0.001, 0.1, 0.01, step=0.001)
            
            # Salva file temporaneo e analizza
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(adm_file.read())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner('Analizzando file ADM...'):
                    timeline, error = analyze_adm_file(tmp_path, hop_ms, motion_threshold)
                    
                if error:
                    st.error(f"Errore analisi: {error}")
                    st.stop()
                elif timeline is None or timeline.empty:
                    st.error("Nessun dato estratto dal file ADM")
                    st.stop()
                else:
                    st.success(f"✅ Analisi completata! {len(timeline)} frame, {timeline['object_id'].nunique()} oggetti")
                    
                    # Opzione download del CSV generato
                    csv_buffer = io.StringIO()
                    timeline.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Scarica CSV Timeline",
                        data=csv_buffer.getvalue(),
                        file_name=f"timeline_{adm_file.name}.csv",
                        mime="text/csv"
                    )
            finally:
                # Cleanup file temporaneo
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # Parametri visualizzazione (comuni)
    st.subheader("Parametri Visualizzazione")
    sr_target = st.number_input("Resample SR (0=none)", 0, 96000, 22050, step=1000)
    n_fft = st.number_input("n_fft", 256, 8192, 2048, step=256)
    hop_length = st.number_input("hop_length", 32, 4096, 512, step=32)
    show_mfcc = st.checkbox("Mostra MFCC heatmap", True)
    show_corr = st.checkbox("Mostra correlazioni", True)
    show_tsf = st.checkbox("🎵 Analisi TSF Avanzata", True)

# Controllo che abbiamo i dati per procedere
if timeline is None:
    st.info(f"Carica un {'CSV timeline' if input_mode == 'CSV Timeline' else 'file WAV ADM'} per iniziare.")
    st.stop()

# Verifica colonne richieste
required_cols = {"time_s", "object_id", "y_fb", "z_height"}
missing = required_cols - set(timeline.columns)
if missing:
    st.error(f"Dati non validi, mancano colonne: {missing}")
    st.stop()

# Analisi spettrale automatica se abbiamo un file WAV e non ci sono già features spettrali
mfcc_cols = [col for col in timeline.columns if col.startswith('mfcc_')]
has_spectral = len(mfcc_cols) > 0 or 'spec_centroid' in timeline.columns

# Debug info
st.write(f"Debug: has_spectral = {has_spectral}, wav_file = {wav_file is not None}, input_mode = {input_mode}")

# Genera analisi spettrale se necessario
if wav_file is not None and not has_spectral:
    st.info("🔍 Analisi spettrale in corso...")
    
    if librosa is None or sf is None:
        st.error("❌ Librosa o soundfile non disponibili. Installa con: pip install librosa soundfile")
    else:
        try:
            # Genera l'analisi spettrale
            y, sr = sf.read(wav_file, always_2d=True, dtype='float32')
            if sr_target > 0 and sr != sr_target:
                y = librosa.resample(y[:, 0], orig_sr=sr, target_sr=sr_target)
                sr = sr_target
            else:
                y = y[:, 0]  # Usa solo il primo canale
            
            # Parametri analisi
            hop_length = int(sr * 0.01)  # 10ms hop
            n_fft = 2048
            
            # Estrai features spettrali
            mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
            centroid = librosa.feature.spectral_centroid(y, sr=sr, hop_length=hop_length)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr, hop_length=hop_length)[0]
            rolloff = librosa.feature.spectral_rolloff(y, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            rms = librosa.feature.rms(y, hop_length=hop_length)[0]
            
            # Converti frame in tempo
            frame_times = librosa.frames_to_time(range(len(centroid)), sr=sr, hop_length=hop_length)
            
            # Allinea con timeline ADM
            timeline_times = timeline['time_s'].values
            
            # Interpola features su timeline ADM
            centroid_aligned = np.interp(timeline_times, frame_times, centroid)
            bandwidth_aligned = np.interp(timeline_times, frame_times, bandwidth)
            rolloff_aligned = np.interp(timeline_times, frame_times, rolloff)
            zcr_aligned = np.interp(timeline_times, frame_times, zcr)
            rms_aligned = np.interp(timeline_times, frame_times, rms)
            
            # Interpola MFCC
            mfcc_aligned = np.zeros((len(timeline_times), 13))
            for i in range(13):
                mfcc_aligned[:, i] = np.interp(timeline_times, frame_times, mfcc[i])
            
            # Aggiungi features al timeline
            timeline['spec_centroid'] = centroid_aligned
            timeline['spec_bandwidth'] = bandwidth_aligned
            timeline['spec_rolloff'] = rolloff_aligned
            timeline['zcr'] = zcr_aligned
            timeline['rms'] = rms_aligned
            
            # Aggiungi MFCC
            for i in range(13):
                timeline[f'mfcc_{i+1:02d}'] = mfcc_aligned[:, i]
            
            has_spectral = True
            st.success("✅ Analisi spettrale completata!")
            
        except Exception as e:
            st.error(f"❌ Errore nell'analisi spettrale: {e}")
            has_spectral = False

# Active objects over time
active_counts = timeline.groupby("time_s")['object_id'].nunique()
front_mask = timeline['y_fb'] > AXIS_CENTER_EPS
rear_mask = timeline['y_fb'] < -AXIS_CENTER_EPS
front_counts = timeline[front_mask].groupby('time_s')['object_id'].nunique().reindex(active_counts.index, fill_value=0)
rear_counts = timeline[rear_mask].groupby('time_s')['object_id'].nunique().reindex(active_counts.index, fill_value=0)

st.subheader("Timeline")
fig, ax = plt.subplots(figsize=(9,3))
ax.plot(active_counts.index, active_counts.values, label='Active', color='#1f77b4')
ax.plot(front_counts.index, front_counts.values, label='Front', color='#ff7f0e')
ax.plot(rear_counts.index, rear_counts.values, label='Rear', color='#2ca02c')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Objects'); ax.legend(loc='upper right', ncol=3)
fig.tight_layout()
st.pyplot(fig)

frames_total = len(active_counts)
front_time = (front_counts > 0).sum() / frames_total if frames_total else 0
rear_time = (rear_counts > 0).sum() / frames_total if frames_total else 0
st.markdown(f"**Occupazione temporale:** Front={front_time:.2%} • Rear={rear_time:.2%} • Center={(1-front_time-rear_time):.2%}")

# Oggetto selezionato
obj_ids = timeline['object_id'].unique().tolist()
sel_obj = st.selectbox("Oggetto", obj_ids)
obj_df = timeline[timeline['object_id'] == sel_obj].sort_values('time_s')
if obj_df.empty:
    st.warning("Oggetto senza dati.")
    st.stop()

# Analisi spettrale per oggetto selezionato
spectral_data = None
obj_features = None

if wav_file is not None and librosa is not None:
    with st.spinner(f"🎵 Analizzando spettro dell'oggetto {sel_obj}..."):
        spectral_data, obj_features = extract_object_spectral_features(wav_file, timeline, sel_obj, sr_target)
    
    if spectral_data is not None:
        st.success(f"✅ Analisi spettrale completata per oggetto {sel_obj}")
        
        # Mostra spettrogramma MEL
        st.subheader("🎼 MEL Spectrogram")
        plot_mel_spectrogram(spectral_data, sel_obj)
        
        # Mostra features spettrali vs movimento spaziale
        if obj_features is not None and not obj_features.empty:
            st.subheader("🎯 Correlazioni Spettro-Spaziali")
            plot_spectral_features_timeline(obj_features, obj_df)
            
            # Aggiorna obj_df con features spettrali per le correlazioni
            # Interpola features su timeline spaziale per correlazioni
            for col in obj_features.columns:
                if col != 'time_s' and col not in obj_df.columns:
                    try:
                        interpolated = np.interp(obj_df['time_s'], obj_features['time_s'], obj_features[col])
                        obj_df[col] = interpolated
                    except:
                        pass
    else:
        st.info("ℹ️ Carica un file WAV per visualizzare il MEL Spectrogram")

st.subheader("Motion & Spatial Series (oggetto)")
fig2, ax2 = plt.subplots(3,1,figsize=(8,6), sharex=True)
ax2[0].plot(obj_df['time_s'], obj_df.get('speed', 0), color='#1f77b4'); ax2[0].set_ylabel('Speed')
ax2[1].plot(obj_df['time_s'], obj_df['y_fb'], color='#d62728'); ax2[1].axhline(AXIS_CENTER_EPS, ls='--', c='gray', lw=0.6); ax2[1].axhline(-AXIS_CENTER_EPS, ls='--', c='gray', lw=0.6); ax2[1].set_ylabel('Y FB')
ax2[2].plot(obj_df['time_s'], obj_df['z_height'], color='#2ca02c'); ax2[2].set_ylabel('Z Height'); ax2[2].set_xlabel('Time (s)')
fig2.tight_layout()
st.pyplot(fig2)

# T-test front vs rear speed
front_speed = obj_df.loc[obj_df['y_fb'] > AXIS_CENTER_EPS, 'speed'].dropna()
rear_speed = obj_df.loc[obj_df['y_fb'] < -AXIS_CENTER_EPS, 'speed'].dropna()
if len(front_speed) > 3 and len(rear_speed) > 3:
    tval, pval = ttest_ind(front_speed, rear_speed, equal_var=False)
    st.caption(f"Speed front vs rear (t-test): t={tval:.2f}, p={pval:.3g}, n_front={len(front_speed)}, n_rear={len(rear_speed)}")

# MFCC heatmap
mfcc_cols = [c for c in obj_df.columns if c.startswith('mfcc_')]
if show_mfcc and mfcc_cols:
    mfcc_mat = obj_df[mfcc_cols].to_numpy().T
    fig3, ax3 = plt.subplots(figsize=(8,3))
    im = ax3.imshow(mfcc_mat, aspect='auto', origin='lower', extent=[obj_df['time_s'].min(), obj_df['time_s'].max(), 1, len(mfcc_cols)], cmap='magma')
    ax3.set_title('MFCC Heatmap'); ax3.set_ylabel('Coeff'); ax3.set_xlabel('Time (s)'); fig3.colorbar(im, ax=ax3)
    fig3.tight_layout(); st.pyplot(fig3)
elif show_mfcc:
    st.info('MFCC non presenti per questo oggetto.')

# Correlazioni
if show_corr:
    corr_pairs = [('spec_centroid','z_height'), ('spec_bandwidth','spread'), ('rms','speed')]
    corr_info = []
    for a,b in corr_pairs:
        if a in obj_df.columns and b in obj_df.columns and obj_df[a].notna().any() and obj_df[b].notna().any():
            r = np.corrcoef(obj_df[a].fillna(0), obj_df[b].fillna(0))[0,1]
            corr_info.append(f"{a}↔{b}: r={r:.2f}")
    if corr_info:
        st.markdown("**Correlazioni oggetto:** " + " • ".join(corr_info))

# Analisi TSF Avanzata
if show_tsf and has_spectral:
    st.markdown("---")
    plot_tsf_correlations(timeline)
    
    # Calcola e mostra Indice di Immersività
    immersivity = calculate_immersivity_index(timeline)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("🌟 Indice di Immersività TSF", f"{immersivity:.3f}", delta=f"{(immersivity-0.5):.3f} vs baseline")
    with col2:
        # Gauge grafico
        fig_gauge, ax_gauge = plt.subplots(figsize=(3, 2))
        colors = ['red' if immersivity < 0.3 else 'orange' if immersivity < 0.7 else 'green']
        ax_gauge.barh([0], [immersivity], color=colors[0], alpha=0.8)
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(-0.5, 0.5)
        ax_gauge.set_xlabel('Immersivity Index')
        ax_gauge.set_yticks([])
        st.pyplot(fig_gauge)
    
    # Interpretazione
    if immersivity < 0.3:
        st.warning("⚠️ **Bassa immersività**: contenuto prevalentemente statico o poco variato timbricamente")
    elif immersivity < 0.7:
        st.info("ℹ️ **Media immersività**: buon equilibrio spazio-timbrico con potenziale di ottimizzazione")
    else:
        st.success("🎉 **Alta immersività**: excellent engagement spazio-timbrico per esperienza coinvolgente")

elif show_tsf and not has_spectral:
    st.warning("🔍 Carica un file audio per abilitare l'Analisi TSF Avanzata")

# Spettrogramma
if wav_file is not None and librosa and sf:
    wav_bytes = wav_file.read()
    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True, dtype='float32')
    if sr_target > 0 and sr_target != sr:
        # Resample ogni canale
        chans = [librosa.resample(data[:,i], orig_sr=sr, target_sr=sr_target) for i in range(data.shape[1])]
        # Pad lunghezze differenti (può accadere per rounding)
        L = min(len(c) for c in chans)
        data = np.stack([c[:L] for c in chans], axis=1)
        sr = sr_target
    # Determina canale
    ch_vals = obj_df['track_index'].dropna().unique()
    ch = int(ch_vals[0]) - 1 if len(ch_vals) else 0
    ch = max(0, min(ch, data.shape[1]-1))
    t0, t1 = float(obj_df['time_s'].min()), float(obj_df['time_s'].max())
    i0, i1 = int(t0*sr), int(t1*sr)
    y = data[i0:i1, ch] if i1>i0 else data[:, ch]
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    S = np.abs(librosa.stft(y, n_fft=int(n_fft), hop_length=int(hop_length)))**2
    S_db = librosa.power_to_db(S, ref=np.max)
    fig4, ax4 = plt.subplots(figsize=(9,3))
    librosa.display.specshow(S_db, sr=sr, hop_length=int(hop_length), x_axis='time', y_axis='hz', cmap='magma')
    ax4.set_title(f'Spectrogram ch {ch+1} ({t0:.2f}-{t1:.2f}s)')
    fig4.colorbar(ax4.images[0], ax=ax4, format='%+2.0f dB')
    fig4.tight_layout(); st.pyplot(fig4)
else:
    st.info('Carica un WAV per lo spettrogramma (richiede librosa e soundfile).')

st.caption("TSF Streamlit UI – analisi timbrico-spaziale front/back.")

def generate_spectral_analysis(wav_file, timeline_df, sr_target=22050):
    """Genera analisi spettrale automatica e la integra nel timeline"""
    if not librosa or not sf:
        st.error("librosa e soundfile richiesti per analisi spettrale")
        return timeline_df
    
    try:
        # Leggi file audio
        wav_bytes = wav_file.read()
        y, sr = sf.read(io.BytesIO(wav_bytes))
        
        if sr_target > 0 and sr_target != sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        
        # Estrai features spettrali su griglia temporale del timeline
        times = timeline_df['time_s'].unique()
        hop_length = 512
        frame_times = librosa.frames_to_time(range(len(y)//hop_length), sr=sr, hop_length=hop_length)
        
        # Calcola features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Interpola sui tempi del timeline
        spectral_features = {}
        for i in range(13):
            spectral_features[f'mfcc_{i+1:02d}'] = np.interp(times, frame_times[:len(mfccs[i])], mfccs[i])
        spectral_features['spec_centroid'] = np.interp(times, frame_times[:len(centroid)], centroid)
        spectral_features['spec_bandwidth'] = np.interp(times, frame_times[:len(bandwidth)], bandwidth)
        spectral_features['spec_rolloff'] = np.interp(times, frame_times[:len(rolloff)], rolloff)
        spectral_features['zcr'] = np.interp(times, frame_times[:len(zcr)], zcr)
        spectral_features['rms'] = np.interp(times, frame_times[:len(rms)], rms)
        
        # Aggiungi features al timeline
        spectral_df = pd.DataFrame(spectral_features, index=times).reset_index()
        spectral_df.rename(columns={'index': 'time_s'}, inplace=True)
        
        # Merge con timeline esistente
        timeline_enhanced = timeline_df.merge(spectral_df, on='time_s', how='left')
        return timeline_enhanced
        
    except Exception as e:
        st.error(f"Errore nell'analisi spettrale: {str(e)}")
        return timeline_df

def plot_tsf_correlations(timeline_df):
    """Genera grafici TSF specifici per correlazioni spettro-spazio"""
    if not any(col.startswith('mfcc_') for col in timeline_df.columns):
        st.warning("Features spettrali non disponibili per analisi TSF")
        return
    
    st.subheader("🎵 Analisi TSF: Correlazioni Timbro-Spazio-Forma")
    
    # 1. Brightness-Elevation Mapping
    if 'spec_centroid' in timeline_df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter = ax1.scatter(timeline_df['spec_centroid'], timeline_df['z_height'], 
                            c=timeline_df['time_s'], cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel('Spectral Centroid (Hz) - Brightness')
        ax1.set_ylabel('Z Height - Elevation')
        ax1.set_title('TSF: Brightness-Elevation Coupling')
        plt.colorbar(scatter, label='Time (s)')
        
        # Calcola correlazione
        from scipy.stats import pearsonr
        corr, p_val = pearsonr(timeline_df['spec_centroid'], timeline_df['z_height'])
        ax1.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3g}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        st.pyplot(fig1)
        st.caption("**Teoria TSF**: Suoni più brillanti (high-frequency) tendono a localizzarsi in alto per clarity perception")
    
    # 2. Spectral Complexity vs Lateral Spread
    if 'spec_bandwidth' in timeline_df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        lateral_pos = np.abs(timeline_df['x_lr'])  # Distanza dal centro
        scatter2 = ax2.scatter(timeline_df['spec_bandwidth'], lateral_pos,
                             c=timeline_df['time_s'], cmap='plasma', alpha=0.6, s=20)
        ax2.set_xlabel('Spectral Bandwidth (Hz) - Complexity')
        ax2.set_ylabel('|X Position| - Lateral Distance from Center')
        ax2.set_title('TSF: Spectral Complexity vs Lateral Positioning')
        plt.colorbar(scatter2, label='Time (s)')
        
        corr2, p_val2 = pearsonr(timeline_df['spec_bandwidth'], lateral_pos)
        ax2.text(0.05, 0.95, f'r = {corr2:.3f}, p = {p_val2:.3g}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        st.pyplot(fig2)
        st.caption("**Teoria TSF**: Suoni complessi possono richiedere separazione laterale per evitare masking")

def calculate_immersivity_index(timeline_df):
    """Calcola l'Indice di Immersività TSF"""
    if timeline_df.empty:
        return 0.0
    
    try:
        # 1. Spatial Coverage (0-1)
        spatial_range_x = timeline_df['x_lr'].max() - timeline_df['x_lr'].min()
        spatial_range_y = timeline_df['y_fb'].max() - timeline_df['y_fb'].min()
        spatial_range_z = timeline_df['z_height'].max() - timeline_df['z_height'].min()
        spatial_coverage = (spatial_range_x + spatial_range_y + spatial_range_z) / 6.0
        
        # 2. Movement Density (0-1)
        movement_density = 0.0
        if 'speed' in timeline_df.columns:
            movement_density = (timeline_df['speed'] > 0.01).mean()
        
        # 3. Spectral Richness (0-1)
        spectral_richness = 0.0
        if 'spec_bandwidth' in timeline_df.columns:
            spectral_richness = np.mean(timeline_df['spec_bandwidth']) / 11025.0
        
        # Formula Indice di Immersività TSF
        alpha, beta, gamma = 0.4, 0.3, 0.3
        immersivity_index = (alpha * spatial_coverage + 
                           beta * movement_density + 
                           gamma * spectral_richness)
        
        return min(1.0, immersivity_index)
        
    except Exception as e:
        st.error(f"Errore nel calcolo Immersivity Index: {e}")
        return 0.0

def extract_object_spectral_features(wav_file, timeline_df, object_id, sr_target=22050):
    """Estrae features spettrali per un oggetto specifico da file WAV ADM"""
    if librosa is None or sf is None:
        return None, None
    
    try:
        # Carica l'audio completo
        y, sr = sf.read(wav_file)
        if sr_target > 0 and sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        
        # Se multicanale, prendi il primo canale (fallback)
        if len(y.shape) > 1:
            y = y[:, 0]
        
        # Parametri analisi
        hop_length = int(sr * 0.01)  # 10ms
        n_fft = 2048
        n_mels = 128
        
        # Timeline dell'oggetto
        obj_timeline = timeline_df[timeline_df['object_id'] == object_id].copy()
        if obj_timeline.empty:
            return None, None
        
        # Range temporale dell'oggetto
        t_start = obj_timeline['time_s'].min()
        t_end = obj_timeline['time_s'].max()
        
        # Estrai segmento audio corrispondente all'oggetto
        start_sample = int(t_start * sr)
        end_sample = int(t_end * sr)
        
        if start_sample >= len(y) or start_sample < 0:
            return None, None
            
        end_sample = min(end_sample, len(y))
        y_obj = y[start_sample:end_sample]
        
        if len(y_obj) < hop_length:
            return None, None
        
        # Calcola spettrogramma MEL
        S = librosa.feature.melspectrogram(
            y=y_obj, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Calcola features spettrali
        mfcc = librosa.feature.mfcc(y=y_obj, sr=sr, n_mfcc=13, hop_length=hop_length)
        centroid = librosa.feature.spectral_centroid(y=y_obj, sr=sr, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y_obj, sr=sr, hop_length=hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=y_obj, sr=sr, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y_obj, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y_obj, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y_obj, hop_length=hop_length)[0]
        
        # Timeline features
        frame_times = librosa.frames_to_time(range(len(centroid)), sr=sr, hop_length=hop_length)
        frame_times = frame_times + t_start  # Offset al tempo assoluto
        
        # Crea DataFrame delle features
        features_df = pd.DataFrame({
            'time_s': frame_times,
            'spec_centroid': centroid,
            'spec_bandwidth': bandwidth,
            'spec_rolloff': rolloff,
            'zcr': zcr,
            'rms': rms,
        })
        
        # Aggiungi MFCC
        for i in range(13):
            features_df[f'mfcc_{i+1:02d}'] = mfcc[i]
        
        # Aggiungi contrast
        for i in range(contrast.shape[0]):
            features_df[f'spec_contrast_b{i}'] = contrast[i]
        
        # Timeline spettrogramma MEL
        mel_times = librosa.frames_to_time(range(S_db.shape[1]), sr=sr, hop_length=hop_length)
        mel_times = mel_times + t_start
        
        spectral_data = {
            'mel_spectrogram': S_db,
            'mel_freqs': librosa.mel_frequencies(n_mels=n_mels, fmax=sr//2),
            'mel_times': mel_times,
            'features': features_df
        }
        
        return spectral_data, features_df
        
    except Exception as e:
        st.error(f"Errore nell'analisi spettrale dell'oggetto {object_id}: {e}")
        return None, None

def plot_mel_spectrogram(spectral_data, object_id):
    """Visualizza lo spettrogramma MEL di un oggetto"""
    if spectral_data is None:
        st.warning(f"Nessun dato spettrale disponibile per oggetto {object_id}")
        return
    
    mel_spec = spectral_data['mel_spectrogram']
    mel_freqs = spectral_data['mel_freqs']
    mel_times = spectral_data['mel_times']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot spettrogramma MEL
    img = librosa.display.specshow(
        mel_spec, x_axis='time', y_axis='mel', sr=22050,
        fmax=8000, ax=ax, hop_length=int(22050*0.01)
    )
    
    # Aggiorna le etichette temporali per riflettere il tempo assoluto
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency')
    ax.set_title(f'MEL Spectrogram - Object {object_id}')
    
    # Aggiorna i tick temporali
    x_ticks = ax.get_xticks()
    x_labels = [f'{mel_times[0] + t:.1f}' for t in x_ticks if t < len(mel_times)]
    ax.set_xticklabels(x_labels)
    
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    st.pyplot(fig)

def plot_spectral_features_timeline(features_df, spatial_df):
    """Plotta features spettrali insieme a movimento spaziale"""
    if features_df is None or features_df.empty:
        st.warning("Nessuna feature spettrale da visualizzare")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. Centroide spettrale (brightness) vs Z position
    axes[0].plot(features_df['time_s'], features_df['spec_centroid'], 'b-', label='Spectral Centroid', linewidth=1.5)
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(spatial_df['time_s'], spatial_df['z_height'], 'r--', alpha=0.7, label='Z Height')
    axes[0].set_ylabel('Centroid (Hz)', color='blue')
    ax0_twin.set_ylabel('Z Height', color='red')
    axes[0].set_title('Brightness vs Elevation')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Bandwidth vs Lateral spread
    axes[1].plot(features_df['time_s'], features_df['spec_bandwidth'], 'g-', label='Bandwidth', linewidth=1.5)
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(spatial_df['time_s'], np.abs(spatial_df['x_lr']), 'm--', alpha=0.7, label='|X Position|')
    axes[1].set_ylabel('Bandwidth (Hz)', color='green')
    ax1_twin.set_ylabel('Lateral Distance', color='magenta')
    axes[1].set_title('Spectral Width vs Lateral Position')
    axes[1].grid(True, alpha=0.3)
    
    # 3. RMS Energy vs Movement
    axes[2].plot(features_df['time_s'], features_df['rms'], 'orange', label='RMS Energy', linewidth=1.5)
    ax2_twin = axes[2].twinx()
    if 'speed' in spatial_df.columns:
        ax2_twin.plot(spatial_df['time_s'], spatial_df['speed'], 'purple', alpha=0.7, label='Speed')
    axes[2].set_ylabel('RMS Energy', color='orange')
    ax2_twin.set_ylabel('Speed', color='purple')
    axes[2].set_title('Energy vs Movement')
    axes[2].grid(True, alpha=0.3)
    
    # 4. MFCC 1-3 (timbral characteristics)
    mfcc_cols = ['mfcc_01', 'mfcc_02', 'mfcc_03']
    colors = ['red', 'blue', 'green']
    for i, (col, color) in enumerate(zip(mfcc_cols, colors)):
        if col in features_df.columns:
            axes[3].plot(features_df['time_s'], features_df[col], color=color, 
                        label=f'MFCC {i+1}', linewidth=1, alpha=0.8)
    axes[3].set_ylabel('MFCC Coefficients')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('MFCC Timbral Evolution')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
