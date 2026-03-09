"""TSF Streamlit App - Versione Semplificata con Analisi Spettrale
Versione semplificata che garantisce il funzionamento dell'analisi spettrale
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy.stats import pearsonr
import io

st.set_page_config(page_title="TSF Spectral Analysis", layout="wide")
st.title("🎵 TSF: Analisi Spettro-Spaziale")

# Sidebar
with st.sidebar:
    st.header("Input Files")
    
    # File uploads
    csv_file = st.file_uploader("Timeline CSV", type=['csv'])
    wav_file = st.file_uploader("Audio WAV", type=['wav'])
    
    # Parameters
    st.header("Parameters")
    hop_ms = st.slider("Hop size (ms)", 5, 50, 10)
    n_mels = st.slider("Mel bins", 64, 256, 128)
    show_correlations = st.checkbox("Show correlations", True)

# Main content
if csv_file and wav_file:
    # Load timeline
    timeline = pd.read_csv(csv_file)
    st.success(f"Timeline loaded: {len(timeline)} frames")
    
    # Load audio
    wav_bytes = wav_file.read()
    y, sr = sf.read(io.BytesIO(wav_bytes))
    
    # If multichannel, use first channel
    if len(y.shape) > 1:
        y = y[:, 0]
    
    st.success(f"Audio loaded: {len(y)/sr:.2f}s @ {sr}Hz")
    
    # Audio analysis parameters
    hop_length = int(sr * hop_ms / 1000)
    n_fft = 2048
    
    # Spectral analysis
    st.header("🔍 Analisi Spettrale")
    
    with st.spinner("Extracting spectral features..."):
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Other spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Time frames
        frame_times = librosa.frames_to_time(range(len(centroid)), sr=sr, hop_length=hop_length)
    
    st.success(f"✅ Features extracted: {mfcc.shape[1]} frames")
    
    # 1. MEL SPECTROGRAM
    st.subheader("📊 Mel Spectrogram")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', 
                           sr=sr, hop_length=hop_length, ax=ax1)
    ax1.set_title('Mel Spectrogram (dB)')
    plt.colorbar(ax1.collections[0], ax=ax1, format='%+2.0f dB')
    st.pyplot(fig1)
    
    # 2. MFCC HEATMAP
    st.subheader("🌡️ MFCC Coefficients")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length, ax=ax2)
    ax2.set_title('MFCC Coefficients')
    ax2.set_ylabel('MFCC Coefficients')
    plt.colorbar(ax2.collections[0], ax=ax2)
    st.pyplot(fig2)
    
    # 3. SPECTRAL FEATURES TIME SERIES
    st.subheader("📈 Spectral Features")
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes3[0].plot(frame_times, centroid, color='red', alpha=0.8)
    axes3[0].set_ylabel('Spectral Centroid (Hz)')
    axes3[0].set_title('Spectral Centroid (Brightness)')
    axes3[0].grid(True, alpha=0.3)
    
    axes3[1].plot(frame_times, bandwidth, color='blue', alpha=0.8)
    axes3[1].set_ylabel('Spectral Bandwidth (Hz)')
    axes3[1].set_title('Spectral Bandwidth (Spread)')
    axes3[1].grid(True, alpha=0.3)
    
    axes3[2].plot(frame_times, rms, color='green', alpha=0.8)
    axes3[2].set_ylabel('RMS Energy')
    axes3[2].set_title('RMS Energy (Loudness)')
    axes3[2].set_xlabel('Time (s)')
    axes3[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # 4. SPATIAL DATA VISUALIZATION
    st.subheader("🗺️ Spatial Movement")
    
    # Check if spatial data is available
    if 'x_lr' in timeline.columns and 'y_fb' in timeline.columns:
        fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8))
        
        # XY trajectory
        scatter = axes4[0,0].scatter(timeline['x_lr'], timeline['y_fb'], 
                                   c=timeline['time_s'], cmap='viridis', alpha=0.6)
        axes4[0,0].set_xlabel('X (Left-Right)')
        axes4[0,0].set_ylabel('Y (Front-Back)')
        axes4[0,0].set_title('XY Trajectory')
        plt.colorbar(scatter, ax=axes4[0,0])
        
        # Time series of positions
        axes4[0,1].plot(timeline['time_s'], timeline['x_lr'], label='X (LR)', alpha=0.7)
        axes4[0,1].plot(timeline['time_s'], timeline['y_fb'], label='Y (FB)', alpha=0.7)
        if 'z_height' in timeline.columns:
            axes4[0,1].plot(timeline['time_s'], timeline['z_height'], label='Z (Height)', alpha=0.7)
        axes4[0,1].set_xlabel('Time (s)')
        axes4[0,1].set_ylabel('Position')
        axes4[0,1].set_title('Spatial Position vs Time')
        axes4[0,1].legend()
        axes4[0,1].grid(True, alpha=0.3)
        
        # Speed if available
        if 'speed' in timeline.columns:
            axes4[1,0].plot(timeline['time_s'], timeline['speed'], color='red', alpha=0.8)
            axes4[1,0].set_xlabel('Time (s)')
            axes4[1,0].set_ylabel('Speed (units/s)')
            axes4[1,0].set_title('Spatial Speed')
            axes4[1,0].grid(True, alpha=0.3)
        
        # Front/Back occupancy
        front_mask = timeline['y_fb'] > 0.02
        rear_mask = timeline['y_fb'] < -0.02
        center_mask = (timeline['y_fb'] >= -0.02) & (timeline['y_fb'] <= 0.02)
        
        occupancy = [front_mask.sum(), center_mask.sum(), rear_mask.sum()]
        axes4[1,1].bar(['Front', 'Center', 'Rear'], occupancy, color=['orange', 'gray', 'green'])
        axes4[1,1].set_ylabel('Frame Count')
        axes4[1,1].set_title('Front/Center/Rear Occupancy')
        
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.warning("Spatial data (x_lr, y_fb) not found in timeline")
    
    # 5. SPETTRO-SPATIAL CORRELATIONS
    if show_correlations and 'x_lr' in timeline.columns:
        st.subheader("🔗 TSF Correlations: Spettro-Spaziali")
        
        # Align spectral and spatial data
        timeline_times = timeline['time_s'].values
        
        # Interpolate spectral features to timeline grid
        centroid_aligned = np.interp(timeline_times, frame_times, centroid)
        bandwidth_aligned = np.interp(timeline_times, frame_times, bandwidth)
        rms_aligned = np.interp(timeline_times, frame_times, rms)
        
        # Calculate correlations
        col1, col2, col3 = st.columns(3)
        
        if 'z_height' in timeline.columns:
            corr_bright_height = pearsonr(centroid_aligned, timeline['z_height'])[0]
            with col1:
                st.metric("Brightness ↔ Height", f"{corr_bright_height:.3f}")
        
        corr_band_lateral = pearsonr(bandwidth_aligned, np.abs(timeline['x_lr']))[0]
        with col2:
            st.metric("Bandwidth ↔ Lateral", f"{corr_band_lateral:.3f}")
        
        if 'speed' in timeline.columns:
            corr_rms_speed = pearsonr(rms_aligned, timeline['speed'])[0]
            with col3:
                st.metric("RMS ↔ Speed", f"{corr_rms_speed:.3f}")
        
        # Correlation scatter plots
        fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
        
        if 'z_height' in timeline.columns:
            axes5[0].scatter(centroid_aligned, timeline['z_height'], alpha=0.6, color='red')
            axes5[0].set_xlabel('Spectral Centroid (Hz)')
            axes5[0].set_ylabel('Z Height')
            axes5[0].set_title(f'Brightness vs Height (r={corr_bright_height:.3f})')
            axes5[0].grid(True, alpha=0.3)
        
        axes5[1].scatter(bandwidth_aligned, np.abs(timeline['x_lr']), alpha=0.6, color='blue')
        axes5[1].set_xlabel('Spectral Bandwidth (Hz)')
        axes5[1].set_ylabel('|X Position|')
        axes5[1].set_title(f'Bandwidth vs Lateral (r={corr_band_lateral:.3f})')
        axes5[1].grid(True, alpha=0.3)
        
        if 'speed' in timeline.columns:
            axes5[2].scatter(rms_aligned, timeline['speed'], alpha=0.6, color='green')
            axes5[2].set_xlabel('RMS Energy')
            axes5[2].set_ylabel('Spatial Speed')
            axes5[2].set_title(f'Energy vs Speed (r={corr_rms_speed:.3f})')
            axes5[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig5)
        
        # TSF Interpretation
        st.subheader("🎯 TSF Theory Interpretation")
        
        if 'z_height' in timeline.columns:
            if corr_bright_height > 0.3:
                st.success(f"✅ **Brightness-Height Coupling**: Strong positive correlation ({corr_bright_height:.3f}) - bright sounds positioned higher for clarity")
            elif corr_bright_height < -0.3:
                st.info(f"ℹ️ **Inverted Brightness-Height**: Negative correlation ({corr_bright_height:.3f}) - creative use of low positioning for bright sounds")
            else:
                st.warning(f"⚠️ **Weak Brightness-Height**: Low correlation ({corr_bright_height:.3f}) - potential for optimization")
        
        if corr_band_lateral > 0.3:
            st.success(f"✅ **Complexity-Separation**: Complex sounds ({corr_band_lateral:.3f}) positioned laterally for masking avoidance")
        elif corr_band_lateral < -0.3:
            st.info(f"ℹ️ **Central Complex**: Complex sounds kept central ({corr_band_lateral:.3f}) - potential masking risk")
        
        if 'speed' in timeline.columns and corr_rms_speed > 0.3:
            st.success(f"✅ **Energy-Movement Sync**: Strong correlation ({corr_rms_speed:.3f}) - energetic sounds move more")

else:
    st.info("📁 Carica sia un file CSV timeline che un file audio WAV per iniziare l'analisi")
    
    st.markdown("""
    ### 📋 Come usare questa app:
    
    1. **Timeline CSV**: Carica il file CSV generato dall'analisi ADM (es. `objects_timeline.csv`)
    2. **Audio WAV**: Carica il file audio originale (es. `Sine Zoom Frontale_Object.wav`)
    3. **Analisi automatica**: L'app genererà automaticamente:
       - Mel Spectrogram
       - MFCC Heatmap  
       - Features spettrali nel tempo
       - Traiettorie spaziali
       - Correlazioni TSF (Timbro-Spazio-Forma)
    
    ### 🎯 Funzionalità TSF:
    - **Brightness-Height Mapping**: correlazione tra centroide spettrale ed elevazione
    - **Complexity-Separation**: relazione tra larghezza spettrale e posizione laterale  
    - **Energy-Movement Sync**: sincronizzazione tra energia RMS e velocità spaziale
    """)
