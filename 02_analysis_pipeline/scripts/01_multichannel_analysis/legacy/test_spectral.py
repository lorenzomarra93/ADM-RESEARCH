#!/usr/bin/env python3
"""
Script di test per generare analisi spettrale e testare la funzionalità
"""
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def test_spectral_analysis():
    # Carica il file audio di test
    audio_file = "/Users/lorenzomarra/Desktop/PHD - VScode/Test/Sinusoide Test 1/Sine Zoom Frontale_Object.wav"
    
    print(f"Caricando: {audio_file}")
    
    # Carica audio
    y, sr = sf.read(audio_file)
    print(f"Audio caricato: shape={y.shape}, sr={sr}")
    
    # Se multichannel, usa solo il primo canale
    if len(y.shape) > 1:
        y = y[:, 0]
        print(f"Usando primo canale: shape={y.shape}")
    
    # Parametri analisi
    hop_length = int(sr * 0.01)  # 10ms
    n_fft = 2048
    n_mels = 128
    
    # Estrai features spettrali
    print("Estraendo MFCC...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
    
    print("Estraendo Mel spectrogram...")
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    print("Estraendo features spettrali...")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Converti frame in tempo
    frame_times = librosa.frames_to_time(range(len(centroid)), sr=sr, hop_length=hop_length)
    
    print(f"Features estratte:")
    print(f"  MFCC shape: {mfcc.shape}")
    print(f"  Mel spectrogram shape: {mel_spec_db.shape}")
    print(f"  Centroid shape: {centroid.shape}")
    print(f"  Frame times: {len(frame_times)} frames, {frame_times[0]:.3f}s to {frame_times[-1]:.3f}s")
    
    # Crea DataFrame con features
    df_spectral = pd.DataFrame({
        'time_s': frame_times,
        'spec_centroid': centroid,
        'spec_bandwidth': bandwidth,
        'spec_rolloff': rolloff,
        'zcr': zcr,
        'rms': rms,
    })
    
    # Aggiungi MFCC
    for i in range(13):
        df_spectral[f'mfcc_{i+1:02d}'] = mfcc[i]
    
    print(f"DataFrame creato: {df_spectral.shape}")
    print("Prime 5 righe:")
    print(df_spectral.head())
    
    # Salva per debug
    df_spectral.to_csv('test_spectral_features.csv', index=False)
    print("Salvato in test_spectral_features.csv")
    
    # Plot mel spectrogram
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, hop_length=hop_length)
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 1, 2)
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length)
    plt.title('MFCC')
    plt.colorbar()
    
    plt.subplot(3, 1, 3)
    plt.plot(frame_times, centroid, label='Centroid')
    plt.plot(frame_times, bandwidth/10, label='Bandwidth/10')
    plt.plot(frame_times, rms*1000, label='RMS*1000')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Spectral Features')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_spectral_analysis.png', dpi=150)
    print("Plot salvato in test_spectral_analysis.png")
    
    return df_spectral, mel_spec_db, frame_times

if __name__ == "__main__":
    test_spectral_analysis()
