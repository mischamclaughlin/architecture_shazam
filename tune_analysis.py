import librosa
from librosa.beat import beat_track
from librosa.feature.rhythm import tempo as librosa_tempo
import numpy as np

def analyse_tune(file_path):
  y, sr = librosa.load(file_path)

  # Analyse Tempo
  tempo_global, _ = beat_track(y=y, sr=sr)
  try:
    tempo_global = float(tempo_global)
  except Exception:
    tempo_global = tempo_global.item()
  
  local_tempo = librosa_tempo(y=y, sr=sr, aggregate=None)
  mean_tempo = float(np.mean(local_tempo))
  median_tempo = float(np.median(local_tempo))

  # Analyse Key
  chroma = librosa.feature.chroma_stft(y=y, sr=sr)
  key_index = chroma.mean(axis=1).argmax()

  # Analyse Bandwidth
  bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  bandwidth = np.squeeze(bandwidth)

  mean_bw = float(np.mean(bandwidth))
  median_bw = float(np.median(bandwidth))
  std_bw = float(np.std(bandwidth))

  # Analyse Centroid
  centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
  centroid = np.squeeze(centroid)
  mean_centroid = float(np.mean(centroid))
  median_centroid = float(np.median(centroid))
  std_centroid = float(np.std(centroid))
  
  # Analyse RMS
  rms = librosa.feature.rms(y=y)
  rms = np.squeeze(rms)
  mean_rms = float(np.mean(rms))
  max_rms = float(np.max(rms))
  
  # Analyse MFCCs
  mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
  mfccs_means = np.mean(mfccs, axis=1)
  mfccs_stds = np.std(mfccs, axis=1)
  
  mfcc_summary = {}
  for i in range(13):
    mfcc_summary[f"mfcc{i+1}_mean"] = round(float(mfccs_means[i]), 2)
    mfcc_summary[f"mfcc{i+1}_std"] = round(float(mfccs_stds[i]), 2)

  return {
    'tempo_global': round(tempo_global, 2),
    'tempo_mean_local': round(mean_tempo, 2),
    'tempo_median_local': round(median_tempo, 2),
    
    'mean_bandwidth': round(mean_bw, 2),
    'median_bandwidth': round(median_bw, 2),
    'std_bandwidth': round(std_bw, 2),
    
    'mean_centroid': round(mean_centroid, 2),
    'median_centroid': round(median_centroid, 2),
    'std_centroid': round(std_centroid, 2),
    
    'mean_rms': round(mean_rms, 4),
    'max_rms': round(max_rms, 4),
    
    **mfcc_summary,
    
    'key': librosa.midi_to_note(key_index + 60),
    'spectral_bandwidth': bandwidth.mean(),
    'mood': 'calm',
    'instruments': ['piano', 'strings']
  }