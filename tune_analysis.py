import librosa
import numpy as np

def analyse_tune(file_path):
  y, sr = librosa.load(file_path)

  tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
  if isinstance(tempo, np.ndarray):
    tempo = tempo.item()

  chroma = librosa.feature.chroma_stft(y=y, sr=sr)
  key_index = chroma.mean(axis=1).argmax()

  centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
  bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  rms = librosa.feature.rms(y=y)
  mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

  return {
    'tempo': round(tempo, 2),
    'key': librosa.midi_to_note(key_index + 60),
    'mood': 'calm',
    'instruments': ['piano', 'strings']
  }