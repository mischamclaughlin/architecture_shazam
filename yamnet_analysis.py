# ./yamnet_analysis.py
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa


# Load once
YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")


def analyse_yamnet(file_path: str):
    # Load & prep audio
    wav, sr = librosa.load(file_path, sr=None, mono=False)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)

    # Inference
    scores, _, _ = YAMNET_MODEL(wav)

    # Load class names, skipping the header row
    class_map_path = YAMNET_MODEL.class_map_path().numpy().decode()
    class_names = []
    with open(class_map_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row['mid'], row['display_name']
            class_names.append((row["mid"], row["display_name"]))

    # Aggregate and pick top-5
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()

    threshold = 0.02
    filtered = []
    for idx, score in enumerate(mean_scores):
        if score >= threshold:
            _, name = class_names[idx]
            filtered.append((name, score))
    filtered.sort(key=lambda x: x[1], reverse=True)

    return filtered
