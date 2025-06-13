# ./instruments_analysis.py
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "dima806/musical_instrument_detection"

# Load feature extractor and model once
INST_FEAT = AutoFeatureExtractor.from_pretrained(MODEL_ID)
INST_MODEL = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
# Map class index to human-readable instrument name
ID2LABEL = INST_MODEL.config.id2label

@torch.no_grad()
def infer_instrument_probs(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Run one audio chunk through the instrument model and return a 1D numpy array of probabilities.
    """
    inputs = INST_FEAT(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    outputs = INST_MODEL(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    return probs


def get_instruments(
    file_path: str,
    segment_s: float = 5.0,
    stride_s: float = 2.0,
    agg_fn = np.max,
    min_score: float = 0.01,
) -> list:
    """
    Analyse multi-instrument probabilities over overlapping chunks of audio.
    Returns a list of (instrument, score) for those above min_score.
    """
    # 1. Load and resample audio to 16 kHz mono
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # 2. Chop into overlapping segments
    seg_len = int(segment_s * sr)
    hop_len = int(stride_s   * sr)
    chunks = [y[i : i + seg_len] for i in range(0, len(y) - seg_len + 1, hop_len)]
    if not chunks:
        chunks = [y]

    # 3. Run inference on each chunk, collect numpy arrays
    all_probs = np.stack([infer_instrument_probs(c, sr) for c in chunks])
    # shape: (n_chunks, n_classes)

    # 4. Aggregate over time (e.g. max or median)
    agg = agg_fn(all_probs, axis=0)
    # agg shape: (n_classes,)

    # 5. Threshold and sort
    picks = [
        (ID2LABEL[i], float(agg[i]))
        for i in np.argsort(agg)[::-1]
        if agg[i] >= min_score
    ]

    return picks
