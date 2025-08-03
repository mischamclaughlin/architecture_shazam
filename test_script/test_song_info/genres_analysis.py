# ./genres_analysis.py
from transformers.pipelines import pipeline

# Load the genre classification pipeline once
MODEL_ID = "dima806/music_genres_classification"
GENRE_CLF = pipeline(
    "audio-classification",
    model=MODEL_ID,
    device=-1,
    return_all_scores=True,
    # optional segmentation
    chunk_length_s=10.0,
    # for smoothing
    stride_length_s=5.0,
)


def get_genre(file_path: str, threshold: float = 0.1) -> list:
    """
    Run the genre classifier on the audio file, returning
    genres with score >= threshold, sorted descending.
    """
    # Invoke the preloaded pipeline directly
    preds = GENRE_CLF(file_path)

    # Filter by confidence threshold
    filtered = [(r["label"], r["score"]) for r in preds if r["score"] >= threshold]

    # Sort by descending score
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
