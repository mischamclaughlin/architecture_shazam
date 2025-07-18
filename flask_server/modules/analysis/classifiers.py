# ./flask_server/modules/analysis/classifiers.py
import torch
import numpy as np
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification


class GenreClassifier:
    def __init__(
        self,
        model_id: str = "dima806/music_genres_classification",
        device: int = -1,
        threshold: float = 0.1,
    ) -> None:
        self.clf = pipeline(
            "audio-classification",
            model=model_id,
            device=device,
            return_all_scores=True,
            chunk_length_s=10,
            stride_length_s=5,
        )
        self.threshold = threshold

    def predict(self, file_path: str) -> list[tuple[str, float]]:
        scores = self.clf(file_path)[0]

        return sorted(
            [(r["label"], r["score"]) for r in scores if r["score"] >= self.threshold],
            key=lambda x: x[1],
            reverse=True,
        )


class InstrumentClassifier:
    def __init__(
        self,
        model_id: str = "dima806/musical_instrument_detection",
        device: int = -1,
        threshold: float = 0.01,
    ) -> None:
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int = 16_000) -> list[tuple[str, float]]:
        inputs = self.fe(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        logits = self.model(**inputs).logits.cpu().numpy()[0]
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        labels = self.model.config.id2label

        return sorted(
            [(labels[i], float(p)) for i, p in enumerate(probs) if p > self.threshold],
            key=lambda x: x[1],
            reverse=True,
        )
