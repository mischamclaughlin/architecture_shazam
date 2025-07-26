# ./flask_server/modules/analysis/features.py
import librosa
import numpy as np

# Default Krumhansl key profiles (Major and Minor)
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


class AudioFeatureExtractor:
    def __init__(
        self,
        y: np.ndarray,
        sr: int = 16_000,
        major_profile: np.ndarray = KRUMHANSL_MAJOR,
        minor_profile: np.ndarray = KRUMHANSL_MINOR,
    ) -> None:
        self.y = y
        self.sr = sr

        # normalize profiles once
        self.major_profile = major_profile / np.linalg.norm(major_profile)
        self.minor_profile = minor_profile / np.linalg.norm(minor_profile)

    def tempo(self) -> dict:
        tempo_global, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        local = librosa.feature.tempogram(y=self.y, sr=self.sr)
        return {
            "global": round(float(tempo_global), 2),
            "mean_local": round(float(np.mean(local)), 2),
            "median_local": round(float(np.median(local)), 2),
        }

    def key(self) -> str:
        chroma = librosa.feature.chroma_cens(y=self.y, sr=self.sr)
        profile = np.mean(chroma, axis=1)
        norm = profile / np.linalg.norm(profile)
        major_scores = [
            np.dot(np.roll(norm, -i), self.major_profile) for i in range(12)
        ]
        minor_scores = [
            np.dot(np.roll(norm, -i), self.minor_profile) for i in range(12)
        ]

        if max(major_scores) >= max(minor_scores):
            note = int(np.argmax(major_scores)) + 60
            return librosa.midi_to_note(note) + " major"

        note = int(np.argmax(minor_scores)) + 60
        return librosa.midi_to_note(note) + " minor"

    def describe_timbre_and_loudness(self) -> tuple[str, str]:
        cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        mean_cent = float(np.mean(cent))
        rms_vals = librosa.feature.rms(y=self.y)[0]
        rms_db = librosa.amplitude_to_db(rms_vals, ref=np.max)
        mean_rms_db = float(np.mean(rms_db))

        if mean_cent > 3000:
            timbre = "bright"
        elif mean_cent > 1500:
            timbre = "warm"
        else:
            timbre = "dark"

        if mean_rms_db > -10:
            loudness = "loud"
        elif mean_rms_db > -20:
            loudness = "medium"
        else:
            loudness = "soft"

        return timbre, loudness
