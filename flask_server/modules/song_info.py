# ./song_info.py
import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import os

import librosa
import numpy as np
import torch
from librosa.beat import beat_track
from librosa.feature.rhythm import tempo as librosa_tempo
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline
import musicbrainzngs
import requests
import mwparserfromhell
from dotenv import load_dotenv
from acrcloud.recognizer import ACRCloudRecognizer


from .logger import default_logger


# Load keys
load_dotenv()

# ACRCloud config
ACRCLOUD_CONFIG = {
    "host": os.environ["ACRCLOUD_HOST"],
    "access_key": os.environ["ACRCLOUD_KEY"],
    "access_secret": os.environ["ACRCLOUD_SECRET"],
    "timeout": 10,  # seconds
}
RECOGNISER = ACRCloudRecognizer(ACRCLOUD_CONFIG)

# MusicBrainz config
musicbrainzngs.set_useragent("MyApp", "0.1", "you@example.com")


# Wikipedia config
HEADERS = {"User-Agent": "ArchitectureShazamScript/0.1 (you@example.com)"}
API_URL = "https://en.wikipedia.org/w/api.php"


# Default Krumhansl key profiles (Major and Minor)
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


class ExtractSongInfo:
    """
    Analyse an audio file for musical features, genre and instrument classification.
    """

    def __init__(
        self,
        file_path: str,
        major_profile: np.ndarray = KRUMHANSL_MAJOR,
        minor_profile: np.ndarray = KRUMHANSL_MINOR,
        genre_model_id: str = "dima806/music_genres_classification",
        inst_model_id: str = "dima806/musical_instrument_detection",
        genre_threshold: float = 0.1,
        genre_chunk: float = 10.0,
        genre_stride: float = 5.0,
        inst_segment: float = 5.0,
        inst_stride: float = 2.0,
        device: int = -1,
        headers: dict = HEADERS,
        api_url: str = API_URL,
        recogniser=RECOGNISER,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.file_path = file_path
        self.major_profile = major_profile / np.linalg.norm(major_profile)
        self.minor_profile = minor_profile / np.linalg.norm(minor_profile)
        self.genre_model_id = genre_model_id
        self.inst_model_id = inst_model_id
        self.genre_threshold = genre_threshold
        self.genre_chunk = genre_chunk
        self.genre_stride = genre_stride
        self.inst_segment = inst_segment
        self.inst_stride = inst_stride
        self.device = device
        self.headers = headers
        self.api_url = api_url
        self.recogniser = recogniser
        self.offsets = [45, 60, 75, 90]
        self.clip_length = 10
        self._genre_clf = None
        self._inst_feat = None
        self._inst_model = None
        self.logger = logger or default_logger(__name__)

    @property
    def genre_clf(self):
        """
        Lazy-load the genre classification pipeline.
        """
        if self._genre_clf is None:
            self._genre_clf = pipeline(
                "audio-classification",
                model=self.genre_model_id,
                device=self.device,
                return_all_scores=True,
                chunk_length_s=self.genre_chunk,
                stride_length_s=self.genre_stride,
            )
            self.logger.debug("Genre classifier initialised.")
        return self._genre_clf

    @property
    def inst_feat(self):
        """
        Lazy-load the instrument feature extractor.
        """
        if self._inst_feat is None:
            self._inst_feat = AutoFeatureExtractor.from_pretrained(self.inst_model_id)
            self.logger.debug("Instrument feature extractor initialised.")
        return self._inst_feat

    @property
    def inst_model(self):
        """
        Lazy-load the instrument classification model.
        """
        if self._inst_model is None:
            self._inst_model = AutoModelForAudioClassification.from_pretrained(
                self.inst_model_id
            )
            self.logger.debug("Instrument model initialised.")
        return self._inst_model

    def analyse_features(self) -> dict:
        """
        Analyse audio for tempo, key, spectral descriptors, MFCCs, timbre and loudness.
        """
        y, sr = librosa.load(self.file_path, sr=None)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)

        tempo_info = self.tempo_analysis(y, sr)
        key_str = self.key_analysis(y, sr)
        timbre, loudness = self.describe_timbre_and_loudness(y, sr)

        return {
            "tempo_global": round(tempo_info["tempo_global"], 2),
            "tempo_mean_local": round(tempo_info["tempo_mean"], 2),
            "tempo_median_local": round(tempo_info["tempo_median"], 2),
            "key": key_str,
            "timbre": timbre,
            "loudness": loudness,
        }

    def tempo_analysis(self, time_series: np.ndarray, sample_rate: int) -> dict:
        """
        Estimate global and local tempo; handle short signals gracefully.
        """
        try:
            tempo_global, _ = beat_track(y=time_series, sr=sample_rate)
        except Exception as e:
            self.logger.warning(f"Beat tracking failed: {e}")
            tempo_global = 0.0
        local_tempo = librosa_tempo(y=time_series, sr=sample_rate, aggregate=None)
        return {
            "tempo_global": float(tempo_global),
            "tempo_mean": float(np.mean(local_tempo)),
            "tempo_median": float(np.median(local_tempo)),
        }

    def key_analysis(self, time_series: np.ndarray, sample_rate: int) -> str:
        """
        Detect musical key (major/minor) using Krumhansl profiles.
        """
        chroma = librosa.feature.chroma_cens(y=time_series, sr=sample_rate)
        profile = np.mean(chroma, axis=1)
        profile_norm = profile / np.linalg.norm(profile)
        maj_corr = [
            np.dot(np.roll(profile_norm, -i), self.major_profile) for i in range(12)
        ]
        min_corr = [
            np.dot(np.roll(profile_norm, -i), self.minor_profile) for i in range(12)
        ]
        if max(maj_corr) >= max(min_corr):
            return librosa.midi_to_note(int(np.argmax(maj_corr)) + 60) + " major"
        return librosa.midi_to_note(int(np.argmax(min_corr)) + 60) + " minor"

    def describe_timbre_and_loudness(
        self, time_series: np.ndarray, sample_rate: int
    ) -> tuple[str, str]:
        """
        Classify timbre ('bright', 'warm', 'dark') and loudness ('loud', 'medium', 'soft').
        """
        cent = librosa.feature.spectral_centroid(y=time_series, sr=sample_rate)[0]
        mean_cent = float(np.mean(cent))
        rms_vals = librosa.feature.rms(y=time_series)[0]
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

    def get_genre(self) -> list[tuple[str, float]]:
        """
        Run genre classifier and return labels above threshold, sorted by confidence.
        """
        preds = self.genre_clf(self.file_path)
        filtered = [
            (r["label"], r["score"])
            for r in preds
            if r["score"] >= self.genre_threshold
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered

    def best_recognition(self) -> dict | None:
        """
        Read the entire MP3 into memory and, for each offset (in seconds),
        call recognize_by_filebuffer to get ACRCloud's best match for that window.
        Returns the single track dict with the highest score (or None).
        """
        # load file once
        with open(self.file_path, "rb") as f:
            mp3_buf = f.read()

        best_track = None
        best_score = -1

        for off in self.offsets:
            # fingerprint the [off, off+length_sec) slice
            resp = self.recogniser.recognize_by_filebuffer(
                mp3_buf, off, self.clip_length
            )
            data = json.loads(resp)

            # skip errors or no‐matches
            if data["status"]["code"] != 0 or not data["metadata"].get("music"):
                continue

            # find the top‐scoring track in this slice
            candidate = max(data["metadata"]["music"], key=lambda t: t["score"])
            if candidate["score"] > best_score:
                best_score = candidate["score"]
                best_track = candidate

        if best_track["score"] > 80:
            return best_track

        return None

    def find_page_title(self, name):
        params = {
            "action": "opensearch",
            "search": name,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
        r = requests.get(self.api_url, params=params, headers=self.headers)
        if r.status_code != 200:
            return None
        titles = r.json()[1]
        return titles[0] if titles else None

    def fetch_wikitext(self, title):
        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
        }
        r = requests.get(self.api_url, params=params, headers=self.headers)
        if r.status_code != 200:
            return None
        return r.json()["parse"]["wikitext"]["*"]

    def get_origin_wikipedia(self, artist_name):
        title = self.find_page_title(artist_name)
        if not title:
            return None

        wikitext = self.fetch_wikitext(title)
        if not wikitext:
            return None

        wd = mwparserfromhell.parse(wikitext)

        infoboxes = [
            tpl
            for tpl in wd.filter_templates()
            if tpl.name.strip().lower().startswith("infobox")
        ]
        if not infoboxes:
            return None

        for tpl in infoboxes:
            for field in ("origin", "birth_place", "birth_place_location"):
                if tpl.has(field):
                    raw = tpl.get(field).value
                    clean = raw.strip_code().strip()
                    return clean

        return None

    def get_country_and_area(self):
        song_info = self.best_recognition()
        if not song_info:
            return None

        artist_name = song_info["artists"][0]
        try:
            query = f"{artist_name}"
            results = musicbrainzngs.search_artists(query=query, limit=1)["artist-list"]

            if not results:
                raise LookupError(f"No MusicBrainz artist found for {artist_name!r}")

            mbid = results[0]["id"]
            details = musicbrainzngs.get_artist_by_id(mbid)
            art = details["artist"]
            country = art.get("country")
            area_name = art.get("area", {}).get("name")

            return country, area_name
        except (musicbrainzngs.WebServiceError, KeyError) as e:
            print(f"[MusicBrainz Error] {e}")

        origin = self.get_origin_wikipedia(artist_name)
        return origin

    @torch.no_grad()
    def infer_instrument_probs(self, wav: np.ndarray, sr: int) -> np.ndarray:
        inputs = self.inst_feat(
            wav, sampling_rate=sr, return_tensors="pt", padding=True
        )
        outputs = self.inst_model(**inputs)
        return torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    def get_instruments(self) -> list[tuple[str, float]]:
        """
        Analyse overlapping chunks for instrument probabilities.
        """
        y, sr = librosa.load(self.file_path, sr=16000, mono=True)
        seg_len = int(self.inst_segment * sr)
        hop_len = int(self.inst_stride * sr)
        chunks = [
            y[i : i + seg_len] for i in range(0, len(y) - seg_len + 1, hop_len)
        ] or [y]
        all_probs = np.stack([self.infer_instrument_probs(c, sr) for c in chunks])
        agg = np.max(all_probs, axis=0)
        labels = self.inst_model.config.id2label
        results = [
            (labels[i], float(score)) for i, score in enumerate(agg) if score >= 0.01
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)


# Test from command line
def main():
    """
    CLI entry point for extracting song info.
    """
    parser = argparse.ArgumentParser(
        description="Extract musical features, genre and instruments from audio."
    )
    parser.add_argument("file", type=Path, help="Path to audio file.")
    parser.add_argument("--output-json", type=Path, help="Save results to a JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = default_logger(log_level)

    esi = ExtractSongInfo(str(args.file), logger=logger)
    features = esi.analyse_features()
    genres = esi.get_genre()
    instruments = esi.get_instruments()
    location = esi.get_country_and_area()

    output = {
        "features": features,
        "genres": genres,
        "instruments": instruments,
        "location": location,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
