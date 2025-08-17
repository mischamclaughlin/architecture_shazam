# flask_server/modules/helpers.py
from typing import Optional, Tuple, List, Dict, Any

from flask_server.modules.services.itunes_service import ITunesService
from flask_server.modules.services.spotify_service import SpotifyService
from flask_server.modules.services.acrcloud_service import ACRCloudService
from flask_server.modules.services.musicbrainz_service import MusicBrainzService
from flask_server.modules.analysis.classifiers import GenreClassifier
from flask_server.modules.analysis.classifiers import InstrumentClassifier
from flask_server.modules.analysis.features import AudioFeatureExtractor


def get_spotify_search(song_name: str) -> Dict[str, Any]:
    """Return SpotifyService.song_info() dict (may include 'error')."""
    return SpotifyService(song_name).lookup()


def get_itunes_search(song_name: str) -> Optional[Dict[str, Any]]:
    """Return ITunesService.lookup() dict or None."""
    return ITunesService(song_name).lookup()


def get_features(y, sr) -> Dict[str, Any]:
    """Extract key features from song."""
    f = AudioFeatureExtractor(y, sr)
    return {
        "tempo": f.tempo(),
        "key": f.key(),
        "timbre": f.describe_timbre_and_loudness()[0],
        "loudness": f.describe_timbre_and_loudness()[1],
    }


def get_song(file_path: str) -> Dict[str, Any]:
    """ACRCloud-based metadata extractor."""
    song = ACRCloudService(file_path).best_recognition() or {}
    if not song or song.get("score", 0) < 80:
        return {}
    return {
        "title": song.get("title", "—"),
        "artists": [a["name"] for a in song.get("artists", [])] or [],
        "album": song.get("album", {}).get("name", "—"),
        "release": song.get("release_date", "—"),
        "genres": [(g["name"], i) for i, g in enumerate(song.get("genres", []))],
    }


def get_origin(artist: str) -> Tuple[Optional[str], Optional[str]]:
    return MusicBrainzService(artist).get_country_and_area()


def get_genres(
    file_path: str, metadata: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float]]:
    """If metadata has genres, use that; else fall back to classifier."""
    if metadata and metadata.get("genres"):
        return metadata["genres"]

    try:
        out = GenreClassifier().predict(file_path)
        if isinstance(out, list) and all(isinstance(x, tuple) for x in out):
            return out
    except Exception as e:
        print("GenreClassifier failed:", e)

    return []


def get_instruments(y, sr) -> List[Tuple[str, float]]:
    return InstrumentClassifier().predict(y, sr)
