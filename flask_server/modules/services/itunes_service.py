# ./flask_server/modules/services/itunes_service.py
from typing import Optional, Dict
import requests


class ITunesService:
    """
    Wrapper around Apple's iTunes Search API to fetch
    track metadata and a 30s snippet URL without any auth.
    """

    def __init__(self, song_name: str) -> None:
        self.base_url = "https://itunes.apple.com/search"
        self.song_name = song_name

    def lookup(self) -> Optional[Dict[str, any]]:
        """
        Search iTunes for the first matching song by term.
        Returns a dict with:
          - title: str
          - artist: str
          - album: str
          - duration_ms: int
          - preview_url: str
          - itunes_url: str
          - artwork_url: str
        or None if no result.
        """
        params = {
            "term": self.song_name,
            "entity": "song",
            "limit": 10,
        }
        r = requests.get(self.base_url, params=params, timeout=5)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None

        bad = ["remix", "live", "version", "edit", "remastered"]
        for song in results:
            name = (song.get("trackName") or "").lower()
            preview = song.get("previewUrl")

            if not preview:
                continue

            if any(word in name for word in bad):
                continue

            # song = results[0]
            return {
                "title": song.get("trackName"),
                "artist": song.get("artistName"),
                "album": song.get("collectionName"),
                "duration_ms": song.get("trackTimeMillis"),
                "preview_url": song.get("previewUrl"),
                "itunes_url": song.get("trackViewUrl"),
                "artwork_url": song.get("artworkUrl100"),
            }

        return None
