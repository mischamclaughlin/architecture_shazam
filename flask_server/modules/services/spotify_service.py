# ./flask_server/modules/services/spotify_service.py
from typing import Dict, Any
import urllib.parse
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from flask import current_app


class SpotifyService:
    """
    Wrapper around Spotipy to fetch public track metadata
    (including a 30s clip) for a given song name.
    """

    def __init__(self, song_name: str) -> None:
        auth = SpotifyClientCredentials(
            client_id=current_app.config["APP_SETTINGS"].SPOTIPY_CLIENT_ID,
            client_secret=current_app.config["APP_SETTINGS"].SPOTIPY_CLIENT_SECRET,
        )
        self.sp = Spotify(auth_manager=auth)
        self.song_name = song_name

    def lookup(self) -> Dict[str, Any]:
        """
        Search Spotify for the first track matching `song_name`.
        Returns a dict with:
          - title, artist, album, duration_ms
          - preview_url (may be None)
          - spotify_url
          - error (only if nothing matched)
        """
        q = f"track:{urllib.parse.quote_plus(self.song_name)}"
        items = self.sp.search(q=q, type="track", limit=1, market="US")["tracks"][
            "items"
        ]
        if not items:
            return {"error": "No matching track on Spotify"}
        t = items[0]
        return {
            "title": t["name"],
            "artist": t["artists"][0]["name"],
            "album": t["album"]["name"],
            "duration_ms": t["duration_ms"],
            "preview_url": t.get("preview_url"),
            "spotify_url": t["external_urls"]["spotify"],
        }
