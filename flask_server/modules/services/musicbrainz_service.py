# ./flask_server/modules/services/musicbrainzs_service.py
from typing import Optional, Tuple
from flask import current_app
import musicbrainzngs
from flask_server.modules.services.wiki_service import WikiService


class MusicBrainzService:
    def __init__(self, artist_name: str) -> None:
        musicbrainzngs.set_useragent(
            "ArchitectureShazam",
            "0.1",
            current_app.config["APP_SETTINGS"].SPOTIPY_CLIENT_ID,
        )
        self.artist_name = artist_name
        self.wiki = WikiService(artist_name=self.artist_name)

    def get_country_and_area(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            query = f"{self.artist_name}"
            results = musicbrainzngs.search_artists(query=query, limit=1)["artist-list"]

            if not results:
                raise LookupError(
                    f"No MusicBrainz artist found for {self.artist_name!r}"
                )

            mbid = results[0]["id"]
            details = musicbrainzngs.get_artist_by_id(mbid)
            art = details["artist"]
            country = art.get("country")
            area_name = art.get("area", {}).get("name")

            return country, area_name
        except (musicbrainzngs.WebServiceError, KeyError) as e:
            print(f"[MusicBrainz Error] {e}")

        origin = self.wiki.get_origin_wikipedia()
        return origin
