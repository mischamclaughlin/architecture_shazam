# ./musicbrainz_api_test.py
import musicbrainzngs
from wikipedia_api_test import get_origin_wikipedia

musicbrainzngs.set_useragent("MyApp", "0.1", "you@example.com")


def get_country_and_area(artist_name):
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

    origin = get_origin_wikipedia(artist_name)
    return origin


print(get_country_and_area("Teresa Teng"))
print(get_country_and_area("Carmen Twillie"))
