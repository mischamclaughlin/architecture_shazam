# ./flask_server/modules/helpers.py
from flask_server.modules import (
    AudioFeatureExtractor,
    GenreClassifier,
    InstrumentClassifier,
    ACRCloudService,
    MusicBrainzService,
)


def get_features(y, sr):
    audio_features = AudioFeatureExtractor(y, sr)
    tempo = audio_features.tempo()
    key = audio_features.key()
    timbre, loudness = audio_features.describe_timbre_and_loudness()

    return {"tempo": tempo, "key": key, "timbre": timbre, "loudness": loudness}


def get_song(file_path):
    song_recognition = ACRCloudService(file_path)
    song = song_recognition.best_recognition()
    if not song or song.get("score", 0) < 80:
        return {}

    title = song.get("title", "—")
    artists = [a["name"] for a in song.get("artists", [])] or "—"
    album = song.get("album", {}).get("name", "—")
    release = song.get("release_date", "—")
    genres = [(g["name"], i) for i, g in enumerate(song.get("genres", []))] or "—"

    return {
        "title": title,
        "artists": artists,
        "album": album,
        "release": release,
        "genres": genres,
    }


def get_origin(artist_name):
    musicbrainz = MusicBrainzService(artist_name)
    return musicbrainz.get_country_and_area()


def get_genres(info, file_name):
    genres = info.get("genres") if info else GenreClassifier().predict(file_name)
    return genres


def get_instruments(y, sr):
    instruments = InstrumentClassifier().predict(y, sr)
    return instruments
