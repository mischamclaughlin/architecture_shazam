# ./spotify_api_test.py
import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


os.environ["SPOTIPY_CLIENT_ID"] = "183e7f5c41ba431898d97e617e8b9d82"
os.environ["SPOTIPY_CLIENT_SECRET"] = "c059dfd0c31440599d418b85a06a6431"

auth_manager = SpotifyClientCredentials()
sp = Spotify(auth_manager=auth_manager)

# result = sp.search(q="track:Tian Mi Mi", type="track", limit=1)
result = sp.search(q="track:Shape of You", type="track", limit=1)
track = result["tracks"]["items"][0]

artist_id = track['artists'][0]['id']
artist = sp.artist(artist_id)

print(f"Track: {track['name']}")
print(f"Artist: {track['artists'][0]['name']}")
print(f"Genres: {artist['genres']}")
print(f"Album: {track['album']['name']}")
print(f"Duration: {track['duration_ms'] // 1000} seconds")
