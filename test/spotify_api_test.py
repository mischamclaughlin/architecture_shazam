# ./spotify_api_test.py
import os
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


load_dotenv()


client_id = os.environ["SPOTIPY_CLIENT_ID"]
client_secret = os.environ["SPOTIPY_CLIENT_SECRET"]


auth_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = Spotify(auth_manager=auth_manager)

result = sp.search(q="track:Tian Mi Mi", type="track", limit=1)
track = result["tracks"]["items"][0]

artist_id = track["artists"][0]["id"]
artist = sp.artist(artist_id)

print(f"Track: {track['name']}")
print(f"Artist: {track['artists'][0]['name']}")
print(f"Genres: {artist['genres']}")
print(f"Album: {track['album']['name']}")
print(f"Duration: {track['duration_ms'] // 1000} seconds")
