# ./flask_server/modules/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ACRCloud
    ACRCLOUD_HOST: str
    ACRCLOUD_KEY: str
    ACRCLOUD_SECRET: str

    # Spotify
    SPOTIPY_CLIENT_ID: str
    SPOTIPY_CLIENT_SECRET: str

    # Wikipedia
    WIKI_USER_AGENT: str = "ArchitectureShazamScript/0.1 (you@example.com)"
    WIKI_API_URL: str = "https://en.wikipedia.org/w/api.php"

    # Meshy
    MESHY_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
