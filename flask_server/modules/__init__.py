# ./flask_server/modules/__init__.py

# API services
from .services.acrcloud_service import ACRCloudService
from .services.musicbrainz_service import MusicBrainzService
from .services.wiki_service import WikiService
from .services.spotify_service import SpotifyService
from .services.itunes_service import ITunesService

# Analysis Tools
from .analysis.features import AudioFeatureExtractor
from .analysis.classifiers import GenreClassifier, InstrumentClassifier

# Generators classes
from .generators.description_generation import GenerateLLMDescription
from .generators.image_generation import GenerateImage
from .generators.model_generation import Generate3d

# Helper Functions
from .helpers import (
    get_features,
    get_song,
    get_origin,
    get_genres,
    get_instruments,
    get_spotify_search,
    get_itunes_search,
)
