# ./build_description.py
from .librosa_analysis import analyse_features
from .genres_analysis import get_genre
from .instruments_analysis import get_instruments

# from yamnet_analysis import analyse_yamnet


def get_features(tune: str) -> str:
    librosa_info = analyse_features(tune)
    print(librosa_info, "\n")

    # yamnet_info = analyse_yamnet(tune)
    # print(yamnet_info, "\n")

    genre_info = get_genre(tune)
    print(genre_info, "\n")

    instrument_info = get_instruments(file_path=tune)
    print(instrument_info, "\n")

    return librosa_info, genre_info, instrument_info
