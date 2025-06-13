# ./tune_analysis.py
import librosa
from librosa.beat import beat_track
from librosa.feature.rhythm import tempo as librosa_tempo
import numpy as np

KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def analyse_features(file_path: str) -> dict:
    """
    Load an audio file, extract musical features, and derive qualitative timbre
    and loudness descriptors.
    Returns a dict of features.
    """
    # Load at native sampling rate
    y, sr = librosa.load(file_path, sr=None)

    # Trim silence and normalise
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    # Feature analyses
    tempo_info = tempo_analysis(y, sr)
    key_str = key_analysis(y, sr)
    bandwidth_info = bandwidth_analysis(y, sr)
    centroid_info = centroid_analysis(y, sr)
    rms_info = rms_analysis(y)
    mfccs_info = mfccs_analysis(y, sr)
    timbre, loudness = describe_timbre_and_loudness(y, sr)

    return {
        "tempo_global": round(tempo_info["tempo_global"], 2),
        "tempo_mean_local": round(tempo_info["tempo_mean"], 2),
        "tempo_median_local": round(tempo_info["tempo_median"], 2),
        "mean_bandwidth": round(bandwidth_info["mean_bw"], 2),
        "median_bandwidth": round(bandwidth_info["median_bw"], 2),
        "std_bandwidth": round(bandwidth_info["std_bw"], 2),
        "mean_centroid": round(centroid_info["mean_centroid"], 2),
        "median_centroid": round(centroid_info["median_centroid"], 2),
        "std_centroid": round(centroid_info["std_centroid"], 2),
        "mean_rms": round(rms_info["mean_rms"], 4),
        "max_rms": round(rms_info["max_rms"], 4),
        **mfccs_info["mfcc_summary"],
        "key": key_str,
        "timbre": timbre,
        "loudness": loudness,
    }


def tempo_analysis(time_series: np.ndarray, sample_rate: int) -> dict:
    tempo_global, _ = beat_track(y=time_series, sr=sample_rate)
    tempo_global = float(tempo_global)
    local_tempo = librosa_tempo(y=time_series, sr=sample_rate, aggregate=None)
    return {
        "tempo_global": tempo_global,
        "tempo_mean": float(np.mean(local_tempo)),
        "tempo_median": float(np.median(local_tempo)),
    }


def key_analysis(time_series: np.ndarray, sample_rate: int) -> str:
    chroma = librosa.feature.chroma_cens(y=time_series, sr=sample_rate)
    profile = np.mean(chroma, axis=1)
    profile_norm = profile / np.linalg.norm(profile)
    maj_norm = KRUMHANSL_MAJOR / np.linalg.norm(KRUMHANSL_MAJOR)
    min_norm = KRUMHANSL_MINOR / np.linalg.norm(KRUMHANSL_MINOR)
    maj_corr = [np.dot(np.roll(profile_norm, -i), maj_norm) for i in range(12)]
    min_corr = [np.dot(np.roll(profile_norm, -i), min_norm) for i in range(12)]
    if max(maj_corr) >= max(min_corr):
        return librosa.midi_to_note(int(np.argmax(maj_corr)) + 60) + " major"
    else:
        return librosa.midi_to_note(int(np.argmax(min_corr)) + 60) + " minor"


def bandwidth_analysis(time_series: np.ndarray, sample_rate: int) -> dict:
    bw = librosa.feature.spectral_bandwidth(y=time_series, sr=sample_rate)
    bw = np.squeeze(bw)
    return {
        "bandwidth": bw,
        "mean_bw": float(np.mean(bw)),
        "median_bw": float(np.median(bw)),
        "std_bw": float(np.std(bw)),
    }


def centroid_analysis(time_series: np.ndarray, sample_rate: int) -> dict:
    cent = librosa.feature.spectral_centroid(y=time_series, sr=sample_rate)
    cent = np.squeeze(cent)
    return {
        "mean_centroid": float(np.mean(cent)),
        "median_centroid": float(np.median(cent)),
        "std_centroid": float(np.std(cent)),
    }


def rms_analysis(time_series: np.ndarray) -> dict:
    rms = librosa.feature.rms(y=time_series)
    rms = np.squeeze(rms)
    return {"mean_rms": float(np.mean(rms)), "max_rms": float(np.max(rms))}


def mfccs_analysis(time_series: np.ndarray, sample_rate: int) -> dict:
    mfccs = librosa.feature.mfcc(y=time_series, sr=sample_rate, n_mfcc=13)
    means = np.mean(mfccs, axis=1)
    stds = np.std(mfccs, axis=1)
    summary = {f"mfcc{i+1}_mean": round(float(means[i]), 2) for i in range(13)}
    summary.update({f"mfcc{i+1}_std": round(float(stds[i]), 2) for i in range(13)})
    return {"mfccs": mfccs, "mfcc_summary": summary}


def describe_timbre_and_loudness(time_series: np.ndarray, sample_rate: int) -> tuple:
    """
    Derive qualitative timbre ('bright','warm','dark') from spectral centroid
    and loudness ('loud','medium','soft') from RMS in dB.
    """
    cent = librosa.feature.spectral_centroid(y=time_series, sr=sample_rate)[0]
    mean_cent = float(np.mean(cent))
    rms_vals = librosa.feature.rms(y=time_series)[0]
    rms_db = librosa.amplitude_to_db(rms_vals, ref=np.max)
    mean_rms_db = float(np.mean(rms_db))
    if mean_cent > 3000:
        timbre = "bright"
    elif mean_cent > 1500:
        timbre = "warm"
    else:
        timbre = "dark"
    if mean_rms_db > -10:
        loudness = "loud"
    elif mean_rms_db > -20:
        loudness = "medium"
    else:
        loudness = "soft"
    return timbre, loudness
