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


def analyse_tune(file_path: str) -> dict:
    # Load file
    try:
        # Load file at original sampling rate
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        raise RuntimeError(f"Could not load '{file_path}': {e}")

    # Preprocess: trim silence (leading and trailing) & normalise (scale peak amplitude ±1)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    # Analyse features
    tempo_info = tempo_analysis(y, sr)
    key_str = key_analysis(y, sr)
    bandwidth_info = bandwidth_analysis(y, sr)
    centroid_info = centroid_analysis(y, sr)
    rms_info = rms_analysis(y)
    mfccs_info = mfccs_analysis(y, sr)

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
        "spectral_bandwidth": bandwidth_info["bandwidth"].mean(),
        # Placeholder for future extension
        "mood": None,
        "instruments": None,
    }


def tempo_analysis(time_series, sample_rate):
    tempo_global, _ = beat_track(y=time_series, sr=sample_rate)
    tempo_global = float(tempo_global)

    local_tempo = librosa_tempo(y=time_series, sr=sample_rate, aggregate=None)
    mean_tempo = float(np.mean(local_tempo))
    median_tempo = float(np.median(local_tempo))

    return {
        "tempo_global": tempo_global,
        "tempo_mean": mean_tempo,
        "tempo_median": median_tempo,
    }


def key_analysis(time_series, sample_rate):
    # Compute CENS chroma
    chroma_cens = librosa.feature.chroma_cens(y=time_series, sr=sample_rate)
    profile = np.mean(chroma_cens, axis=1)

    # Normalise vectors
    profile_norm = profile / np.linalg.norm(profile)
    maj_norm = KRUMHANSL_MAJOR / np.linalg.norm(KRUMHANSL_MAJOR)
    min_norm = KRUMHANSL_MINOR / np.linalg.norm(KRUMHANSL_MINOR)

    # Correlate rotated profiles with templates
    maj_corr = [np.dot(np.roll(profile_norm, -i), maj_norm) for i in range(12)]
    min_corr = [np.dot(np.roll(profile_norm, -i), min_norm) for i in range(12)]

    maj_key = int(np.argmax(maj_corr))
    min_key = int(np.argmax(min_corr))
    if maj_corr[maj_key] >= min_corr[min_key]:
        return librosa.midi_to_note(maj_key + 60) + " major"
    return librosa.midi_to_note(min_key + 60) + " minor"


def bandwidth_analysis(time_series, sample_rate):
    bandwidth = librosa.feature.spectral_bandwidth(y=time_series, sr=sample_rate)
    bandwidth = np.squeeze(bandwidth)

    mean_bw = float(np.mean(bandwidth))
    median_bw = float(np.median(bandwidth))
    std_bw = float(np.std(bandwidth))

    return {
        "bandwidth": bandwidth,
        "mean_bw": mean_bw,
        "median_bw": median_bw,
        "std_bw": std_bw,
    }


def centroid_analysis(time_series, sample_rate):
    centroid = librosa.feature.spectral_centroid(y=time_series, sr=sample_rate)
    centroid = np.squeeze(centroid)
    mean_centroid = float(np.mean(centroid))
    median_centroid = float(np.median(centroid))
    std_centroid = float(np.std(centroid))

    return {
        "mean_centroid": mean_centroid,
        "median_centroid": median_centroid,
        "std_centroid": std_centroid,
    }


def rms_analysis(time_series):
    rms = librosa.feature.rms(y=time_series)
    rms = np.squeeze(rms)
    mean_rms = float(np.mean(rms))
    max_rms = float(np.max(rms))

    return {"mean_rms": mean_rms, "max_rms": max_rms}


def mfccs_analysis(time_series, sample_rate):
    mfccs = librosa.feature.mfcc(y=time_series, sr=sample_rate, n_mfcc=13)
    mfccs_means = np.mean(mfccs, axis=1)
    mfccs_stds = np.std(mfccs, axis=1)

    mfcc_summary = {}
    for i in range(13):
        mfcc_summary[f"mfcc{i+1}_mean"] = round(float(mfccs_means[i]), 2)
        mfcc_summary[f"mfcc{i+1}_std"] = round(float(mfccs_stds[i]), 2)

    return {
        "mfccs": mfccs,
        "mfccs_means": mfccs_means,
        "mfccs_stds": mfccs_stds,
        "mfcc_summary": mfcc_summary,
    }
