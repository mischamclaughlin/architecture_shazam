# ./acrcloud_test.py
import os
import json
from dotenv import load_dotenv
from acrcloud.recognizer import ACRCloudRecognizer


load_dotenv()

# Configuration
config = {
    "host": os.environ["ACRCLOUD_HOST"],
    "access_key": os.environ["ACRCLOUD_KEY"],
    "access_secret": os.environ["ACRCLOUD_SECRET"],
    "timeout": 10,  # seconds
}
recogniser = ACRCloudRecognizer(config)

# Helpers


def best_recognition(
    path: str, offsets: list[int], length_sec: int = 15
) -> dict | None:
    """
    Read the entire MP3 into memory and, for each offset (in seconds),
    call recognize_by_filebuffer to get ACRCloud's best match for that window.
    Returns the single track dict with the highest score (or None).
    """
    # load file once
    with open(path, "rb") as f:
        mp3_buf = f.read()

    best_track = None
    best_score = -1

    for off in offsets:
        # fingerprint the [off, off+length_sec) slice
        resp = recogniser.recognize_by_filebuffer(mp3_buf, off, length_sec)
        data = json.loads(resp)

        # skip errors or no‐matches
        if data["status"]["code"] != 0 or not data["metadata"].get("music"):
            continue

        # find the top‐scoring track in this slice
        candidate = max(data["metadata"]["music"], key=lambda t: t["score"])
        if candidate["score"] > best_score:
            best_score = candidate["score"]
            best_track = candidate

    return best_track


if __name__ == "__main__":
    mp3_path = "./tunes/the_lion_king.mp3"
    # try offsets around the vocal hook / chorus
    offsets = [45, 60, 75, 90]  # in seconds
    clip_length = 10  # seconds to fingerprint

    track = best_recognition(mp3_path, offsets, clip_length)

    if track:
        print(track)
        title = track.get("title", "—")
        artists = ", ".join(a["name"] for a in track.get("artists", [])) or "—"
        album = track.get("album", {}).get("name", "—")
        release = track.get("release_date", "—")
        genres = ", ".join(g["name"] for g in track.get("genres", [])) or "—"
        score = track.get("score", 0)

        print(f"Title:    {title}")
        print(f"Artists:  {artists}")
        print(f"Album:    {album}")
        print(f"Released: {release}")
        print(f"Genres:   {genres}")
        print(f"Score:    {score}/100")
    else:
        print("No confident match found.")
