# ./flask_server/modules/services/acrcloud_service.py
import json
from typing import Any
from acrcloud.recognizer import ACRCloudRecognizer
from flask import current_app


class ACRCloudService:
    def __init__(self, file_path: str) -> None:
        cfg = {
            "host": current_app.config["APP_SETTINGS"].ACRCLOUD_HOST,
            "access_key": current_app.config["APP_SETTINGS"].ACRCLOUD_KEY,
            "access_secret": current_app.config["APP_SETTINGS"].ACRCLOUD_SECRET,
            "timeout": 10,
        }
        self.recogniser = ACRCloudRecognizer(cfg)
        self.file_path = file_path
        self.offsets = [45, 60, 75, 90]
        self.clip_length = 10

    def best_recognition(self) -> dict[str, Any]:
        """
        Read the entire MP3 into memory and, for each offset (in seconds),
        call recognize_by_filebuffer to get ACRCloud's best match for that window.
        Returns the single track dict with the highest score (or None).
        """
        # load file once
        with open(self.file_path, "rb") as f:
            mp3_buf = f.read()

        best_track = {}
        best_score = -1

        for off in self.offsets:
            # fingerprint the [off, off+length_sec) slice
            resp = self.recogniser.recognize_by_filebuffer(
                mp3_buf, off, self.clip_length
            )
            data = json.loads(resp)

            # skip errors or no‐matches
            if data["status"]["code"] != 0 or not data["metadata"].get("music"):
                continue

            # find the top‐scoring track in this slice
            candidate = max(data["metadata"]["music"], key=lambda t: t["score"])
            if candidate.get("score", -1) > best_score:
                best_score = candidate["score"]
                best_track: dict = candidate

        if best_track and best_track.get("score", 0) > 80:
            return best_track

        return {}
