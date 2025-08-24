# ./scripts/batch_generate_freeze_prompt.py
"""
Fixed-prompt trials:
- For each (song, building_type) in a CSV, call /api/describe ONCE to get a promptId
- Then call /api/render N times with freeze=True (and optional --steps)
- Save all outputs and compute aHash repeatability for images
- Write a per-group summary CSV

CSV format:
  song,building_type
  Sade - Smooth Operator,house
  Sade - Smooth Operator,skyscraper
  ...

Usage example:
  python scripts/fixed_prompt_trials.py \
    --base http://127.0.0.1:8000 \
    --username demo --password secret --email demo@example.com \
    --songs ./songs.csv \
    --trials 10 --steps 25 --action image \
    --out ./out_fixed_prompt --sleep 0.5
"""
import argparse, csv, io, time, mimetypes
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import requests
from PIL import Image
import numpy as np


# ---------- auth ----------
def login_or_register(
    s: requests.Session, base: str, username: str, password: str, email: str
) -> None:
    r = s.post(f"{base}/api/login", json={"username": username, "password": password})
    if r.ok:
        print("✓ Logged in")
        return
    print("… login failed; registering")
    s.post(
        f"{base}/api/register",
        json={"username": username, "email": email, "password": password},
    ).raise_for_status()
    s.post(
        f"{base}/api/login", json={"username": username, "password": password}
    ).raise_for_status()
    print("✓ Registered + logged in")


# ---------- helpers ----------
def slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")


def ahash_from_bytes(data: bytes, hash_size: int = 8) -> np.ndarray:
    img = (
        Image.open(io.BytesIO(data))
        .convert("L")
        .resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    )
    arr = np.asarray(img, dtype=np.float32)
    return (arr > arr.mean()).astype(np.uint8).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ---------- pipeline calls ----------
def fetch_snippet(s: requests.Session, base: str, song_name: str) -> Dict[str, Any]:
    r = s.post(f"{base}/api/track_snippet", json={"songName": song_name}, timeout=60)
    if r.status_code == 404:
        raise RuntimeError(f"No snippet for: {song_name}")
    r.raise_for_status()
    js = r.json()
    if not js.get("preview_url"):
        raise RuntimeError(f"No preview_url for: {song_name}")
    return js  # keys: preview_url, title, artist, album?, release?


def start_pipeline_once(
    s: requests.Session, base: str, song: str, building_type: str
) -> Tuple[str, str]:
    """
    Returns (prompt_id, display_title)
    """
    meta = fetch_snippet(s, base, song)
    title = meta.get("title") or song
    artist = meta.get("artist") or ""
    preview = meta["preview_url"]

    # download preview bytes
    pr = s.get(preview, timeout=120)
    pr.raise_for_status()
    blob = pr.content
    ctype = pr.headers.get("Content-Type") or "audio/mpeg"
    ext = mimetypes.guess_extension(ctype) or ".mp3"

    # /api/analyse (multipart form with file + metadata)
    files = {"file": (f"snippet{ext}", io.BytesIO(blob), ctype)}
    data = {
        "title": title,
        "artist": artist,
        **({"album": meta["album"]} if meta.get("album") else {}),
        **({"release": meta["release"]} if meta.get("release") else {}),
    }
    r = s.post(f"{base}/api/analyse", files=files, data=data, timeout=600)
    r.raise_for_status()
    analysis_id = r.json()["analysisId"]

    # /api/describe (JSON with analysisId + building_type)
    r = s.post(
        f"{base}/api/describe",
        json={"analysisId": analysis_id, "building_type": building_type},
        timeout=600,
    )
    r.raise_for_status()
    pid = r.json()["promptId"]
    return pid, title


def render_once(
    s: requests.Session,
    base: str,
    prompt_id: str,
    action: str,
    freeze: bool,
    steps: Optional[int],
) -> Tuple[Dict[str, Any], bytes]:
    """
    Returns (response_json, file_bytes)
    Expects server to accept {promptId, action, freeze, steps?} and return {url, filename, ...}
    """
    body: Dict[str, Any] = {
        "promptId": prompt_id,
        "action": action,
        "freeze": bool(freeze),
    }
    if steps is not None:
        body["steps"] = int(steps)

    r = s.post(f"{base}/api/render", json=body, timeout=1800)
    r.raise_for_status()
    js = r.json()
    url = js.get("url") or js.get("imageUrl")
    if not url:
        raise RuntimeError("render: missing url in response")
    d = s.get(url, stream=True, timeout=600)
    d.raise_for_status()
    return js, d.content


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Describe once per (song,building_type) then render N frozen-prompt trials."
    )
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--email", default=None)
    ap.add_argument(
        "--songs", required=True, help="CSV with columns: song,building_type"
    )
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument(
        "--steps", type=int, default=None, help="Inference steps to pass to /api/render"
    )
    ap.add_argument("--action", choices=["image", "model"], default="image")
    ap.add_argument("--out", default="out_fixed_prompt")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--summary", default="fixed_prompt_summary.csv")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    login_or_register(
        s,
        base,
        args.username,
        args.password,
        args.email or f"{args.username}@example.com",
    )

    # Load CSV
    todo: List[Dict[str, str]] = []
    with open(args.songs, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            song = (r.get("song") or "").strip()
            if not song:
                continue
            todo.append(
                {
                    "song": song,
                    "building_type": (r.get("building_type") or "house")
                    .strip()
                    .lower(),
                }
            )

    # Summary rows
    summary_rows: List[Dict[str, Any]] = []

    for row in todo:
        song = row["song"]
        btype = row["building_type"]
        group_key = f"{slug(song)}_{btype}_{args.action}"
        out_dir = out_root / group_key
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== {song} ({btype}) — describe once, then {args.trials} frozen renders → {args.action} ==="
        )

        # 1) One describe → promptId
        try:
            prompt_id, title = start_pipeline_once(s, base, song, btype)
        except Exception as e:
            print(f"  ✗ describe failed: {e}")
            continue

        # 2) Trials with freeze=True
        hashes: List[np.ndarray] = []
        captured_prompt: Optional[str] = None
        captured_steps = args.steps

        for i in range(args.trials):
            try:
                js, data_bytes = render_once(
                    s, base, prompt_id, args.action, freeze=True, steps=args.steps
                )

                # try to capture prompt once if server returns it
                if captured_prompt is None:
                    captured_prompt = js.get("sdxl_prompt") or js.get("prompt") or None
                    if captured_prompt:
                        (out_dir / "prompt.txt").write_text(
                            captured_prompt, encoding="utf-8"
                        )

                # filename (prefix trial number to avoid collisions)
                filename = js.get("filename") or (
                    f"result.png" if args.action == "image" else "result.obj"
                )
                filename = f"t{i+1:02d}_{filename}"
                (out_dir / filename).write_bytes(data_bytes)
                print(f"  ✓ Trial {i+1}: {filename}")

                if args.action == "image":
                    hashes.append(ahash_from_bytes(data_bytes))
            except Exception as e:
                print(f"  ✗ Trial {i+1} failed: {e}")
            time.sleep(args.sleep)

        # 3) Metrics for images
        if args.action == "image" and len(hashes) >= 2:
            dists = [
                hamming(hashes[i], hashes[j])
                for i in range(len(hashes))
                for j in range(i + 1, len(hashes))
            ]
            avg = float(sum(dists) / len(dists))
            print(f"  → avg aHash Hamming distance (64-bit): {avg:.2f}")

            summary_rows.append(
                {
                    "group": group_key,
                    "song": song,
                    "building_type": btype,
                    "trials": len(hashes),
                    "steps": captured_steps if captured_steps is not None else "",
                    "avg_ahash_hamming": round(avg, 3),
                    "prompt_captured": "yes" if captured_prompt else "no",
                }
            )
        else:
            summary_rows.append(
                {
                    "group": group_key,
                    "song": song,
                    "building_type": btype,
                    "trials": args.trials,
                    "steps": captured_steps if captured_steps is not None else "",
                    "avg_ahash_hamming": "",
                    "prompt_captured": "yes" if captured_prompt else "no",
                }
            )

    # Write summary CSV
    if summary_rows:
        summary_path = out_root / args.summary
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote summary: {summary_path.as_posix()}")
    else:
        print("\nNo successful groups to summarize.")


if __name__ == "__main__":
    main()
