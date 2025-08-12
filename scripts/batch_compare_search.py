# ./scripts/batch_compare_search.py
import argparse, csv, io, time, mimetypes, hashlib
from pathlib import Path
import requests
from PIL import Image
import numpy as np

# ---------- auth ----------
def login_or_register(s, base, username, password, email):
    r = s.post(f"{base}/api/login", json={"username": username, "password": password})
    if r.ok:
        print("✓ Logged in"); return
    print("… login failed; registering")
    s.post(f"{base}/api/register", json={"username": username, "email": email, "password": password}).raise_for_status()
    s.post(f"{base}/api/login", json={"username": username, "password": password}).raise_for_status()
    print("✓ Registered + logged in")

# ---------- helpers ----------
def slug(s): return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")

def ahash_from_bytes(data: bytes, hash_size=8) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return (arr > arr.mean()).astype(np.uint8).flatten()

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))

def obj_stats_from_bytes(data: bytes):
    v=f=0
    for line in data.splitlines():
        if line.startswith(b"v "): v+=1
        elif line.startswith(b"f "): f+=1
    return {"vertices": v, "faces": f, "sha256": hashlib.sha256(data).hexdigest()}

# ---------- search + pipeline ----------
def fetch_snippet(s, base, song_name: str):
    r = s.post(f"{base}/api/track_snippet", json={"songName": song_name}, timeout=60)
    if r.status_code == 404:
        raise RuntimeError(f"No snippet for: {song_name}")
    r.raise_for_status()
    js = r.json()
    if not js.get("preview_url"):
        raise RuntimeError(f"No preview_url for: {song_name}")
    return js  # expected keys: preview_url, title, artist, album?, release?

def run_once_via_search(s, base, song_name, building_type, action, out_dir: Path, seed=None):
    # 1) lookup preview
    meta = fetch_snippet(s, base, song_name)
    title  = meta.get("title")  or song_name
    artist = meta.get("artist") or ""
    preview = meta["preview_url"]

    # 2) download preview bytes (keep in-memory)
    pr = s.get(preview, timeout=120)
    pr.raise_for_status()
    blob = pr.content
    ctype = pr.headers.get("Content-Type", "audio/mpeg")
    ext = mimetypes.guess_extension(ctype) or ".mp3"
    fake_name = f"snippet{ext}"

    # 3) /api/analyse (multipart form with file + metadata)
    files = {"file": (fake_name, io.BytesIO(blob), ctype)}
    data = {
        "title": title,
        "artist": artist,
        # album/release optional
        **({"album": meta["album"]}   if meta.get("album")   else {}),
        **({"release": meta["release"]} if meta.get("release") else {}),
        # (note: building_type is used on /api/describe, not here)
    }
    r = s.post(f"{base}/api/analyse", files=files, data=data, timeout=600)
    r.raise_for_status()
    analysis_id = r.json()["analysisId"]

    # 4) /api/describe (JSON with analysisId + building_type)
    r = s.post(f"{base}/api/describe",
               json={"analysisId": analysis_id, "building_type": building_type},
               timeout=600)
    r.raise_for_status()
    prompt_id = r.json()["promptId"]

    # 5) /api/render (JSON with promptId + action)
    body = {"promptId": prompt_id, "action": action}
    if seed is not None:
        body["seed"] = int(seed)  # only works if your server threads this through
    r = s.post(f"{base}/api/render", json=body, timeout=1800)
    r.raise_for_status()
    res = r.json()
    url = res.get("url") or res.get("imageUrl")
    filename = res.get("filename") or ("result.png" if action=="image" else "result.obj")

    # 6) download result
    d = s.get(url, stream=True, timeout=600)
    d.raise_for_status()
    data_bytes = d.content
    out_path = out_dir / filename
    out_path.write_bytes(data_bytes)
    return out_path, data_bytes, title

def main():
    ap = argparse.ArgumentParser(description="Compare N renders per song using search-based previews.")
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--email", default=None)
    ap.add_argument("--songs", required=True,
                    help="CSV with columns: song,building_type (building_type optional; defaults to house)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--action", choices=["image","model"], default="image")
    ap.add_argument("--out", default="out_search_batch")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--seed-start", type=int, default=None)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    login_or_register(s, base, args.username, args.password, args.email or f"{args.username}@example.com")

    # load song list
    todo = []
    with open(args.songs, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            song = r["song"].strip()
            if not song: continue
            todo.append({
                "song": song,
                "building_type": (r.get("building_type") or "house").lower()
            })

    for row in todo:
        song = row["song"]
        btype = row["building_type"]
        print(f"\n=== {song} ({btype}) — {args.trials} trial(s) → {args.action} ===")
        out_dir = out_root / f"{slug(song)}_{btype}_{args.action}"
        out_dir.mkdir(parents=True, exist_ok=True)

        hashes = []
        models = []
        for i in range(args.trials):
            seed = (args.seed_start + i) if args.seed_start is not None else None
            try:
                path, data_bytes, title = run_once_via_search(
                    s, base, song, btype, args.action, out_dir, seed
                )
                print(f"  ✓ Trial {i+1}: {path.name}")
                if args.action == "image":
                    hashes.append(ahash_from_bytes(data_bytes))
                else:
                    models.append(obj_stats_from_bytes(data_bytes))
            except Exception as e:
                print(f"  ✗ Trial {i+1} failed: {e}")
            time.sleep(args.sleep)

        if args.action == "image" and len(hashes) >= 2:
            dists = [hamming(hashes[i], hashes[j])
                     for i in range(len(hashes)) for j in range(i+1, len(hashes))]
            avg = sum(dists)/len(dists)
            print(f"  → avg aHash Hamming distance (64-bit): {avg:.2f}")
        elif args.action == "model" and models:
            for i, st in enumerate(models, 1):
                print(f"  • Trial {i}: V={st['vertices']} F={st['faces']} hash[:8]={st['sha256'][:8]}")

if __name__ == "__main__":
    main()
