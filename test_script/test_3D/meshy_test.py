# ./test_script/test_3D/meshy_preview_only.py
import os, time, requests, random
from pathlib import Path

API_KEY = "*****"
BASE = "https://api.meshy.ai/openapi/v2"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _post(path: str, json: dict):
    r = requests.post(f"{BASE}{path}", json=json, headers=HEADERS, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} failed {r.status_code}: {r.text}")
    return r.json()["result"]


def _get_task(task_id: str):
    r = requests.get(f"{BASE}/text-to-3d/{task_id}", headers=HEADERS, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"GET task {task_id} failed {r.status_code}: {r.text}")
    return r.json()


def _wait(task_id: str, poll=6, timeout=60 * 20):
    t0 = time.time()
    while True:
        data = _get_task(task_id)
        status = data.get("status")
        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            return data
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Task {task_id} timed out")
        time.sleep(poll)


def _download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, headers=HEADERS, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)


def create_preview_variations(prompt: str, out_dir="meshy_out", num_variations=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_variations):
        # seed = random.randint(0, 2**31 - 1)
        seed = 1754937022
        payload = {
            "mode": "preview",  # can't use "draft" publicly
            "ai_model": "meshy-5.1",  # try their newer checkpoint
            "prompt": prompt,
            "art_style": "realistic",
            "topology": "quad",
            "target_polycount": 300_000,
            "should_remesh": True,
            "symmetry_mode": "off",
            "seed": seed,
        }

        print(f"\n=== Variation {i+1}/{num_variations} | Seed: {seed} ===")
        task_id = _post("/text-to-3d", payload)
        result = _wait(task_id)

        if result.get("status") != "SUCCEEDED":
            print(f"❌ Failed for seed {seed}")
            continue

        model_urls = result.get("model_urls") or {}
        thumbs = (result.get("preview") or {}).get("images")

        if thumbs:
            img_path = out_dir / f"{task_id}_preview.jpg"
            _download(thumbs[0], img_path)
            print(f"🖼️ saved preview image: {img_path}")

        if model_urls.get("glb"):
            glb_path = out_dir / f"{task_id}_preview.glb"
            _download(model_urls["glb"], glb_path)
            print(f"✅ saved model: {glb_path}")


if __name__ == "__main__":
    prompt = (
        "A modern, dynamic house with a bold presence, featuring a layered, diagonally-"
        "oriented massing, warm yet vibrant materials (golden-brick cladding, timber accents, "
        "and large glass panels), and bold articulation through angular lines and deep overhangs. "
        "The overall silhouette is inviting and lively."
    )
    create_preview_variations(prompt, num_variations=1)
