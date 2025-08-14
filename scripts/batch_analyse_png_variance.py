# script/batch_analyse_png_variance.py
import csv
import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image


# ---------- perceptual hashes (64-bit) ----------
def ahash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    g = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    return (a > a.mean()).astype(np.uint8).ravel()


def dhash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    # Horizontal gradient hash: compare adjacent pixels
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    diff = a[:, 1:] > a[:, :-1]
    return diff.astype(np.uint8).ravel()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ---------- color histogram (8x8x8 RGB, normalized) ----------
def color_hist(img: Image.Image, bins_per_ch: int = 8) -> np.ndarray:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    # Quantize to bins_per_ch per channel
    q = (arr // (256 // bins_per_ch)).clip(0, bins_per_ch - 1)
    idx = q[:, :, 0] * (bins_per_ch**2) + q[:, :, 1] * bins_per_ch + q[:, :, 2]
    hist = np.bincount(idx.ravel(), minlength=bins_per_ch**3).astype(np.float32)
    norm = np.linalg.norm(hist) + 1e-8
    return hist / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ---------- optional CLIP embeddings ----------
def maybe_clip_model(enable: bool):
    if not enable:
        return None, None
    try:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        model.to(device)
        return (model, preprocess, device)
    except Exception as e:
        print(f"[warn] CLIP disabled ({e})")
        return None, None


def clip_embed(img: Image.Image, model, preprocess, device) -> np.ndarray:
    import torch

    with torch.no_grad():
        t = preprocess(img).unsqueeze(0).to(device)
        feats = model.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy()


# ---------- grouping ----------
def default_group_from_name(stem: str) -> str:
    # Strip trailing _YYYYMMDD_HHMMSS if present
    m = re.match(r"^(.*?)(?:_\d{8}_\d{6})?$", stem)
    return m.group(1) if m else stem


def build_groups(files: List[Path], mode: str, regex: str) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    rx = re.compile(regex) if (mode == "regex" and regex) else None
    for p in files:
        if mode == "parent":
            key = p.parent.name
        elif mode == "regex" and rx:
            m = rx.search(p.name)
            key = m.group(1) if m else p.stem
        else:
            key = default_group_from_name(p.stem)
        groups.setdefault(key, []).append(p)
    return groups


# ---------- pairwise helpers ----------
def pairwise_stats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float32)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, sd


def pairwise_distances_hash(hashes: List[np.ndarray]) -> List[float]:
    out = []
    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(hamming(hashes[i], hashes[j]))
    return out


def pairwise_similarities(vectors: List[np.ndarray]) -> List[float]:
    out = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(cosine_sim(vectors[i], vectors[j]))
    return out


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Measure intra-group similarity for PNG images and rank repeatability."
    )
    ap.add_argument("root", help="Directory to scan for .png")
    ap.add_argument(
        "--group-by",
        choices=["auto", "parent", "regex"],
        default="auto",
        help="auto=strip _YYYYMMDD_HHMMSS; parent=group by folder; regex=use --group-regex",
    )
    ap.add_argument(
        "--group-regex",
        default=r"^(.*?)(?:_\d{8}_\d{6})?",
        help="Regex with one capturing group for the group key (used when --group-by=regex)",
    )
    ap.add_argument(
        "--max-per-group",
        type=int,
        default=200,
        help="Optional cap per group to speed up",
    )
    ap.add_argument("--csv", default="png_group_metrics.csv")
    ap.add_argument(
        "--clip",
        action="store_true",
        help="Also compute CLIP distances (requires open_clip_torch)",
    )

    # Weights for combined repeatability score (default: structure heavy)
    ap.add_argument(
        "--w-ahash", type=float, default=0.4, help="Weight for aHash similarity"
    )
    ap.add_argument(
        "--w-dhash", type=float, default=0.4, help="Weight for dHash similarity"
    )
    ap.add_argument(
        "--w-palette",
        type=float,
        default=0.2,
        help="Weight for palette (color) similarity",
    )
    ap.add_argument(
        "--w-clip",
        type=float,
        default=0.0,
        help="Weight for CLIP similarity (if --clip)",
    )

    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.png"))
    if not files:
        print("No .png files found.")
        return

    groups = build_groups(files, args.group_by, args.group_regex)

    # Prepare optional CLIP
    clip_tuple = None
    if args.clip:
        clip_tuple = maybe_clip_model(True)
        if clip_tuple[0] is None:
            clip_tuple = None
            if args.w_clip > 0:
                print(
                    "[warn] --w-clip > 0 but CLIP is unavailable; ignoring CLIP weight"
                )

    rows = []
    print(f"Found {len(files)} PNGs in {len(groups)} group(s).")
    for key, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        if args.max_per_group and len(paths) > args.max_per_group:
            paths = paths[: args.max_per_group]

        ah_list, dh_list, hist_list, clip_list = [], [], [], []

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            ah_list.append(ahash(img))
            dh_list.append(dhash(img))
            hist_list.append(color_hist(img))
            if clip_tuple:
                model, preprocess, device = clip_tuple
                clip_list.append(clip_embed(img, model, preprocess, device))

        n = min(len(ah_list), len(dh_list), len(hist_list))
        if n < 2:
            continue

        # Pairwise metrics
        ah_dists = pairwise_distances_hash(ah_list)  # 0..64 (lower better)
        dh_dists = pairwise_distances_hash(dh_list)  # 0..64 (lower better)
        pal_sims = pairwise_similarities(hist_list)  # 0..1 (higher better)

        ah_mean, ah_sd = pairwise_stats(ah_dists)
        dh_mean, dh_sd = pairwise_stats(dh_dists)
        pal_mean, pal_sd = pairwise_stats(pal_sims)

        # Convert to similarities for combined score
        ah_sim = clamp01(1.0 - (ah_mean / 64.0))
        dh_sim = clamp01(1.0 - (dh_mean / 64.0))
        pal_sim = clamp01(pal_mean)

        clip_sim = None
        clip_dist_mean = None
        clip_dist_sd = None
        if clip_tuple and len(clip_list) >= 2:
            clip_sims = pairwise_similarities(clip_list)  # cosine similarity
            clip_dists = [1.0 - s for s in clip_sims]  # convert to distance
            clip_dist_mean, clip_dist_sd = pairwise_stats(clip_dists)
            clip_sim = clamp01(
                1.0 - (clip_dist_mean if np.isfinite(clip_dist_mean) else 1.0)
            )

        # Combined score (0..100). Higher = more repeatable.
        wa, wd, wp = args.w_ahash, args.w_dhash, args.w_palette
        wc = args.w_clip if (clip_tuple and clip_sim is not None) else 0.0
        wsum = max(1e-8, wa + wd + wp + wc)
        combined = (
            100.0
            * (wa * ah_sim + wd * dh_sim + wp * pal_sim + wc * (clip_sim or 0.0))
            / wsum
        )

        row = {
            "group": key,
            "n_images": n,
            "ahash_mean_hamming": round(ah_mean, 3),
            "ahash_sd": round(ah_sd, 3),
            "dhash_mean_hamming": round(dh_mean, 3),
            "dhash_sd": round(dh_sd, 3),
            "palette_cosine_mean": round(pal_mean, 4),
            "palette_cosine_sd": round(pal_sd, 4),
            # Similarities (0..1)
            "ahash_similarity": round(ah_sim, 4),
            "dhash_similarity": round(dh_sim, 4),
            "palette_similarity": round(pal_sim, 4),
            # Combined score
            "repeatability_score": round(combined, 2),
        }

        if clip_tuple and clip_sim is not None:
            row["clip_dist_mean"] = round(clip_dist_mean, 4)
            row["clip_dist_sd"] = round(clip_dist_sd, 4)
            row["clip_similarity"] = round(clip_sim, 4)

        rows.append(row)

        # Console preview
        preview = (
            f"[{key}] n={n} | aHash Δ={row['ahash_mean_hamming']}±{row['ahash_sd']} "
            f"| dHash Δ={row['dhash_mean_hamming']}±{row['dhash_sd']} "
            f"| Palette sim={row['palette_similarity']} "
            f"| Score={row['repeatability_score']:.1f}"
        )
        if "clip_similarity" in row:
            preview += f" | CLIP sim={row['clip_similarity']}"
        print(preview)

    # Write CSV (sorted by score desc)
    if rows:
        rows_sorted = sorted(rows, key=lambda r: r["repeatability_score"], reverse=True)
        fieldnames = list(rows_sorted[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_sorted)
        print(f"\nWrote: {args.csv}")

        # Print a compact ranking
        print("\nOverall repeatability ranking (top 20):")
        for r in rows_sorted[:20]:
            print(f"  {r['group']}: {r['repeatability_score']:.1f} (n={r['n_images']})")
    else:
        print("No groups with ≥2 valid images.")


if __name__ == "__main__":
    main()
