# ./scripts/image_noise_metrics.py
"""
Compute simple no-reference "noise" proxies for images and write CSVs.

Metrics per image:
- lap_var: Variance of Laplacian response (higher ≈ more high-freq energy / grain)
- hf_ratio: Fraction of FFT energy outside a low-frequency circle (higher ≈ more high-freq)
- resid_rms: RMS of (image - GaussianBlur(image)) (higher ≈ more high-freq detail/noise)
- noise_index: 0-100 composite (robustly scaled + averaged)

Grouping:
- auto: group by filename stem with trailing _YYYYMMDD_HHMMSS removed
- parent: group by parent folder name
- regex: use --group-regex with a single capturing group
- all: put everything in one group "ALL"

Usage example:
    python scripts/image_noise_metrics.py ./out_runs \
    --pattern "*.png" \
    --group-by auto \
    --per-image-out noise_per_image.csv \
    --group-out noise_groups.csv
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


# -------------------- helpers --------------------
def default_group_from_name(stem: str) -> str:
    # Strip trailing _YYYYMMDD_HHMMSS if present
    m = re.match(r"^(.*?)(?:_\d{8}_\d{6})?$", stem)
    return m.group(1) if m else stem


def build_groups(files: List[Path], mode: str, regex: str) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    rx = re.compile(regex) if (mode == "regex" and regex) else None

    if mode == "all":
        groups["ALL"] = list(files)
        return groups

    for p in files:
        if mode == "parent":
            key = p.parent.name
        elif mode == "regex" and rx:
            m = rx.search(p.name)
            key = m.group(1) if m else p.stem
        else:  # auto
            key = default_group_from_name(p.stem)
        groups.setdefault(key, []).append(p)
    return groups


def robust_scale(
    values: List[float], lo_pct=5.0, hi_pct=95.0
) -> Tuple[List[float], float, float]:
    """Scale to [0,1] using robust min/max from percentiles, return scaled list and (lo,hi)."""
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo:
        # degenerate case
        scaled = [0.5 for _ in values]
        return scaled, lo, hi
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return scaled.tolist(), lo, hi


# -------------------- core metrics --------------------
def to_gray_f32(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    a = np.asarray(g, dtype=np.float32) / 255.0
    return a


def laplacian_var(gray: np.ndarray) -> float:
    """
    Variance of a 3x3 Laplacian response (no SciPy/OpenCV).
    Kernel:
      0  1  0
      1 -4  1
      0  1  0
    """
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(gray, 1, mode="reflect")
    # manual 3x3 conv by slicing
    acc = (
        k[0, 0] * pad[:-2, :-2]
        + k[0, 1] * pad[:-2, 1:-1]
        + k[0, 2] * pad[:-2, 2:]
        + k[1, 0] * pad[1:-1, :-2]
        + k[1, 1] * pad[1:-1, 1:-1]
        + k[1, 2] * pad[1:-1, 2:]
        + k[2, 0] * pad[2:, :-2]
        + k[2, 1] * pad[2:, 1:-1]
        + k[2, 2] * pad[2:, 2:]
    )
    return float(acc.var())


def high_freq_ratio(gray: np.ndarray, lowpass_frac: float = 0.10) -> float:
    """
    Ratio of FFT energy outside a centered low-frequency circle.
    lowpass_frac: fraction of min(H,W) used as radius for the low-frequency mask (e.g., 0.10 = 10%).
    """
    H, W = gray.shape
    F = np.fft.fft2(gray)
    S = np.fft.fftshift(F)
    mag2 = (S.real * S.real) + (S.imag * S.imag)

    cy, cx = H // 2, W // 2
    r0 = lowpass_frac * min(H, W)
    y, x = np.ogrid[:H, :W]
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    mask_low = dist2 <= (r0 * r0)

    low_energy = float(mag2[mask_low].sum())
    total = float(mag2.sum()) + 1e-12
    hf = 1.0 - (low_energy / total)
    return hf


def residual_rms(gray: np.ndarray, blur_sigma: float = 1.0) -> float:
    """
    RMS of (image - GaussianBlur(image)).
    Uses Pillow's GaussianBlur; sigma ~ radius.
    """
    # go through PIL for blur
    g8 = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
    b = g8.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    b_arr = np.asarray(b, dtype=np.float32) / 255.0

    resid = gray - b_arr
    return float(np.sqrt((resid * resid).mean()))


# -------------------- CSV writing --------------------
def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Estimate noise metrics for images and output CSVs."
    )
    ap.add_argument("root", help="Directory to scan")
    ap.add_argument(
        "--pattern", default="*.png", help='Glob pattern, e.g. "*.png" or "*.jpg"'
    )
    ap.add_argument(
        "--recursive", action="store_true", help="Recurse into subfolders (uses rglob)"
    )
    ap.add_argument(
        "--group-by", choices=["auto", "parent", "regex", "all"], default="auto"
    )
    ap.add_argument(
        "--group-regex",
        default=r"^(.*?)(?:_\d{8}_\d{6})?",
        help="Regex with one capturing group when --group-by=regex",
    )
    ap.add_argument(
        "--lowpass-frac",
        type=float,
        default=0.10,
        help="Low-frequency radius fraction for FFT ratio",
    )
    ap.add_argument(
        "--blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian blur sigma for residual RMS",
    )
    ap.add_argument(
        "--per-image-out",
        default="noise_per_image.csv",
        help="CSV file for per-image metrics",
    )
    ap.add_argument(
        "--group-out",
        default="noise_groups.csv",
        help="CSV file for grouped summary (optional)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted((root.rglob if args.recursive else root.glob)(args.pattern))
    if not files:
        print("No images found for pattern:", args.pattern)
        return

    groups = build_groups(files, args.group_by, args.group_regex)

    # Pass 1: compute per-image raw metrics
    per_image_rows = []
    all_lap, all_hf, all_resid = [], [], []

    for gkey, paths in sorted(groups.items()):
        for p in paths:
            try:
                im = Image.open(p).convert("RGB")
                w, h = im.size
                gray = to_gray_f32(im)

                lv = laplacian_var(gray)
                hf = high_freq_ratio(gray, args.lowpass_frac)
                rr = residual_rms(gray, args.blur_sigma)

                per_image_rows.append(
                    {
                        "group": gkey,
                        "path": str(p),
                        "width": w,
                        "height": h,
                        "lap_var": round(lv, 6),
                        "hf_ratio": round(hf, 6),
                        "resid_rms": round(rr, 6),
                    }
                )
                all_lap.append(lv)
                all_hf.append(hf)
                all_resid.append(rr)
            except Exception as e:
                print(f"[warn] failed on {p.name}: {e}")

    if not per_image_rows:
        print("No valid images processed.")
        return

    # Pass 2: robust scaling + composite noise index
    lap_scaled, lap_lo, lap_hi = robust_scale(all_lap)
    hf_scaled, hf_lo, hf_hi = robust_scale(all_hf)
    rr_scaled, rr_lo, rr_hi = robust_scale(all_resid)

    for idx, row in enumerate(per_image_rows):
        s_lap = lap_scaled[idx]
        s_hf = hf_scaled[idx]
        s_rr = rr_scaled[idx]
        noise_index = 100.0 * (s_lap + s_hf + s_rr) / 3.0
        row.update(
            {
                "lap_var_scaled": round(s_lap, 6),
                "hf_ratio_scaled": round(s_hf, 6),
                "resid_rms_scaled": round(s_rr, 6),
                "noise_index": round(noise_index, 2),
            }
        )

    write_csv(Path(args.per_image_out), per_image_rows)
    print(f"Wrote per-image CSV: {args.per_image_out}")

    # Optional group summary
    if args.group_out:
        by_group: Dict[str, List[dict]] = {}
        for r in per_image_rows:
            by_group.setdefault(r["group"], []).append(r)

        group_rows = []
        for gkey, rows in sorted(by_group.items()):

            def stats(key):
                vals = np.asarray([r[key] for r in rows], dtype=np.float32)
                return float(vals.mean()), (
                    float(vals.std(ddof=1))
                    if len(vals) > 1
                    else (float(vals.mean()), 0.0)
                )

            n = len(rows)
            lap_m, lap_sd = stats("lap_var")
            hf_m, hf_sd = stats("hf_ratio")
            rr_m, rr_sd = stats("resid_rms")
            ni_m, ni_sd = stats("noise_index")

            group_rows.append(
                {
                    "group": gkey,
                    "n_images": n,
                    "lap_var_mean": round(lap_m, 6),
                    "lap_var_sd": round(lap_sd, 6),
                    "hf_ratio_mean": round(hf_m, 6),
                    "hf_ratio_sd": round(hf_sd, 6),
                    "resid_rms_mean": round(rr_m, 6),
                    "resid_rms_sd": round(rr_sd, 6),
                    "noise_index_mean": round(ni_m, 2),
                    "noise_index_sd": round(ni_sd, 2),
                }
            )

        write_csv(Path(args.group_out), group_rows)
        print(f"Wrote group summary CSV: {args.group_out}")


if __name__ == "__main__":
    main()
