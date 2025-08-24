# ./scripts/rank_repeatability.py
import argparse, csv, sys
from pathlib import Path

def to_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser(
        description="Rank per-group repeatability by lift over a baseline row (e.g., ALL)."
    )
    ap.add_argument("-i", "--input", required=True, help="Input CSV (from png variance script)")
    ap.add_argument("-o", "--output", default=None, help="Output CSV (defaults to <input>.ranked.csv)")
    ap.add_argument("--score-col", default="repeatability_score",
                    help="Column to rank by (default: repeatability_score)")
    ap.add_argument("--baseline-group", default="ALL",
                    help="Group name to use as baseline (default: ALL)")
    ap.add_argument("--baseline-score", type=float, default=None,
                    help="Override baseline score manually (skips baseline lookup)")
    ap.add_argument("--min-images", type=int, default=2,
                    help="Minimum images per group to keep (default: 2)")
    ap.add_argument("--top", type=int, default=25, help="Print top N to console (default: 25)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"Input not found: {in_path}")

    rows = []
    with open(in_path, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        sys.exit("No rows found in CSV.")

    # Determine baseline score
    if args.baseline_score is not None:
        baseline_score = float(args.baseline_score)
        baseline_found = f"(override {baseline_score})"
    else:
        baseline_row = next((r for r in rows if r.get("group") == args.baseline_group), None)
        if not baseline_row:
            sys.exit(f"Baseline group '{args.baseline_group}' not found. "
                     f"Use --baseline-score to override or include an ALL row.")
        baseline_score = to_float(baseline_row.get(args.score_col))
        if baseline_score != baseline_score:  # NaN check
            sys.exit(f"Baseline row found but '{args.score_col}' is missing/not numeric.")
        baseline_found = f"{args.baseline_group} = {baseline_score:.4f}"

    # Build ranked list (exclude baseline row, enforce min-images)
    out_rows = []
    for r in rows:
        if r.get("group") == args.baseline_group and args.baseline_score is None:
            continue
        n_images = int(r.get("n_images", "0") or 0)
        if n_images < args.min_images:
            continue
        score = to_float(r.get(args.score_col))
        if score != score:  # NaN
            continue
        lift = score - baseline_score
        r2 = dict(r)  # keep original columns
        r2["baseline_group"] = args.baseline_group
        r2["baseline_score"] = f"{baseline_score:.6f}"
        r2["lift"] = f"{lift:.6f}"
        # optional: percent lift
        r2["lift_pct"] = f"{(lift / baseline_score * 100.0):.2f}%" if baseline_score else ""
        out_rows.append((score, lift, r2))

    # Sort by lift desc (you could sort by score instead if you prefer)
    out_rows.sort(key=lambda t: (t[1], t[0]), reverse=True)
    ranked = [r2 for _, _, r2 in out_rows]

    # Output CSV
    out_path = Path(args.output) if args.output else in_path.with_suffix(".ranked.csv")
    if ranked:
        fieldnames = list(ranked[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(ranked)
    print(f"Baseline: {baseline_found}")
    print(f"Wrote: {out_path}")

    # Pretty console preview
    print("\nTop results:")
    col_w = 42
    print(f"{'group':{col_w}}  {'n':>4}  {'score':>8}  {'lift':>8}  {'lift %':>8}")
    print("-" * (col_w + 36))
    for r in ranked[:args.top]:
        g = (r.get("group") or "")[:col_w]
        n = r.get("n_images") or ""
        sc = to_float(r.get(args.score_col))
        lf = to_float(r.get("lift"))
        lp = r.get("lift_pct", "")
        print(f"{g:{col_w}}  {n:>4}  {sc:8.2f}  {lf:8.2f}  {lp:>8}")

if __name__ == "__main__":
    main()
