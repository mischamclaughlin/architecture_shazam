#!/usr/bin/env python3
"""
End-to-end Nerfstudio pipeline script:
1. Generate transforms.json from zero123 outputs
2. Train NeRF model directly with manual poses and iteration override
3. Preview results
4. Export mesh
"""
import os
import math
import json
import subprocess
import argparse
import torch


def generate_transforms(
    images_dir,
    output_json,
    num_views,
    polar_deg=30.0,
    radius=1.0,
    image_w=1024,
    image_h=1024,
    fov_deg=40.0,
    focal=None,
):
    """
    Generate a transforms.json for Nerfstudio from zero123 views.
    """
    if focal is None:
        focal = image_w / (2 * math.tan(math.radians(fov_deg) / 2))

    frames = []
    for i in range(num_views):
        az = math.radians(i * 360.0 / num_views)
        el = math.radians(polar_deg)
        x = radius * math.cos(el) * math.sin(az)
        y = radius * math.sin(el)
        z = radius * math.cos(el) * math.cos(az)
        transform = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append(
            {
                "file_path": os.path.join("images", f"{i:03d}.png"),
                "transform_matrix": transform,
            }
        )

    data = {
        "camera_angle_x": math.radians(fov_deg),
        "fl_x": focal,
        "fl_y": focal,
        "cx": image_w / 2,
        "cy": image_h / 2,
        "w": image_w,
        "h": image_h,
        "frames": frames,
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated transforms.json at {output_json}")


def run_command(cmd, cwd=None):
    """
    Run a shell command and stream output.
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Nerfstudio pipeline for zero123 outputs"
    )
    parser.add_argument(
        "--images", required=True, help="Directory of zero123-generated PNGs"
    )
    parser.add_argument(
        "--dataset", default="dataset", help="Output Nerfstudio dataset dir"
    )
    parser.add_argument("--views", type=int, default=12, help="Number of views")
    parser.add_argument("--radius", type=float, default=1.0, help="Camera radius")
    parser.add_argument(
        "--polar", type=float, default=30.0, help="Polar angle in degrees"
    )
    parser.add_argument(
        "--fov", type=float, default=40.0, help="Camera field of view in degrees"
    )
    parser.add_argument(
        "--pipeline",
        default="nerfacto",
        help="Nerfstudio pipeline to use (e.g. nerfacto)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override trainer.max_num_iterations for quicker tests",
    )
    args = parser.parse_args()

    # Step 1: organise dataset
    os.makedirs(args.dataset, exist_ok=True)
    imgs_dest = os.path.join(args.dataset, "images")
    os.makedirs(imgs_dest, exist_ok=True)
    for filename in os.listdir(args.images):
        if filename.lower().endswith(".png"):
            src = os.path.join(args.images, filename)
            dst = os.path.join(imgs_dest, filename)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

    # Step 2: Generate transforms.json
    transforms_path = os.path.join(args.dataset, "transforms.json")
    generate_transforms(
        images_dir=imgs_dest,
        output_json=transforms_path,
        num_views=args.views,
        polar_deg=args.polar,
        radius=args.radius,
        image_w=1024,
        image_h=1024,
        fov_deg=args.fov,
    )

    # Step 3: train NeRF using manual poses
    device_type = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    # construct command: method, flags, then Hydra overrides
    cmd = [
        "ns-train",
        args.pipeline,
        "--data",
        args.dataset,
        "--machine.device_type",
        device_type,
    ]
    if args.iterations:
        cmd.append(f"trainer.max_num_iterations={args.iterations}")

    run_command(cmd)

    # Step 4: preview
    print("To preview your model, run:")
    print(f"  ns-preview --logdir logs/{os.path.basename(args.dataset)}")

    # Step 5: export mesh
    print("To export a mesh, run:")
    print(
        f"  ns-export --logdir logs/{os.path.basename(args.dataset)} --export-format obj"
    )


if __name__ == "__main__":
    main()
