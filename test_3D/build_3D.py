#!/usr/bin/env python3
"""
High-quality single-image 3D mesh pipeline with robust reconstruction:
1. Depth estimation with full MiDaS
2. Point-cloud generation + denoising
3. Surface reconstruction: Ball Pivoting + hole fill
4. Mesh smoothing and decimation
5. Export textured mesh (vertex colours)
"""
import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from pathlib import Path

# -----------------------
# 1. Load MiDaS model
# -----------------------
model_type = "MiDaS"  # full model for best quality
midas = torch.hub.load("intel-isl/MiDaS", model_type)
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.default_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
midas.to(device).eval()


# -----------------------
# 2. Depth prediction
# -----------------------
def predict_depth(img: np.ndarray) -> np.ndarray:
    inp = transform(img).to(device)
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        pred = midas(inp)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        depth_resized = torch.nn.functional.interpolate(
            pred,
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        depth = depth_resized[0, 0].cpu().numpy()
    return depth


# -----------------------
# 3. Point cloud + denoise
# -----------------------
def depth_to_pointcloud(rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
    h, w = depth.shape
    fx = fy = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth.flatten()
    xs = (us.flatten() - cx) * zs / fx
    ys = (vs.flatten() - cy) * zs / fy
    pts = np.stack([xs, ys, zs], axis=-1)
    cols = rgb.reshape(-1, 3) / 255.0
    mask = zs > (zs.min() + 1e-6)
    pts, cols = pts[mask], cols[mask]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    return pcd


# -----------------------
# 4. Surface reconstruction via Ball Pivoting + hole fill
# -----------------------
def reconstruct_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    # Estimate normals
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(50)
    # Ball Pivoting
    radii = [0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    # Convert to trimesh for hole filling
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(
            np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        ),
        process=False,
    )
    # In-place hole filling
    tm.fill_holes()
    # Convert back to Open3D
    filled = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tm.vertices),
        triangles=o3d.utility.Vector3iVector(tm.faces),
    )
    if tm.visual.vertex_colors is not None:
        colors = tm.visual.vertex_colors[:, :3] / 255.0
        filled.vertex_colors = o3d.utility.Vector3dVector(colors)
    return filled


# -----------------------
# 5. Mesh post-processing
# -----------------------
def refine_mesh(
    mesh: o3d.geometry.TriangleMesh, target_triangles: int = None
) -> o3d.geometry.TriangleMesh:
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    if target_triangles:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


# -----------------------
# Main pipeline
# -----------------------
def main():
    img_path = Path(
        "generated_images/lion_king/deepseek-r1_14b/deepseek-r1_14b_20250621_225427.png"
    )
    out_mesh = Path("output_mesh.ply")
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Predicting depth...")
    depth = predict_depth(rgb)
    print("Building point cloud...")
    pcd = depth_to_pointcloud(rgb, depth)
    print("Reconstructing mesh...")
    mesh = reconstruct_mesh(pcd)
    print("Refining mesh...")
    mesh = refine_mesh(mesh, target_triangles=len(mesh.triangles) // 2)
    print(f"Saving mesh to {out_mesh}")
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    print("Done.")


if __name__ == "__main__":
    main()
