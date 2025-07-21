# sharp_it_mac_highres_pipeline.py
# High-resolution mesh pipeline: Shap-E → reorient → subdivide/smooth → multi-view depth render → TSDF fusion

import os
import math
import torch
import numpy as np
from PIL import Image
from diffusers import ShapEPipeline
from diffusers.utils import export_to_obj
import trimesh
import open3d as o3d

# PyTorch3D for depth rendering
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)

# 1. Device & output
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
)
out_dir = "outputs_highres"
mesh_dir = os.path.join(out_dir, "meshes")
depth_dir = os.path.join(out_dir, "depths")
fused_dir = os.path.join(out_dir, "fused")
for d in [mesh_dir, depth_dir, fused_dir]:
    os.makedirs(d, exist_ok=True)

# 2. Generate high-res Shap-E mesh
pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float32).to(
    device
)
pipe.enable_attention_slicing()


# Patch scheduler for MPS float32 timesteps if necessary
def patch_mps_scheduler(p):
    orig_set = p.scheduler.set_timesteps

    def patched_set(num_steps, device=None):
        # Use CPU for initial timesteps and cast to float32 on MPS
        orig_set(num_steps, device="cpu")
        p.scheduler.timesteps = p.scheduler.timesteps.to(
            device=device, dtype=torch.float32
        )

    p.scheduler.set_timesteps = patched_set


if device.type == "mps":
    patch_mps_scheduler(pipe)
res = pipe(
    prompt="A wooden tiki mask",
    num_inference_steps=150,
    guidance_scale=22.0,
    output_type="mesh",
    generator=torch.Generator(device=device).manual_seed(1234),
)
mesh_out = res.images[0]
mesh_path = os.path.join(mesh_dir, "coarse.obj")
export_to_obj(mesh_out, mesh_path)
print(f"Shap-E mesh saved to {mesh_path}")

# 3. Reorient mesh
tm = trimesh.load(mesh_path)
centroid = tm.centroid
R = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0], centroid)
tm.apply_transform(R)
mesh_reorient = os.path.join(mesh_dir, "coarse_reoriented.obj")
tm.export(mesh_reorient)
print(f"Reoriented mesh saved to {mesh_reorient}")

# 4. Subdivide + smooth (Open3D)
omesh = o3d.io.read_triangle_mesh(mesh_reorient)
omesh.compute_vertex_normals()
omesh = omesh.subdivide_loop(number_of_iterations=2)
omesh = omesh.filter_smooth_laplacian(number_of_iterations=3)
fine_path = os.path.join(mesh_dir, "highres.ply")
o3d.io.write_triangle_mesh(fine_path, omesh)
print(f"High-res subdivided mesh saved to {fine_path}")

# 5. Render multi-view depth maps
print("Rendering depth maps...")
# load fine mesh in PyTorch3D
# Convert PLY to OBJ for PyTorch3D compatibility
ply_mesh = trimesh.load(fine_path)
obj_temp = os.path.join(mesh_dir, "highres_temp.obj")
ply_mesh.export(obj_temp)
mesh_obj = load_objs_as_meshes([obj_temp], device=device)

# Define camera parameters
bounds = mesh_obj.verts_packed().norm(dim=1).max().item()
dist = bounds * 2.5
elev = 30.0
angles = [0, 60, 120, 180, 240, 300]
raster_settings = RasterizationSettings(
    image_size=512, blur_radius=0.0, faces_per_pixel=1
)
mesh_rasterizer = MeshRasterizer(cameras=None, raster_settings=raster_settings)

for i, az in enumerate(angles):
    print(f"Rendering view {i} (azimuth {az}°) ...")
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=az, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh_obj)
    zbuf = fragments.zbuf[0, ..., 0].cpu().numpy()
    zbuf[np.isinf(zbuf)] = 0
    zmin, zmax = zbuf.min(), zbuf.max()
    znorm = (zbuf - zmin) / (zmax - zmin + 1e-8)
    depth_img = (znorm * 255).astype(np.uint8)
    depth_path = os.path.join(depth_dir, f"depth_{i}.png")
    Image.fromarray(depth_img).save(depth_path)
    print(f"Saved depth map for view {i}")

# 6. TSDF fusion (Open3D)
print("Starting TSDF fusion...")
# Intrinsics for pinhole camera
depth_scale = 1.0


def create_intrinsics(width, height, fov_deg):
    fx = fy = width / (2 * math.tan(math.radians(fov_deg) / 2))
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, width / 2, height / 2
    )


intrinsics = create_intrinsics(512, 512, 60)
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.002,
    sdf_trunc=0.04,
    color_type=getattr(o3d.pipelines.integration.TSDFVolumeColorType, "None"),
)

# Integrate each depth
for i, az in enumerate(angles):
    depth = o3d.io.read_image(os.path.join(depth_dir, f"depth_{i}.png"))
    # Compute camera extrinsic (world_to_camera)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=az, device=device)
    R = R[0].cpu().numpy()
    T = T[0].cpu().numpy()
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)),
        depth,
        depth_scale=depth_scale,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False,
    )
    vol.integrate(rgbd, intrinsics, np.linalg.inv(extrinsic))

# Extract and save fused mesh
mesh_fused = vol.extract_triangle_mesh()
mesh_fused.compute_vertex_normals()
fused_path = os.path.join(fused_dir, "fused_mask.ply")
o3d.io.write_triangle_mesh(fused_path, mesh_fused)
print(f"Fused high-res mesh saved to {fused_path}")
