# ./sharp_it_pipeline.py

import os
import math
import torch
import numpy as np
from PIL import Image
import trimesh
from diffusers import (
    ShapEPipeline,
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_obj

# PyTorch3D imports
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments

# 1. Device and dtype
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
)
torch_dtype = torch.float32 if device.type == "mps" else torch.float16
print(f"Using device: {device}, dtype: {torch_dtype}")


# 2. Output directories
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


out_dir = "outputs"
mesh_dir = os.path.join(out_dir, "meshes")
views_dir = os.path.join(out_dir, "views")
masks_dir = os.path.join(out_dir, "masks")
depth_dir = os.path.join(out_dir, "depths")
refine_dir = os.path.join(out_dir, "refined")
for d in [mesh_dir, views_dir, masks_dir, depth_dir, refine_dir]:
    ensure_dir(d)

# 3. Generate coarse mesh with Shap-E
print("Loading Shap-E pipeline...")
pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch_dtype).to(
    device
)
pipe.enable_attention_slicing()

if device.type == "mps":
    # Patch scheduler for MPS
    orig = pipe.scheduler.set_timesteps

    def patched(self, num_steps, device=None):
        orig(num_steps, device="cpu")
        self.timesteps = self.timesteps.to(device=device, dtype=torch.float32)

    pipe.scheduler.set_timesteps = patched.__get__(pipe.scheduler, type(pipe.scheduler))

print("Generating coarse 3D mesh...")
res = pipe(
    prompt="A wooden tiki mask",
    num_inference_steps=64,
    guidance_scale=15.0,
    output_type="mesh",
)
mesh_out = res.images[0]
mesh_path = os.path.join(mesh_dir, "coarse_model.obj")
export_to_obj(mesh_out, mesh_path)
print(f"Exported coarse mesh to {mesh_path}")

# 4. Reorient mesh: rotate -90° about X-axis so front faces +Y
print("Reorienting mesh for correct camera framing...")
mesh_tr = trimesh.load(mesh_path)
R = trimesh.transformations.rotation_matrix(
    angle=-math.pi / 2, direction=[1, 0, 0], point=mesh_tr.centroid
)
mesh_tr.apply_transform(R)
mesh_reorient_path = os.path.join(mesh_dir, "coarse_model_reoriented.obj")
mesh_tr.export(mesh_reorient_path)
print(f"Reoriented mesh saved to {mesh_reorient_path}")

# 5. Multi-view rendering and depth/mask extraction
print("Rendering multi-view images and depth maps with PyTorch3D...")
render_dev = torch.device("cpu") if device.type == "mps" else device
mesh_obj = load_objs_as_meshes([mesh_reorient_path], device=render_dev)
# Dummy white vertex textures
verts = mesh_obj.verts_packed()
mesh_obj.textures = TexturesVertex(
    verts_features=torch.ones((1, verts.shape[0], 3), device=render_dev)
)
# Camera and rasteriser settings
dist = verts.norm(dim=1).max().item() * 2.5
elev = 30.0
angles = [0, 60, 120, 180, 240, 300]
raster_settings = RasterizationSettings(
    image_size=512, blur_radius=0.0, faces_per_pixel=1
)
mesh_rasterizer = MeshRasterizer(cameras=None, raster_settings=raster_settings)

for i, az in enumerate(angles):
    print(f"Rendering view {i} (azimuth {az}°) ...")
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=az, device=render_dev)
    cameras = FoVPerspectiveCameras(device=render_dev, R=R, T=T)
    lights = PointLights(device=render_dev, location=T, ambient_color=[[0.3, 0.3, 0.3]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=render_dev, cameras=cameras, lights=lights),
    )
    # Render RGBA
    images = renderer(mesh_obj)
    rgba = images[0].cpu().numpy()
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    mask = (rgba[..., 3] > 0).astype(np.uint8) * 255
    # Save view and mask
    Image.fromarray(rgb).save(os.path.join(views_dir, f"view_{i}.png"))
    Image.fromarray(mask).save(os.path.join(masks_dir, f"mask_{i}.png"))
    # Render depth
    fragments: Fragments = mesh_rasterizer(mesh_obj.extend(1), cameras=cameras)
    zbuf = fragments.zbuf[0, ..., 0].cpu().numpy()
    zbuf[np.isinf(zbuf)] = 0
    zmin, zmax = zbuf.min(), zbuf.max()
    znorm = (zbuf - zmin) / (zmax - zmin + 1e-8)
    depth_img = (znorm * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(os.path.join(depth_dir, f"depth_{i}.png"))
    print(f"Saved view_{i}.png, mask_{i}.png, depth_{i}.png")

# 6. Silhouette-preserving inpainting
print("Loading inpainting pipeline...")
paint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch_dtype
).to(device)
paint_pipe.scheduler = UniPCMultistepScheduler.from_config(paint_pipe.scheduler.config)
paint_pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
paint_pipe.enable_attention_slicing()

prompt = "An intricately carved wooden tiki mask with realistic lighting and fine texture detail"
seed = 42
generator = torch.Generator(device=device).manual_seed(seed)
steps, guidance = 30, 9.0

for i in range(len(angles)):
    print(f"Inpainting view {i}...")
    init_image = Image.open(os.path.join(views_dir, f"view_{i}.png")).convert("RGB")
    mask_image = Image.open(os.path.join(masks_dir, f"mask_{i}.png")).convert("L")
    result = paint_pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    out_image = result.images[0]
    out_image.save(os.path.join(refine_dir, f"refined_{i}.png"))
    print(f"Saved refined_{i}.png")

# Final confirmation
print("Pipeline complete!")
