import torch
import trimesh
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh

# 1) Pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load models
xm = load_model("transmitter", device=device)
model = load_model("text300M", device=device)

# 3) Build diffusion scheduler
diffusion = diffusion_from_config(load_config("diffusion"))

# 4) Your prompt
prompt = (
    "A bold, modern rock-inspired house with dynamic, angular massing and "
    "staggered volumes, creating a sense of movement and progression. The "
    "structure features a warm, earthy material palette with reclaimed wood "
    "cladding and weathered metal accents, evoking the organic yet gritty sound "
    "of rock music. Subdued façade brightness with strategically placed windows "
    "hints at vibrant interior life."
)

# 5) Sample latents
latents = sample_latents(
    batch_size=1,
    model=model,
    diffusion=diffusion,
    guidance_scale=10,
    model_kwargs={"texts": [prompt]},
    progress=True,
    clip_denoised=True,
    use_fp16=False,
    use_karras=True,
    karras_steps=100,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0.5,
)

# 6) Decode to a TorchMesh
torch_mesh = decode_latent_mesh(xm, latents[0])

# 7) Convert TorchMesh → Ply via the .tri_mesh() helper
ply_path = "house_mesh.ply"
with open(ply_path, "wb") as f:
    # .tri_mesh() returns a mesh object with write_ply/write_obj
    torch_mesh.tri_mesh().write_ply(f)

# 8) Load the PLY with trimesh and export to GLB
tm = trimesh.load(ply_path)
tm.export("house_mesh.glb", file_type="glb")

print("Saved -> house_mesh.glb")
