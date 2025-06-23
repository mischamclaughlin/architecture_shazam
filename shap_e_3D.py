import time
import torch
import trimesh
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh
from extract_features import get_features
from description_prompt import generate_description
from summarise_prompt import summarise_for_3D


start_time = time.time()

tune_file = "./tunes/the_lion_king.mp3"
# tune_file = "./tunes/Tian_Mi_Mi-Teresa_Teng.mp3"
# tune_file = "./tunes/Guqin_Solo.mp3"
# tune_file = "./tunes/Beethoven-Für_Elise.mp3"

librosa_info, genre_info, instrument_info = get_features(tune_file)
model = "deepseek-r1:14b"

description = generate_description(
    librosa_info, genre_info, instrument_info, model, "house"
)
print(description)

prompt = summarise_for_3D(description)
print(prompt)

# Pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
xm = load_model("transmitter", device=device)
model = load_model("text300M", device=device)

# Build diffusion scheduler
diffusion = diffusion_from_config(load_config("diffusion"))

# Sample latents
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

# Decode to a TorchMesh
torch_mesh = decode_latent_mesh(xm, latents[0])

# Convert TorchMesh -> Ply via the .tri_mesh() helper
ply_path = "house_mesh.ply"
with open(ply_path, "wb") as f:
    # .tri_mesh() returns a mesh object with write_ply/write_obj
    torch_mesh.tri_mesh().write_ply(f)

# Load the PLY with trimesh and export to GLB
tm = trimesh.load(ply_path)
tm.export("house_mesh.glb", file_type="glb")

print("Saved -> house_mesh.glb")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for whole process: {elapsed_time:.2f} seconds.")
