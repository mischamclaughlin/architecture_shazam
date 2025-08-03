from datetime import datetime
import time
import os

import torch
import numpy as np
import trimesh
from trimesh.remesh import subdivide

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

from . import get_features, generate_description, summarise_for_3D


start_time = time.time()

# Ensure float32 tensors when coming from NumPy
_orig_from_numpy = torch.from_numpy


def _from_numpy32(arr: np.ndarray):
    return _orig_from_numpy(arr.astype(np.float32))


torch.from_numpy = _from_numpy32

# Pick tune
tune_file = "./tunes/the_lion_king.mp3"
librosa_info, genre_info, instrument_info = get_features(tune_file)

# Generate text prompts
llm = "deepseek-r1:14b"
description = generate_description(
    librosa_info, genre_info, instrument_info, llm, "house"
)
print(f"\nDescription: {description}\n")

prompt = summarise_for_3D(description)
print(f"3D Prompt: {prompt}\n")

# Default to CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f"Using device: {device}")

# Only use FP16 on CUDA
use_fp16 = device.type == "cuda"

xm = load_model("transmitter", device=device)
model = load_model("text300M", device=device)
diffusion = diffusion_from_config(load_config("diffusion"))

batch_size = 1
guidance_scale = 25.0

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=use_fp16,
    use_karras=True,
    karras_steps=256,
    sigma_min=1e-7,
    sigma_max=100,
    s_churn=0.1,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = "generated_3D_models"
os.makedirs(out_dir, exist_ok=True)

# Choose the best device for decoding (CUDA if available, else CPU)
optimal_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the transmitter model onto that device
optimal_xm = load_model("transmitter", device=optimal_device)

for i, latent in enumerate(latents):
    print(f"[{i}] Decoding mesh at {datetime.now().strftime('%H:%M:%S')}…")
    latent = latent.to(optimal_device)
    raw_mesh = decode_latent_mesh(optimal_xm, latent).tri_mesh()

    filename = f"house_mesh_{i}_{timestamp}.obj"
    path = os.path.join(out_dir, filename)
    try:
        with open(path, "w", encoding="utf8") as f:
            raw_mesh.write_obj(f)
        print(f"Wrote OBJ → {path}\n")
    except Exception as e:
        print(f"Failed to write OBJ: {e}\n")

total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f}s")
