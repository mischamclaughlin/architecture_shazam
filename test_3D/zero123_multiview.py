import os
import torch
from PIL import Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "sudo-ai/zero123plus-v1.1"

# Load Pipeline
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# Input and Output
seed_path = (
    "generated_images/lion_king/deepseek-r1_14b/deepseek-r1_14b_20250621_225427.png"
)
out_dir = "data/zero123/images"
os.makedirs(out_dir, exist_ok=True)

# Load the input image
if not os.path.exists(seed_path):
    raise FileNotFoundError(f"Seed image not found at: {seed_path}")
seed_image = Image.open(seed_path).convert("RGB")


# Generate Multi-View Images
num_views = 12
for i in range(num_views):
    # Calculate angles for a full circle
    azimuth_deg = i * (360 / num_views)
    polar_deg = 30.0  # Elevation angle, 0 is from the side, 90 is from above

    print(f"Generating view {i+1}/{num_views} at azimuth {azimuth_deg:.1f}°...")

    # The pipeline takes the image and angles directly
    result = pipe(
        image=seed_image,
        num_inference_steps=28,
        guidance_scale=7.5,
        azimuth=azimuth_deg,
        polar=polar_deg,
        output_type="pil",
    )

    img = result.images[0]
    output_path = os.path.join(out_dir, f"{i:03d}.png")
    img.save(output_path)
    print(f"Saved to {output_path}")

print(f"\nSuccessfully generated {num_views} multi-view images in {out_dir}")
