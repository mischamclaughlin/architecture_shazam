from diffusers import StableDiffusionXLPipeline
import torch
from summarise_output import summarise
import test_llama3_2
import test_ds_8b
import test_ds_14b
import os
from datetime import datetime

device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained(
  'stabilityai/stable-diffusion-xl-base-1.0',
  torch_dtype=torch.float16 if device != "cpu" else torch.float32,
  variant='fp16' if device != "cpu" else None,
).to(device)

prompt_llama3_2 = summarise(test_llama3_2.description)
print('Llama:', prompt_llama3_2)
prompt_ds_8b = summarise(test_ds_8b.description)
print('DeepSeek-8b:', prompt_ds_8b)
prompt_ds_14b = summarise(test_ds_14b.description)
print('DeepSeek-14b', prompt_ds_14b)

image_llama = pipe(prompt=prompt_llama3_2).images[0]
image_ds_8b = pipe(prompt=prompt_ds_8b).images[0]
image_ds_14b = pipe(prompt=prompt_ds_14b).images[0]


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.abspath('./generated_images/')
image_llama.save(os.path.join(save_dir, f"llama_{timestamp}.png"))
image_ds_8b.save(os.path.join(save_dir, f"ds_8b_{timestamp}.png"))
image_ds_14b.save(os.path.join(save_dir, f"ds_14b_{timestamp}.png"))