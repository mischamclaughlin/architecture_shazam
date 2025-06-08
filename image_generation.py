# ./image_generation.py
import os, time, json
from datetime import datetime
import torch
from tune_analysis import analyse_tune
from model_prompt import generate_description
from diffusers import StableDiffusionXLPipeline
from summarise_output import summarise

start_time = time.time()

tune_file = "./tunes/the_lion_king.mp3"
# tune_file = "./tunes/house_lo.mp3"

audio_info = analyse_tune(tune_file)
print(audio_info, "\n")

models = ["llama3.2:latest", "deepseek-r1:8b", "deepseek-r1:14b"]
model_description = {}
for model in models:
    start = time.time()
    description = generate_description(audio_info, model)
    print(f"{model.title()}: {description}")
    model_description[model] = description

    end = time.time()
    total = end - start
    print(f"Total time taken for {model} description: {total:.2f} seconds\n")

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    variant=None,
).to(device)

print("Tokenizer model max length:", pipe.tokenizer.model_max_length)
print("Text encoder:", pipe.text_encoder.__class__.__name__)

# Warm-up (improves first-time speed)
pipe(prompt="warmup", num_inference_steps=1)

# Get summaries from longer description
summaries = {}
for model in model_description:
    start = time.time()
    summary = summarise(model_description[model])
    summaries[model] = summary
    print(f"{model.title()}: {summary}")

    end = time.time()
    total = end - start
    print(f"Total time taken for {model} summary: {total:.2f} seconds\n")

num_inference = 25
for model in summaries:
    start = time.time()
    # Generate image
    image = pipe(
        prompt=summaries[model],
        # height=512,
        # width=512,
        num_inference_steps=num_inference,
    ).images[0]

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "_")
    # save_dir = os.path.abspath(f"./generated_images/house_lo/{safe_model}")
    save_dir = os.path.abspath(f"./generated_images/lion_king/{safe_model}")
    os.makedirs(save_dir, exist_ok=True)

    image_filename = f"{safe_model}_{timestamp}.png"
    image_path = os.path.join(save_dir, image_filename)
    image.save(image_path)

    # Get JSON info and save to metadata
    metadata = {
        "model": model,
        "timestamp": timestamp,
        "audio_tags": audio_info,
        "llm_description": model_description[model],
        "sdxl_prompt": summaries[model],
        "num_inference_steps": num_inference,
        "device": device,
    }
    json_filename = f"{safe_model}_{timestamp}.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    end = time.time()
    total = end - start
    print(f"Saved {image_filename} and {json_filename} in {save_dir}")
    print(f"Total time taken for {model} image: {total:.2f} seconds\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for whole process: {elapsed_time:.2f} seconds.")
