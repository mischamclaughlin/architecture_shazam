# ./summarise_output.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to the folder cloned & pulled LFS into
LOCAL_MODEL_PATH = "models/mistral-7b-instruct-v0.3"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH, local_files_only=True, use_slow=True
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)
model.eval()


def summarise(raw_description: str) -> str:
    prompt = f"""
	Rewrite the following “overall look” description into a visual prompt for Stable Diffusion XL that depicts the entire building exterior—massing, roofline, elevation and context—rather than just a façade.

	Use vivid, concrete language to describe the full structure's form, materials, textures and overall silhouette from base to roof.

	Avoid interior details, abstract emotions or partial close-ups.

	Keep it concise and focused, max 65 tokens.

	Description:
	{raw_description}

	Return only the image prompt. No extra commentary.
	""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=75,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()
