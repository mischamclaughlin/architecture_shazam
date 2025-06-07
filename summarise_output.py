# ./summarise_output.py
import subprocess
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer

_tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
)
_model_max_len = _tokenizer.model_max_length


def summarise(raw_description: str) -> str:
    # Build the prompt that instructs Llama3.2 to summarise the longer description and keep it under 77 tokens
    system_prompt = f"""
	Rewrite the following “overall look” section from the description into a visual prompt for Stable Diffusion XL.
	It has to depicts the entire building exterior—massing, roofline, elevation and context—rather.

	Use vivid, concrete language to describe the full structure's form, materials, textures and overall silhouette from base to roof.

	Avoid interior details, abstract emotions or partial close-ups.

	Keep it concise and focused, max 70 tokens.

	Description:
	{raw_description}

	Return only the image prompt. No extra commentary.
	""".strip()

    # Call Llama3.2 via Ollama
    result = subprocess.run(
        ["ollama", "run", "llama3.2:latest"],
        input=system_prompt,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed:\n{result.stderr.strip()}")

    raw_output = result.stdout.strip()

    encoded = _tokenizer(raw_output, max_length=_model_max_len, truncation=True)
    truncated_ids = encoded.input_ids
    final_prompt = _tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()

    return final_prompt
