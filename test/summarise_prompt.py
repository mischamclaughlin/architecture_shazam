# ./summarise_prompt.py
from ollama import chat, ResponseError


def summarise_for_image(raw_description: str) -> str:
    prompt = f"""
	Rewrite the following “overall look” description into a visual prompt for Stable Diffusion XL that depicts the entire building exterior—massing, roofline, elevation and context—rather than just a façade.
	Use vivid, concrete language to describe the full structure's form, materials, textures and overall silhouette from base to roof.
	Avoid interior details, abstract emotions or partial close-ups.
	Keep it concise and focused, max 75 tokens.

	Description:
	{raw_description}

	Return only the image prompt. No extra commentary.
	""".strip()

    summary = call_ollama(prompt)
    summary_cut = truncate_to_last_sentence(summary)
    return summary_cut


def summarise_for_3D(raw_description: str) -> str:
    prompt = f"""
    Convert the following description into a concise 3D modelling prompt (<75 tokens) that specifies:
    - Full exterior massing & roofline
    - Context/environment
    - PBR materials
    - Low-poly UV-unwrapped geometry with separate material IDs
    Suitable for OBJ/FBX export. Omit interiors, emotions, and close-ups.

    Description:
    {raw_description}
    """.strip()

    summary = call_ollama(prompt)
    summary_cut = truncate_to_last_sentence(summary)
    return summary_cut


def call_ollama(input_prompt: str, llm: str = "deepseek-r1:14b", max_tokens=75):
    try:
        # call the Ollama chat endpoint directly
        response = chat(
            model=llm,
            messages=[{"role": "user", "content": input_prompt}],
            think=False,
            options={"temperature": 0.0, "num_predict": max_tokens},
        )
        # response is a dict-like; grab the content string
        image_text = response["message"]["content"].strip()
    except ResponseError as e:
        # Ollama API errors
        print(f"Ollama API error ({e.status_code}): {e.error}")
        return ""

    text = truncate_to_last_sentence(image_text)
    return text


def truncate_to_last_sentence(text: str) -> str:
    idx = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if idx != -1 and idx < len(text) - 1:
        return text[: idx + 1]
    return text
