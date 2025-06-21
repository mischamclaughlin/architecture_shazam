# ./summarise_output.py
from ollama import chat, ResponseError


def summarise(raw_description: str, llm="deepseek-r1:8b", max_tokens=75) -> str:
    prompt = f"""
	Rewrite the following “overall look” description into a visual prompt for Stable Diffusion XL that depicts the entire building exterior—massing, roofline, elevation and context—rather than just a façade.

	Use vivid, concrete language to describe the full structure's form, materials, textures and overall silhouette from base to roof.

	Avoid interior details, abstract emotions or partial close-ups.

	Keep it concise and focused, max 75 tokens.

	Description:
	{raw_description}

	Return only the image prompt. No extra commentary.
	""".strip()

    try:
        # call the Ollama chat endpoint directly
        response = chat(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
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
