# ./model_prompt.py
from ollama import chat, ResponseError


def generate_description(
    librosa_tags: dict,
    genre_tags: list,
    instrument_tags: list,
    model: str,
    building_type: str,
) -> str:
    prompt = f"""
	Act as an expert architectural designer.
	Using the following musical analytics, produce a concise exterior concept for a {building_type}:
		- Tempo: {librosa_tags['tempo_global']} BPM
		- Key: {librosa_tags['key']}
		- Genre: {', '.join([g for g,_ in genre_tags])}
		- Instruments: {', '.join([i for i,_ in instrument_tags])}
		- Timbre: {librosa_tags['timbre']}
		- Loudness: {librosa_tags['loudness']}

	Instructions
		1. One-sentence concept statement that captures the genre-driven vision.
		2. Three bullets explaining how you translated:
			- Rhythm: massing & form
			- Tonal colour: material palette & façade brightness
			- Energy & timbre: façade articulation
		3. Overall look (3-4 sentences): a magazine-style summary emphasising how the genre shapes the design language.

	Keep the total response under 500 words.
	""".strip()

    try:
        # call the Ollama chat endpoint directly
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0.7},
        )
        print(prompt, "\n")
        # response is a dict-like; grab the content string
        return response["message"]["content"].strip()
    except ResponseError as e:
        # Ollama API errors
        print(f"Ollama API error ({e.status_code}): {e.error}")
        return ""
