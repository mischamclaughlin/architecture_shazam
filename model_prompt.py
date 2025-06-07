# ./model_prompt.py
import subprocess


def generate_description(audio_tags: str, model: str) -> str:
    prompt = f"""
	You are an expert architectural designer. 
	Design a concise exterior concept for a building, using all of the following musical analytics. 
	Explain how each feature informs your form-making, material choices, and façade details.

	Music features:
		- Tempo (global): {audio_tags['tempo_global']} BPM
		- Tempo (mean, local): {audio_tags['tempo_mean_local']} BPM
		- Tempo (median, local): {audio_tags['tempo_median_local']} BPM
		- Mean spectral bandwidth: {audio_tags['mean_bandwidth']}
		- Median spectral bandwidth: {audio_tags['median_bandwidth']}
		- Std dev of bandwidth: {audio_tags['std_bandwidth']}
		- Key: {audio_tags['key']}
		- Spectral centroid: {audio_tags.get('mean_centroid', 'N/A')}
		- Mean RMS: {audio_tags.get('mean_rms', 'N/A')}
		- MFCC means: {', '.join(str(audio_tags.get(f'mfcc{i}_mean', '-')) for i in range(1, 14))}

	Instructions:
		1. Write a one-sentence concept statement.  
		2. In three bullet points, explain how you translate:  
			- Rhythm & pacing into massing & form.  
			- Tonal colour & texture into material palette & façade brightness.  
			- Energy & timbre into façade articulation or pattern.  
		3. Write a 3-4 sentence “Overall look” summary (imagine a headline in an architecture magazine).
	Keep your response under 500 words.
	""".strip()

    results = subprocess.run(
        ["ollama", "run", model], input=prompt, capture_output=True, text=True
    )
    return results.stdout.strip()
