import subprocess

def summarise(raw_description):
  prompt = f"""
  Rewrite the following architectural description into a single-sentence visual prompt suitable for Stable Diffusion XL. Use rich, vivid language to describe the building's exterior appearance only — as if for an image generation model.

  Focus on overall form, façade features, materials, and visual impact. Avoid abstract concepts, emotions, or interior details.

  Limit the result to **a maximum of 77 tokens**.

  Description:
  {raw_description}

  Return only the rewritten image prompt. No explanations or extra commentary.
  """.strip()
  
  results = subprocess.run(
    ['ollama', 'run', 'llama3.2:latest'],
    input=prompt,
    capture_output=True,
    text=True
  )
  return results.stdout.strip()