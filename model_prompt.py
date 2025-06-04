import subprocess

def generate_description(audio_tags, model):
  prompt = f"""
            Create a brief architectural concept of the exterior based on these music features:
            Tempo: {audio_tags['tempo']} BPM
            Key: {audio_tags['key']}
            Mood: {audio_tags['mood']}
            Instruments: {', '.join(audio_tags['instruments'])}
            """
  
  results = subprocess.run(['ollama', 'run', model, prompt],
                           capture_output=True,
                           text=True
                           )
  
  return results.stdout