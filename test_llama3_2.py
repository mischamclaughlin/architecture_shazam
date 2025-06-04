from tune_analysis import analyse_tune
from model_prompt import generate_description
import time

start_time = time.time()

file = './tunes/the_lion_king.mp3'
audio_info = analyse_tune(file)
print(audio_info)
print('\n')

llm = 'llama3.2:latest'
description = generate_description(audio_info, llm)
print(description)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time taken: {elapsed_time:.2f} seconds with {llm}")