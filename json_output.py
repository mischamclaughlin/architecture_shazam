# ./json_output.py
import json, re
from ollama import chat, ResponseError
from librosa_analysis import analyse_features
from genres_analysis import get_genre
from instruments_analysis import get_instruments
from description_prompt import generate_description

tune_file = "./tunes/the_lion_king.mp3"

librosa_info = analyse_features(tune_file)
print(librosa_info, "\n")
# yamnet_info = analyse_yamnet(tune_file)
# print(yamnet_info, "\n")
genre_info = get_genre(tune_file)
print(genre_info, "\n")
instrument_info = get_instruments(file_path=tune_file)
print(instrument_info, "\n")

description = generate_description(
    librosa_info, genre_info, instrument_info, "deepseek-r1:14b", "house"
)


arch_schema = {
    "floors": "integer",
    "floor_height": "number",
    "window_style": "string",
    "windows_per_floor": "integer",
    "roof_pitch": "number",
    "facade_pattern": "string",
    "ornament_level": "number",
}


def extract_params(
    raw_description: str, schema: dict, llm: str = "deepseek-r1:14b"
) -> dict:
    prompt = f"""
    Here is the building concept:
    {raw_description}.
    
    Now **only** emit the JSON parameters between <<JSON>> and <<END_JSON>> matching this schema:
    {json.dumps(schema, indent=2)}
    <<JSON>>
    {{ … }}
    <<END_JSON>>
    """
    try:
        response = chat(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0.0, "stop": ["<<END_JSON>>"]},
            format="json",
        )
        raw = response["message"]["content"]
        m = re.search(r"<<JSON>>(.*)<<END_JSON>>", raw, re.S)
        json_str = m.group(1).strip() if m else raw
        return json.loads(json_str)
    except (ResponseError, ValueError) as e:
        print(f"Extraction failed: {e}")
        return {}


generate_json = extract_params(description, arch_schema)
print(generate_json)

with open("params.json", "w", encoding="utf-8") as f:
    json.dump(generate_json, f, ensure_ascii=False)
