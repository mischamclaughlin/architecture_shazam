# ./llm_description.py
import logging
from typing import Any, Dict, List, Tuple, Optional
import argparse
import json
from pathlib import Path
from ollama import chat, ResponseError
from .logger import default_logger


class GenerateLLMDescription:
    """
    Generates architectural design prompts based on musical analytics by interacting with an Ollama LLM.

    Methods:
        - build_description_prompt: construct the expert architect concept prompt.
        - generate_description: send concept prompt to LLM and return response.
        - build_image_prompt: construct Stable Diffusion XL image prompt.
        - summarise_for_image: send image prompt, truncate and return.
        - build_3d_prompt: construct low-poly 3D modelling prompt.
        - summarise_for_3d: send 3D prompt, truncate and return.
    """

    def __init__(
        self,
        llm: str,
        temperature: float,
        max_tokens: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger or default_logger(__name__)

    def build_description_prompt(
        self,
        features: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
        origin: Tuple,
        building_type: str,
    ) -> str:
        """
        Construct the prompt to generate a concise architectural concept based on musical analytics.
        """

        genres = ", ".join(label for label, _ in genre_tags)
        instruments = ", ".join(label for label, _ in instrument_tags)
        
        return (
            f"Act as an expert architectural designer."
            f" Using the following musical analytics, produce a concise exterior concept for a {building_type}:"
            f" Tempo: {features['tempo']['global']} BPM,"
            f" Key: {features['key']},"
            f" Genre: {genres},"
            f" Instruments: {instruments},"
            f" Origin: {origin},"
            f" Timbre: {features['timbre']},"
            f" Loudness: {features['loudness']}\n\n"
            "Instructions:\n"
            "1. One-sentence concept statement that captures the genre-driven vision.\n"
            "2. Three bullets explaining how you translated:\n"
            "   - Rhythm: massing & form\n"
            "   - Tonal colour: material palette & façade brightness\n"
            "   - Energy & timbre: façade articulation\n"
            "3. Overall look (3-4 sentences): a magazine-style summary emphasising how the genre shapes the design language.\n\n"
            "Keep the total response under 500 words."
        )

    def build_description_prompt_with_song_info(
        self,
        features: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
        song_info: Dict[Any, Any],
        origin: Tuple,
        building_type: str,
    ) -> str:
        """
        Construct the prompt to generate a concise architectural concept based on musical analytics.
        """

        genres = ", ".join(label for label, _ in genre_tags)
        instruments = ", ".join(label for label, _ in instrument_tags)
        artists = ", ".join(name for name in song_info.get("artists"))

        return (
            f"Act as an expert architectural designer."
            f" Using the following musical analytics, produce a concise exterior concept for a {building_type}:"
            f" Artists: {artists},"
            f" Title: {song_info['title']},"
            f" Origin: {origin},"
            f" Tempo: {features['tempo']['global']} BPM,"
            f" Key: {features['key']},"
            f" Genre: {genres},"
            f" Instruments: {instruments},"
            f" Timbre: {features['timbre']},"
            f" Loudness: {features['loudness']}\n\n"
            "Instructions:\n"
            "1. One-sentence concept statement that captures the genre-driven vision.\n"
            "2. Three bullets explaining how you translated:\n"
            "   - Rhythm: massing & form\n"
            "   - Tonal colour: material palette & façade brightness\n"
            "   - Energy & timbre: façade articulation\n"
            "3. Overall look (3-4 sentences): a magazine-style summary emphasising how the genre shapes the design language.\n\n"
            "Keep the total response under 500 words."
        )

    def build_image_prompt(self, raw_description: str) -> str:
        """
        Construct a Stable Diffusion XL prompt to visualise the entire building exterior.
        """
        return (
            'Rewrite the following "overall look" description into a visual prompt for Stable Diffusion XL that depicts the entire building exterior—massing, roofline, elevation and context—rather than just a façade. '
            "Use vivid, concrete language to describe the full structure's form, materials, textures and overall silhouette from base to roof. "
            "Avoid interior details, abstract emotions or partial close-ups. "
            "Keep it concise and focused, max 75 tokens.\n\n"
            f"Description:\n{raw_description}\n\n"
            "Return only the image prompt. No extra commentary."
        )

    def build_3d_prompt(self, raw_description: str) -> str:
        """
        Construct a concise 3D modelling prompt for low-poly UV-unwrapped geometry.
        """
        return (
            "Convert the following description into a concise 3D modelling prompt (<75 tokens) that specifies:"
            " full exterior massing & roofline, context/environment, PBR materials, low-poly UV-unwrapped geometry with separate material IDs. "
            "Suitable for OBJ/FBX export. Omit interiors, emotions, and close-ups.\n\n"
            f"Description:\n{raw_description}\n\n"
            "Return only the prompt. No extra commentary."
        )

    def generate_description(
        self,
        features: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
        song_info: Optional[Dict[str, Any]],
        origin: Tuple,
        building_type: str,
    ) -> str:
        """
        Generate the architectural concept by building the prompt and querying the LLM.
        """
        if song_info:
            prompt = self.build_description_prompt_with_song_info(
                features, genre_tags, instrument_tags, song_info, origin, building_type
            )
        else:
            prompt = self.build_description_prompt(
                features, genre_tags, instrument_tags, origin, building_type
            )

        self.logger.info(f"LLM Prompt:\n{prompt}")
        return self.call_ollama(prompt)

    def summarise_for_image(self, raw_description: str) -> str:
        """
        Generate and truncate an image prompt for Stable Diffusion XL.
        """
        prompt = self.build_image_prompt(raw_description)
        return self._run_and_truncate(prompt)

    def summarise_for_3d(self, raw_description: str) -> str:
        """
        Generate and truncate a prompt for 3D modelling.
        """
        prompt = self.build_3d_prompt(raw_description)
        return self._run_and_truncate(prompt)

    def call_ollama(self, input_prompt: str) -> str:
        """
        Send a prompt to the Ollama chat API and return the raw text response.
        """
        try:
            response = chat(
                model=self.llm,
                messages=[{"role": "user", "content": input_prompt}],
                think=False,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            content = response["message"]["content"].strip()
            return content
        except ResponseError as e:
            self.logger.error(f"Ollama API error ({e.status_code}): {e.error}")
            raise

    def truncate_to_last_sentence(self, text: str) -> str:
        """
        Truncate text to end at the last full sentence, based on punctuation.
        """

        idx = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if idx != -1 and idx < len(text) - 1:
            return text[: idx + 1]
        return text

    def _run_and_truncate(self, prompt: str) -> str:
        """
        Helper to call Ollama, truncate response, and return final text.
        """
        raw = self.call_ollama(prompt)
        return self.truncate_to_last_sentence(raw)


# Test from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate architectural prompts from musical analytics"
    )
    parser.add_argument("--llm", required=True, help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--building-type", required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--genre-json", type=Path, required=True)
    parser.add_argument("--instrument-json", type=Path, required=True)
    parser.add_argument("--song-json", type=Path, help="Optional song_info JSON file")
    parser.add_argument(
        "--origin", nargs=2, metavar=("COUNTRY", "AREA"), help="Origin country and area"
    )
    args = parser.parse_args()

    features = json.loads(args.features_json.read_text())
    genre_tags = json.loads(args.genre_json.read_text())
    instrument_tags = json.loads(args.instrument_json.read_text())
    song_info = json.loads(args.song_json.read_text()) if args.song_json else None
    origin = tuple(args.origin) if args.origin else ("—", "")

    gen = GenerateLLMDescription(args.llm, args.temperature, args.max_tokens)
    desc = gen.generate_description(
        features, genre_tags, instrument_tags, song_info, origin, args.building_type
    )
    print(desc)
