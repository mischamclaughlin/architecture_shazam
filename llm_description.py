# ./llm_description.py
import logging
from typing import Any, Dict, List, Tuple, Optional
import re
import argparse
import json
from pathlib import Path

from ollama import chat, ResponseError

from logger import default_logger


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
        librosa_tags: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
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
            f" Tempo: {librosa_tags['tempo_global']} BPM,"
            f" Key: {librosa_tags['key']},"
            f" Genre: {genres},"
            f" Instruments: {instruments},"
            f" Timbre: {librosa_tags['timbre']},"
            f" Loudness: {librosa_tags['loudness']}\n\n"
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
        librosa_tags: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
        building_type: str,
    ) -> str:
        """
        Generate the architectural concept by building the prompt and querying the LLM.
        """
        prompt = self.build_description_prompt(
            librosa_tags, genre_tags, instrument_tags, building_type
        )
        return self._run_and_truncate(prompt)

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

        sentences = re.findall(r".+?[\.!?](?:\s|$)", text)
        if sentences:
            return sentences[-1].strip()
        # Fallback: hard cap at max_tokens*4 characters
        return text[: self.max_tokens * 4].strip()

    def _run_and_truncate(self, prompt: str) -> str:
        """
        Helper to call Ollama, truncate response, and return final text.
        """
        raw = self.call_ollama(prompt)
        return self.truncate_to_last_sentence(raw)


# Test from command line
def main():
    """
    CLI entry point for llm text generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate architectural prompts from musical analytics"
    )
    parser.add_argument("--llm", required=True, help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--building-type", required=True)
    parser.add_argument(
        "--librosa-tags",
        type=Path,
        required=True,
        help="Path to JSON file of librosa features",
    )
    parser.add_argument(
        "--genre-tags",
        type=Path,
        required=True,
        help="Path to JSON file of genre tuples",
    )
    parser.add_argument(
        "--instrument-tags",
        type=Path,
        required=True,
        help="Path to JSON file of instrument tuples",
    )
    parser.add_argument(
        "--output", type=Path, help="Where to save the generated prompts as JSON"
    )
    args = parser.parse_args()

    # Load inputs
    librosa_tags = json.loads(args.librosa_tags.read_text())
    genre_tags = json.loads(args.genre_tags.read_text())
    instrument_tags = json.loads(args.instrument_tags.read_text())

    # Instantiate and run
    gen = GenerateLLMDescription(
        llm=args.llm,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    desc = gen.generate_description(
        librosa_tags, genre_tags, instrument_tags, args.building_type
    )
    img_p = gen.summarise_for_image(desc)
    model_p = gen.summarise_for_3d(desc)

    result = {
        "description": desc,
        "image_prompt": img_p,
        "3d_prompt": model_p,
    }

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Prompts written to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
