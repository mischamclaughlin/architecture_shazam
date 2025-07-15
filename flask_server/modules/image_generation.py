import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast, List, Tuple

import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)

from .logger import default_logger


@dataclass
class ImageMetadata:
    model: str
    timestamp: str
    audio_tags: Dict[str, Any]
    genre_tags: List[Tuple[str, float]]
    instrument_tags: List[Tuple[str, float]]
    llm_description: str
    sdxl_prompt: str
    num_inference_steps: int
    device: str


class GenerateImage:
    """
    Generate and save Stable Diffusion XL images for a song, plus metadata.
    """

    def __init__(
        self,
        llm: str,
        song_name: str,
        img_prompt: str,
        num_inference_steps: int,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialise with LLM name, song identifier, image prompt, inference steps, SDXL model and logger.
        """
        self.llm = llm
        self.song_name = (
            song_name.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )
        self.img_prompt = img_prompt
        self.num_inference_steps = num_inference_steps
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = self.pick_device()
        self.logger = logger or default_logger(__name__)

        # Safe directory naming
        project_root = Path(__file__).parent.parent
        self.save_dir = project_root / "static" / "images"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._pipe: Optional[StableDiffusionXLPipeline] = None

    def pick_device(self) -> torch.device:
        """
        Select the best available device: CUDA > MPS > CPU.
        """
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
        print(f"Using device: {dev}")
        return dev

    def load_model(self) -> StableDiffusionXLPipeline:
        """
        Lazy-load the Stable Diffusion XL pipeline and warm up.
        """
        if self._pipe is None:
            self.logger.info(f"Loading SDXL model: {self.model_name} on {self.device}")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            pipe = pipe.to(self.device)
            pipe(prompt="warmup", num_inference_steps=1)
            self._pipe = pipe
        assert self._pipe is not None
        return self._pipe

    def create_image(self) -> Image.Image:
        """
        Generate an image from the prompt using SDXL.
        """
        try:
            pipe = self.load_model()
            raw_output = pipe(
                prompt=self.img_prompt,
                num_inference_steps=self.num_inference_steps,
                return_dict=True,
            )
            result = cast(StableDiffusionXLPipelineOutput, raw_output)
            image = result.images[0]
            return image
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise

    def save_image(self) -> Path:
        """
        Generate and save the image, returning the file path.
        """
        img = self.create_image()
        filename = f"{self.song_name}_{self.timestamp}.png"
        output_path = self.save_dir / filename
        img.save(output_path)
        self.logger.info(f"Image saved to {output_path}")
        return output_path

    def save_metadata(
        self,
        audio_tags: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
        llm_description: str,
        sdxl_prompt: str,
    ) -> Path:
        """
        Save metadata about the generation to a JSON file, returning its path.
        """
        meta = ImageMetadata(
            model=self.llm,
            timestamp=self.timestamp,
            audio_tags=audio_tags,
            genre_tags=genre_tags,
            instrument_tags=instrument_tags,
            llm_description=llm_description,
            sdxl_prompt=sdxl_prompt,
            num_inference_steps=self.num_inference_steps,
            device=str(self.device),
        )
        json_path = self.save_dir / f"metadata_{self.timestamp}.json"
        json_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        self.logger.info(f"Metadata saved to {json_path}")
        return json_path


# Test from command line
def main():
    """
    CLI entry point for generating an image and metadata.
    """
    parser = argparse.ArgumentParser(
        description="Generate SDXL images for a song and save metadata."
    )
    parser.add_argument(
        "--llm", required=True, help="Name of the LLM used for descriptions."
    )
    parser.add_argument("--song-name", required=True, help="Identifier for the song.")
    parser.add_argument("--prompt", required=True, help="Image prompt for SDXL.")
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument(
        "--model-name",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model checkpoint to use.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Path to JSON file with audio_tags, llm_description, and sdxl_prompt.",
    )
    args = parser.parse_args()

    # Load metadata inputs if provided
    audio_tags: Dict[str, Any] = {}
    genre_tags: List[Tuple[str, float]] = []
    instrument_tags: List[Tuple[str, float]] = []
    llm_description: str = ""
    sdxl_prompt: str = args.prompt
    if args.metadata_json:
        data = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        audio_tags = data.get("audio_tags", {})
        genre_tags = data.get("genre_tags", {})
        instrument_tags = data.get("instrument_tags", {})
        llm_description = data.get("llm_description", "")

    generator = GenerateImage(
        llm=args.llm,
        song_name=args.song_name,
        img_prompt=args.prompt,
        num_inference_steps=args.steps,
        model_name=args.model_name,
    )
    image_path = generator.save_image()
    metadata_path = generator.save_metadata(
        audio_tags, genre_tags, instrument_tags, llm_description, sdxl_prompt
    )
    print(f"Done. Image: {image_path}, Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
