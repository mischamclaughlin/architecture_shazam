# ./flask_server/modules/generators/image_generation.py
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

from flask_server.modules.logger import default_logger


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
    guidance_scale: float
    height: int
    width: int
    negative_prompt: Optional[str]
    scheduler: str
    device: str


class GenerateImage:
    """
    Generate and save Stable Diffusion XL images for a song, plus a metadata sidecar.
    """

    def __init__(
        self,
        llm: str,
        song_name: str,
        img_prompt: str,
        num_inference_steps: int,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        guidance_scale: float = 6.5,
        height: int = 1024,
        width: int = 1024,
        negative_prompt: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm
        self.song_name = (
            song_name.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )
        self.img_prompt = img_prompt
        self.num_inference_steps = int(num_inference_steps)
        self.model_name = model_name
        self.guidance_scale = float(guidance_scale)
        self.height = int(height)
        self.width = int(width)
        self.negative_prompt = negative_prompt
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.device = self._pick_device()
        self.dtype = self._pick_dtype(self.device)
        self.logger = logger or default_logger(__name__)

        project_root = Path(__file__).parents[2]
        self.save_dir = project_root / "static" / "images"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._pipe: Optional[StableDiffusionXLPipeline] = None

    # ---------------- internal: device/dtype ----------------
    def _pick_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _pick_dtype(self, device: torch.device):
        # SDXL typically benefits from fp16 on CUDA; use fp32 elsewhere
        return torch.float16 if device.type == "cuda" else torch.float32

    # ---------------- model loading ----------------
    def load_model(self) -> StableDiffusionXLPipeline:
        if self._pipe is None:
            self.logger.info(
                f"Loading SDXL: {self.model_name} on {self.device} ({self.dtype})"
            )
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None,
                add_watermarker=False,
            )
            # Force DPM-Solver++ (multistep)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )

            # memory-friendly toggles
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()

            pipe = pipe.to(self.device)

            # quick warmup (safe to ignore failures)
            try:
                _ = pipe(prompt="warmup", num_inference_steps=1).images[0]
            except Exception:
                pass

            self._pipe = pipe
        return self._pipe

    # ---------------- generation ----------------
    def create_image(self) -> Image.Image:
        prompt = (self.img_prompt or "").strip()
        if not prompt:
            raise ValueError("Empty SDXL prompt")

        self.logger.info(
            f"SDXL prompt (steps={self.num_inference_steps}, guidance={self.guidance_scale}, "
            f"size={self.width}x{self.height}): {prompt[:140]}{'…' if len(prompt) > 140 else ''}"
        )

        pipe = self.load_model()
        out = pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            return_dict=True,
            output_type="pil",
        )
        return out.images[0]

    # ---------------- saving ----------------
    def save_image(self) -> Path:
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
        pipe = self.load_model()
        meta = ImageMetadata(
            model=self.llm,
            timestamp=self.timestamp,
            audio_tags=audio_tags,
            genre_tags=genre_tags,
            instrument_tags=instrument_tags,
            llm_description=llm_description,
            sdxl_prompt=sdxl_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            negative_prompt=self.negative_prompt,
            scheduler=pipe.scheduler.__class__.__name__,
            device=str(self.device),
        )
        json_path = self.save_dir / f"metadata_{self.song_name}_{self.timestamp}.json"
        json_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        self.logger.info(f"Metadata saved to {json_path}")
        return json_path
