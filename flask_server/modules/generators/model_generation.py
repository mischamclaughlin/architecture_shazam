# ./flask_server/modules/generators/3d_generation.py
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import numpy as np

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

from flask_server.modules.logger import default_logger


# Monkey-patch torch.from_numpy to enforce float32 on MPS devices
_orig_from_numpy = torch.from_numpy
torch.from_numpy = lambda arr: _orig_from_numpy(arr.astype(np.float32))

logger = default_logger(__name__)


class Generate3d:
    """
    Generate and export 3D meshes from text prompts via Shap-E diffusion.
    """

    def __init__(
        self,
        prompt: str,
        *,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        fp16_on_mps: bool = False,
        diffusion_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise the 3D generator.

        Args:
            prompt: Text prompt guiding the mesh generation.
            device: Torch device to use (CUDA, MPS, or CPU). If None, auto-selected.
            seed: Optional random seed for reproducibility.
            fp16_on_mps: Whether to enable half precision on MPS devices.
            diffusion_params: Override parameters for diffusion process.
        """
        self.prompt = prompt
        self.device = device or self._pick_device()
        self.fp16_on_mps = fp16_on_mps
        self.use_fp16 = self.device.type == "cuda" or (
            fp16_on_mps and self.device.type == "mps"
        )

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger.info(f"Random seed set to {seed}")

        self.diffusion_params = diffusion_params or {}

        # Load models
        logger.info("Loading transmitter model for sampling")
        logger.info("Loading text encoder model")
        self.model = load_model("text300M", device=self.device)

        config = load_config("diffusion")
        self.diffusion = diffusion_from_config(config)

    @staticmethod
    def _to_fp32_tensor(arr: np.ndarray) -> torch.Tensor:
        """Convert a NumPy array to a float32 torch Tensor."""
        return torch.from_numpy(arr)

    def _pick_device(self) -> torch.device:
        """Select the best available torch device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
        logger.info(f"Using device: {dev}")
        return dev

    def generate_latent(
        self,
        *,
        batch_size: int = 1,
        guidance_scale: float = 25.0,
        karras_steps: int = 256,
        sigma_min: float = 1e-8,
        sigma_max: float = 50.0,
        s_churn: float = 0.1,
        **override_params: Any,
    ) -> torch.Tensor:
        """
        Sample latent tensors from the diffusion model.

        Args:
            batch_size: Number of samples to generate.
            guidance_scale: Classifier-free guidance scale.
            karras_steps: Number of Karras sampling steps.
            sigma_min: Minimum sigma for Karras diffusion.
            sigma_max: Maximum sigma for Karras diffusion.
            s_churn: Churn parameter for Karras diffusion.
            **override_params: Additional diffusion parameters.

        Returns:
            A tensor of shape [batch_size, ...] containing the latents.
        """
        params = {
            **self.diffusion_params,
            **override_params,
            "batch_size": batch_size,
            "guidance_scale": guidance_scale,
            "model_kwargs": {"texts": [self.prompt] * batch_size},
            "progress": True,
            "clip_denoised": True,
            "use_fp16": self.use_fp16,
            "use_karras": True,
            "karras_steps": karras_steps,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "s_churn": s_churn,
        }

        return sample_latents(
            model=self.model,
            diffusion=self.diffusion,
            **params,
        )

    def save_meshes(
        self,
        title: str,
        *,
        batch_size: int = 1,
        guidance_scale: float = 25.0,
        **latent_kwargs: Any,
    ) -> str:
        """
        Decode latents to OBJ files and save to disk.

        Args:
            out_dir: Directory to save OBJ files.
            filename_prefix: Prefix for output filenames.
            batch_size: Number of samples to generate.
            guidance_scale: Classifier-free guidance scale.
            **latent_kwargs: Additional kwargs for generate_latent.

        Returns:
            List of file paths written.
        """
        base = Path(__file__).parents[2]
        out_dir: Path = base / "static" / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh_path: Optional[str] = None

        latents = self.generate_latent(
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            **latent_kwargs,
        )
        song_title = (
            title.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )

        for idx, latent in enumerate(latents):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{song_title}_{timestamp}.obj"
            path: Path = out_dir / filename

            try:
                logger.info(f"[{idx}] Decoding mesh")
                optimal_device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                xm = load_model("transmitter", device=optimal_device)
                latent = latent.to(optimal_device)
                raw_mesh = decode_latent_mesh(xm, latent).tri_mesh()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("Out of memory on device; retrying on CPU")
                    torch.cuda.empty_cache()
                    device = torch.device("cpu")
                    xm = load_model("transmitter", device=device)
                    latent = latent.to(device)
                    raw_mesh = decode_latent_mesh(xm, latent).tri_mesh()
                else:
                    logger.error(f"Decoding failed: {e}")
                    continue

            try:
                with open(path, "w", encoding="utf8") as f:
                    raw_mesh.write_obj(f)
                logger.info(f"Wrote OBJ -> {path}")
                mesh_path = f"models/{filename}"
            except Exception as e:
                logger.error(f"Failed to write OBJ {path}: {e}")

        if mesh_path is None:
            raise RuntimeError("Mesh generation failed, no file written")

        return mesh_path
