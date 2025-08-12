# ./flask_server/modules/services/meshy_service.py
from __future__ import annotations

import logging
import re
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from flask import current_app

from flask_server.modules.logger import default_logger


class MeshyService:
    """
    Minimal Meshy API wrapper that:
      - creates preview (text-to-3d)
      - optionally refines/textures
      - downloads GLB + textures
      - saves into flask_server/static/models, returning a relative path 'models/<file>'
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.meshy.ai/openapi/v2",
        logger: Optional[logging.Logger] = None,
    ):
        # Pull API key from Flask config if not passed explicitly
        if api_key is None:
            api_key = current_app.config["APP_SETTINGS"].MESHY_API_KEY
        if not api_key:
            raise RuntimeError(
                "No API key provided. Set MESHY_API_KEY in settings or pass api_key=..."
            )

        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.logger = logger or default_logger(__name__)

    # ---------------- internal helpers ----------------
    def _models_root(self) -> Path:
        """Return absolute path to static/models (creating it if missing)."""
        root = Path(__file__).parents[2] / "static" / "models"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _safe_filename(self, base: str, ext: str = ".glb") -> str:
        """Sanitize a base name and append timestamp + extension."""
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{slug or 'model'}_{ts}{ext}"

    def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        self.logger.info(f"POST {url}")
        r = requests.post(url, json=json, headers=self.headers, timeout=60)
        if r.status_code >= 400:
            self.logger.error(f"POST {path} failed {r.status_code}: {r.text}")
            raise RuntimeError(f"POST {path} failed {r.status_code}: {r.text}")
        data = r.json()
        return data["result"]

    def _get_task(self, task_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/text-to-3d/{task_id}"
        r = requests.get(url, headers=self.headers, timeout=60)
        if r.status_code >= 400:
            self.logger.error(f"GET task {task_id} failed {r.status_code}: {r.text}")
            raise RuntimeError(f"GET task {task_id} failed {r.status_code}: {r.text}")
        return r.json()

    def _wait(
        self, task_id: str, poll: int = 6, timeout: int = 60 * 20
    ) -> Dict[str, Any]:
        """Polls until SUCCEEDED/FAILED/CANCELED or timeout."""
        self.logger.info(
            f"Waiting for task {task_id} (poll={poll}s, timeout={timeout}s)"
        )
        t0 = time.time()
        last_status = None
        while True:
            data = self._get_task(task_id)
            status = data.get("status")
            if status != last_status:
                self.logger.info(f"Task {task_id} status: {status}")
                last_status = status
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return data
            if time.time() - t0 > timeout:
                self.logger.error(f"Task {task_id} timed out after {timeout}s")
                raise TimeoutError(f"Task {task_id} timed out")
            time.sleep(poll)

    def _download(self, url: str, out_path: Path) -> None:
        self.logger.info(f"Downloading {url} → {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, headers=self.headers, timeout=60) as r:
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                self.logger.error(f"Download failed {url}: {e} | Body: {r.text[:200]}")
                raise
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        self.logger.info(f"Saved file: {out_path}")

    # ---------------- API methods ----------------
    def create_preview(
        self,
        prompt: str,
        *,
        ai_model: str = "meshy-5.1",
        symmetry_mode: str = "off",
        topology: str = "triangle",
        target_polycount: int = 300_000,
        should_remesh: bool = True,
        art_style: str = "sculpture",
        seed: Optional[int] = None,
    ) -> str:
        """Create ONE preview and return task_id."""
        payload: Dict[str, Any] = {
            "mode": "preview",
            "ai_model": ai_model,
            "prompt": prompt,
            "art_style": art_style,
            "topology": topology,
            "target_polycount": target_polycount,
            "should_remesh": should_remesh,
            "symmetry_mode": symmetry_mode,
        }
        if seed is not None:
            payload["seed"] = seed
        self.logger.info(
            f"Creating preview | model={ai_model}, topology={topology}, poly={target_polycount}, seed={seed}"
        )
        return self._post("/text-to-3d", payload)

    def create_preview_variations(
        self,
        prompt: str,
        out_dir: Optional[str] = None,
        num_variations: int = 4,
        **preview_kwargs,
    ) -> None:
        """Fire multiple previews with different seeds and save thumbs/GLB when available."""
        out_dir_path = self._models_root() if out_dir is None else Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        for i in range(num_variations):
            seed = random.randint(0, 2**31 - 1)
            self.logger.info(f"=== Variation {i+1}/{num_variations} | Seed: {seed} ===")
            task_id = self.create_preview(prompt, seed=seed, **preview_kwargs)
            result = self._wait(task_id)

            status = result.get("status")
            if status != "SUCCEEDED":
                self.logger.warning(f"Preview failed for seed {seed}: status={status}")
                continue

            # Optional preview image
            preview = (
                result.get("preview") or result.get("result", {}).get("draft") or {}
            )
            thumbs = preview.get("images") or []
            if thumbs:
                img_path = out_dir_path / f"{task_id}_preview.jpg"
                self._download(thumbs[0], img_path)
                self.logger.info(f"Saved preview image: {img_path}")

            # GLB into static/models
            model_urls = result.get("model_urls") or {}
            if model_urls.get("glb"):
                fname = self._safe_filename(task_id, ".glb")
                glb_path = out_dir_path / fname
                self._download(model_urls["glb"], glb_path)
                self.logger.info(
                    f"Saved preview model: {glb_path} (relative: models/{fname})"
                )

        self.logger.info("Finished all variations.")

    def refine_texture(
        self,
        preview_task_id: str,
        *,
        enable_pbr: bool = True,
        ai_model: Optional[str] = None,
        texture_prompt: Optional[str] = None,
        texture_image_url: Optional[str] = None,
    ) -> str:
        """Start a refine/texture job from a successful preview task. Returns refine task_id."""
        payload: Dict[str, Any] = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
            "enable_pbr": enable_pbr,
        }
        if ai_model:
            payload["ai_model"] = ai_model
        if texture_prompt:
            payload["texture_prompt"] = texture_prompt
        if texture_image_url:
            payload["texture_image_url"] = texture_image_url

        self.logger.info(
            f"Refine start | preview_task_id={preview_task_id}, enable_pbr={enable_pbr}, ai_model={ai_model}"
        )
        return self._post("/text-to-3d", payload)

    def download_result_assets(
        self,
        task_result: dict,
        out_dir: Optional[str] = None,
        name_hint: Optional[str] = None,
    ) -> dict:
        """
        Download GLB + textures into static/models by default.
        Returns dict with absolute paths, and 'relative' key for DB use: models/<filename>
        """
        out_dir_path = self._models_root() if out_dir is None else Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Any] = {}

        # GLB
        model_urls = task_result.get("model_urls") or {}
        if model_urls.get("glb"):
            fname = self._safe_filename(
                name_hint or task_result.get("id", "model"), ".glb"
            )
            glb_path = out_dir_path / fname
            self._download(model_urls["glb"], glb_path)
            saved["glb"] = str(glb_path)
            saved["relative"] = f"models/{fname}"
            self.logger.info(f"Saved GLB: {glb_path} (relative: models/{fname})")
        else:
            self.logger.warning("No GLB URL present in task_result.model_urls")

        # Textures (if present)
        textures = (
            task_result.get("textures")
            or task_result.get("result", {}).get("textures")
            or {}
        )
        if isinstance(textures, dict) and textures:
            for name, url in textures.items():
                tex_path = (
                    out_dir_path
                    / f"{(name_hint or task_result.get('id','model'))}_{name}{Path(url).suffix or '.png'}"
                )
                self._download(url, tex_path)
                saved.setdefault("textures", {})[name] = str(tex_path)
                self.logger.info(f"Saved texture {name}: {tex_path}")

        # Preview image (optional)
        preview = (
            task_result.get("preview")
            or task_result.get("result", {}).get("draft")
            or {}
        )
        thumbs = preview.get("images") or []
        if thumbs:
            img_path = (
                out_dir_path
                / f"{(name_hint or task_result.get('id','model'))}_thumb.jpg"
            )
            self._download(thumbs[0], img_path)
            saved["preview_image"] = str(img_path)
            self.logger.info(f"Saved preview image: {img_path}")

        return saved

    def preview_then_texture(
        self,
        prompt: str,
        *,
        out_dir: Optional[str] = None,
        preview_kwargs: Optional[dict] = None,
        texture_kwargs: Optional[dict] = None,
        name_hint: Optional[str] = None,
    ) -> dict:
        """
        Runs preview, waits, then refine/texture, waits, then downloads assets.
        Returns dict with 'relative' key you can store (e.g., 'models/House_20250810_141200.glb').
        """
        preview_kwargs = preview_kwargs or {}
        texture_kwargs = texture_kwargs or {}

        self.logger.info("Starting preview_then_texture pipeline")
        self.logger.info(f"Preview kwargs: {preview_kwargs}")
        self.logger.info(f"Texture  kwargs: {texture_kwargs}")

        # 1) preview
        preview_id = self.create_preview(prompt, **preview_kwargs)
        self.logger.info(f"Preview task_id: {preview_id}")
        prev_res = self._wait(preview_id)
        if prev_res.get("status") != "SUCCEEDED":
            self.logger.error(f"Preview failed: {prev_res}")
            raise RuntimeError(f"Preview failed: {prev_res}")

        # 2) refine/texture
        refine_id = self.refine_texture(preview_id, **texture_kwargs)
        self.logger.info(f"Refine task_id: {refine_id}")
        ref_res = self._wait(refine_id)
        if ref_res.get("status") != "SUCCEEDED":
            self.logger.error(f"Refine failed: {ref_res}")
            raise RuntimeError(f"Refine failed: {ref_res}")

        # 3) download
        saved = self.download_result_assets(
            ref_res, out_dir=out_dir, name_hint=name_hint
        )
        self.logger.info(f"Finished pipeline. Saved assets: {saved}")
        return saved
