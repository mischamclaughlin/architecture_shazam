# ./flask_server/modules/generators/description_generation.py
import logging
from typing import Any, Dict, List, Tuple, Optional
from ollama import chat, ResponseError, generate
from flask_server.modules.logger import default_logger

try:
    from transformers import CLIPTokenizerFast as _CLIPTokenizer
except Exception:
    _CLIPTokenizer = None


class GenerateLLMDescription:
    """
    Generates architectural design prompts based on musical analytics by interacting with an Ollama LLM.
    Also provides exact SDXL prompt trimming to <=77 CLIP tokens.
    """

    # Cache tokenisers across instances to avoid repeated downloads
    _tok1 = None
    _tok2 = None

    def __init__(
        self,
        llm: str,
        temperature: float,
        max_tokens: int,
        logger: Optional[logging.Logger] = None,
        enable_token_trim: bool = True,
        tok1_repo: str = "openai/clip-vit-large-patch14",
        tok2_repo: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger or default_logger(__name__)
        self.enable_token_trim = enable_token_trim
        self.tok1_repo = tok1_repo
        self.tok2_repo = tok2_repo

    def build_description_prompt(
        self,
        features: Dict[str, Any],
        genre_tags: List[Tuple[str, float]],
        instrument_tags: List[Tuple[str, float]],
        origin: Tuple,
        building_type: str,
    ) -> str:
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
            "3. Overall look (3-4 sentences): a magazine-style summary emphasising how the genre shapes the design language. Make sure to mention the building type.\n\n"
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
            "3. Overall look (3-4 sentences): a magazine-style summary emphasising how the genre shapes the design language. Make sure to mention the building type.\n\n"
            "Keep the total response under 500 words."
        )

    def build_image_prompt(self, raw_description: str) -> str:
        return (
            'Rewrite the following "overall look" description into a visual prompt for Stable Diffusion XL that depicts the entire building exterior—massing, roofline, elevation and context—rather than just a façade. '
            "Use vivid, concrete language to describe the full structure's form, materials, textures and overall silhouette from base to roof. "
            "Avoid interior details, abstract emotions or partial close-ups. "
            "Keep it concise and focused, max 75 tokens.\n\n"
            f"Description:\n{raw_description}\n\n"
            "Return only the image prompt. No extra commentary."
        )

    def build_3d_prompt(self, raw_description: str) -> str:
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
        prompt = self.build_image_prompt(raw_description)
        self.logger.info(f"LLM Summary Prompt:\n{prompt}")
        raw = self._run_and_truncate(prompt)

        # Precise SDXL trim (≤77 tokens) if enabled and tokenisers available
        final = self._sdxl_trim_to_77(raw) if self.enable_token_trim else raw
        self.logger.info(f"SDXL Prompt (≤77 tokens):\n{final}")
        return final

    def summarise_for_3d(self, raw_description: str) -> str:
        prompt = self.build_3d_prompt(raw_description)
        self.logger.info(f"LLM Summary Prompt:\n{prompt}")
        return self._run_and_truncate(prompt)

    def call_ollama(self, input_prompt: str) -> str:
        """
        Call ollama to run model and generate description.
        Try chat() first and fallback to generate(), more reliable with newer models.
        """
        try:
            r = chat(
                model=self.llm,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only the final prompt. No analysis or chain-of-thought.",
                    },
                    {"role": "user", "content": input_prompt},
                ],
                think=False,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            msg = (
                r.get("message") if isinstance(r, dict) else getattr(r, "message", None)
            )
            content = ""
            if msg:
                content = (
                    msg.get("content")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", "")
                ) or ""
                content = content.strip()
            if content:
                return content
            self.logger.warning(
                "chat() returned empty; falling back to generate(). Raw: %r", r
            )
        except ResponseError as e:
            self.logger.warning("chat() failed (%s). Falling back to generate().", e)

        r = generate(
            model=self.llm,
            prompt="Return only the final prompt. No analysis or chain-of-thought.\n\n"
            + input_prompt,
            think=False,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        text = (
            r.get("response") if isinstance(r, dict) else getattr(r, "response", "")
        ) or ""
        return text.strip()

    def truncate_to_last_sentence(self, text: str) -> str:
        idx = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if idx != -1 and idx < len(text) - 1:
            self.logger.info(f"LLM Summary:\n{text}")
            return text[: idx + 1]
        self.logger.info(f"LLM Summary:\n{text}")
        return text

    def _run_and_truncate(self, prompt: str) -> str:
        raw = self.call_ollama(prompt)
        self.logger.info(f"LLM RAW:\n{raw}")
        return self.truncate_to_last_sentence(raw)

    def _load_tokenizers(self):
        """
        Lazily load SDXL's two CLIP tokenisers.
        """
        if _CLIPTokenizer is None:
            self.logger.warning(
                "transformers not available; skipping precise CLIP trimming."
            )
            return None, None

        if GenerateLLMDescription._tok1 is None:
            try:
                GenerateLLMDescription._tok1 = _CLIPTokenizer.from_pretrained(
                    self.tok1_repo
                )
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer {self.tok1_repo}: {e}")

        if GenerateLLMDescription._tok2 is None:
            try:
                GenerateLLMDescription._tok2 = _CLIPTokenizer.from_pretrained(
                    self.tok2_repo
                )
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer {self.tok2_repo}: {e}")

        return GenerateLLMDescription._tok1, GenerateLLMDescription._tok2

    @staticmethod
    def _clip_trim(tok, text: str, user_max: int = 75) -> str:
        """
        Clamp to <= (user_max + 2) tokens with BOS/EOS preserved.
        """
        if tok is None:
            return text
        enc = tok(text, add_special_tokens=True, truncation=False)
        ids = enc["input_ids"]
        if not ids:
            return text

        bos = ids[0]
        eos = ids[-1] if tok.eos_token_id is None else tok.eos_token_id
        inner = ids[1:-1]
        if len(inner) <= user_max:
            return text

        inner = inner[:user_max]
        new_ids = [bos] + inner + [eos]

        # decode without special tokens to avoid stray markers
        return tok.decode(new_ids, skip_special_tokens=True).strip()

    def _sdxl_trim_to_77(self, text: str) -> str:
        """
        Ensure prompt fits in 77 tokens for BOTH SDXL text encoders.
        """
        tok1, tok2 = self._load_tokenizers()
        if tok1 is None or tok2 is None:
            words = text.split()
            if len(words) > 75:
                text = " ".join(words[:75])
            return text

        t = self._clip_trim(tok1, text, 75)
        t = self._clip_trim(tok2, t, 75)

        # pass again through tok1 to be extra safe
        t = self._clip_trim(tok1, t, 75)

        return t
