"""
Google Veo video provider.

Uses async polling for video generation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from ai_content.core.registry import ProviderRegistry
from ai_content.core.result import GenerationResult
from ai_content.core.exceptions import (
    ProviderError,
    AuthenticationError,
    TimeoutError,
)
from ai_content.config import get_settings

logger = logging.getLogger(__name__)


@ProviderRegistry.register_video("veo")
class GoogleVeoProvider:
    """
    Google Veo 3.1 video provider.

    Features:
        - Text-to-video generation
        - Image-to-video (with first frame)
        - Multiple aspect ratios
        - Fast generation (~30 seconds typical)

    Example:
        >>> provider = GoogleVeoProvider()
        >>> result = await provider.generate(
        ...     "Dragon soaring over mountains",
        ...     aspect_ratio="16:9",
        ... )
    """

    name = "veo"
    supports_image_to_video = True
    max_duration_seconds = 8

    def __init__(self):
        self.settings = get_settings().google
        self._client = None

    def _get_client(self):
        """Lazy-load the Google GenAI client."""
        if self._client is None:
            try:
                from google import genai

                api_key = self.settings.api_key
                if not api_key:
                    raise AuthenticationError("veo")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ProviderError(
                    "veo",
                    "google-genai package not installed. Run: pip install google-genai",
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "16:9",
        duration_seconds: int = 5,
        first_frame_url: str | None = None,
        output_path: str | None = None,
        use_fast_model: bool = False,
        person_generation: str = "dont_allow",
    ) -> GenerationResult:
        """
        Generate video using Veo 3.1.

        Args:
            prompt: Scene description
            aspect_ratio: "16:9", "9:16", or "1:1"
            duration_seconds: Currently ignored (model determines)
            first_frame_url: Optional image URL to animate
            output_path: Where to save the video
            use_fast_model: Use faster but lower quality model
            person_generation: "allow_adult" or "dont_allow"
        """
        from google.genai import types

        client = self._get_client()

        model = (
            self.settings.video_fast_model
            if use_fast_model
            else self.settings.video_model
        )

        logger.info(f"🎬 Veo: Generating video ({aspect_ratio})")
        logger.debug(f"   Prompt: {prompt[:50]}...")
        logger.debug(f"   Model: {model}")

        if duration_seconds > self.max_duration_seconds:
            logger.warning(
                f"Veo supports max {self.max_duration_seconds}s. "
                f"Clamping {duration_seconds}s to limit."
            )

        # Validate aspect ratio
        supported_aspects = ["16:9", "9:16", "1:1"]
        if aspect_ratio not in supported_aspects:
            logger.warning(
                f"Veo does not support '{aspect_ratio}'. "
                f"Falling back to '16:9'."
            )
            aspect_ratio = "16:9"

        try:
            # Build config
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
            )

            # Generate
            if first_frame_url:
                # Image-to-video
                image_data = await self._fetch_image(first_frame_url)
                image = types.Image(image_bytes=image_data)
                operation = await client.aio.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=image,
                    config=config,
                )
            else:
                # Text-to-video
                operation = await client.aio.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    config=config,
                )

            # Poll until complete
            logger.info("   Waiting for generation...")
            while not operation.done:
                await asyncio.sleep(5)
                operation = await client.aio.operations.get(operation)

            if not operation.response or not operation.response.generated_videos:
                return GenerationResult(
                    success=False,
                    provider=self.name,
                    content_type="video",
                    error="No video generated",
                )

            # Get video data
            video = operation.response.generated_videos[0]
            video_data = video.video.video_bytes

            # Save
            if output_path:
                file_path = Path(output_path)
            else:
                output_dir = get_settings().output_dir
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = output_dir / f"veo_{timestamp}.mp4"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(video_data)

            logger.info(f"✅ Veo: Saved to {file_path}")

            return GenerationResult(
                success=True,
                provider=self.name,
                content_type="video",
                file_path=file_path,
                data=video_data,
                metadata={
                    "aspect_ratio": aspect_ratio,
                    "model": model,
                    "prompt": prompt,
                },
            )

        except Exception as e:
            logger.error(f"Veo generation failed: {e}")
            return GenerationResult(
                success=False,
                provider=self.name,
                content_type="video",
                error=str(e),
            )

    async def _fetch_image(self, url: str) -> bytes:
        """Fetch image data from URL."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
