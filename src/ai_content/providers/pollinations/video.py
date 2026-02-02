"""
Pollinations.ai video provider.

Uses the Pollinations.ai API for video generation.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ai_content.core.registry import ProviderRegistry
from ai_content.core.result import GenerationResult
from ai_content.core.exceptions import (
    ProviderError,
    AuthenticationError,
)
from ai_content.config import get_settings

logger = logging.getLogger(__name__)


@ProviderRegistry.register_video("pollinations")
class PollinationsVideoProvider:
    """
    Video generation provider via Pollinations.ai.

    Features:
        - High-quality models like Wan 2.1
        - Supports API keys for higher rate limits
    """

    name = "pollinations"
    max_duration_seconds = 10

    def __init__(self):
        self.settings = get_settings().pollinations

    @property
    def headers(self) -> dict[str, str]:
        """Get authorization headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "16:9",
        duration_seconds: int = 5,
        first_frame_url: str | None = None,
        output_path: str | None = None,
        use_fast_model: bool = False,
    ) -> GenerationResult:
        """
        Generate video using Pollinations.ai via GET /image/{prompt}.
        """
        import urllib.parse
        
        model = self.settings.video_model
        # URL encode the prompt for the path
        encoded_prompt = urllib.parse.quote(prompt)
        base_endpoint = f"{self.settings.base_url}/image/{encoded_prompt}"

        # Build query parameters
        params = {
            "model": model,
            "aspect_ratio": aspect_ratio,
            "duration": str(duration_seconds),
        }
        
        # Add nologo if we have an API key
        if self.settings.api_key:
            params["nologo"] = "true"

        logger.info(f"🎬 Pollinations: Generating video with model='{model}'")
        logger.debug(f"   Prompt: {prompt[:50]}...")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.get(
                    base_endpoint,
                    params=params,
                    headers=self.headers,
                )
                
                if response.status_code == 401:
                    raise AuthenticationError("pollinations")
                
                response.raise_for_status()
                
                # The response is the direct video binary data
                video_data = response.content

            # Save
            if output_path:
                file_path = Path(output_path)
            else:
                output_dir = get_settings().output_dir
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = output_dir / f"pollinations_{timestamp}.mp4"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(video_data)

            logger.info(f"✅ Pollinations: Saved to {file_path}")

            return GenerationResult(
                success=True,
                provider=self.name,
                content_type="video",
                file_path=file_path,
                data=video_data,
                metadata={
                    "model": model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": duration_seconds,
                },
            )

        except Exception as e:
            logger.error(f"Pollinations generation failed: {e}")
            return GenerationResult(
                success=False,
                provider=self.name,
                content_type="video",
                error=str(e),
            )

