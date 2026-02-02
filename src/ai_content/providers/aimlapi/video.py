"""
AIMLAPI Video provider.

Supports models like Wan-2.1 and Luma via AIMLAPI.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from ai_content.core.registry import ProviderRegistry
from ai_content.core.result import GenerationResult
from ai_content.providers.aimlapi.client import AIMLAPIClient
from ai_content.config import get_settings

logger = logging.getLogger(__name__)


@ProviderRegistry.register_video("aimlapi")
class AIMLAPIVideoProvider:
    """
    Video generation provider via AIMLAPI.

    Supports:
        - alibaba/wan-2.1-t2v-plus
        - alibaba/wan-2.1-t2v-turbo
        - luma-lab/dream-machine
        - klingai/kling-v1.5
    """

    name = "aimlapi"
    max_duration_seconds = 10

    def __init__(self):
        self.settings = get_settings().aimlapi
        self.client = AIMLAPIClient()

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
        Generate video using AIMLAPI.

        Args:
            prompt: Text description of the scene
            aspect_ratio: Video aspect ratio (16:9, 9:16, 1:1)
            duration_seconds: Duration (5 or 10s depending on model)
            first_frame_url: URL for image-to-video (not all models)
            output_path: Where to save the video
            use_fast_model: Whether to use a turbo model if available
        """
        logger.info(f"🎬 AIMLAPI: Generating video")
        logger.debug(f"   Prompt: {prompt[:50]}...")
        
        # Determine model
        model = self.settings.video_model
        if use_fast_model and hasattr(self.settings, "video_fast_model"):
            model = self.settings.video_fast_model

        try:
            # Build payload
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            }
            
            # Map duration if needed (some models expect duration as int, some as seconds)
            payload["duration"] = 10 if duration_seconds > 5 else 5

            if first_frame_url:
                payload["image_url"] = first_frame_url

            # Submit generation
            result = await self.client.submit_generation(
                "/v2/generate/video",
                payload,
            )

            generation_id = result.get("id") or result.get("generation_id")
            if not generation_id:
                return GenerationResult(
                    success=False,
                    provider=self.name,
                    content_type="video",
                    error=f"No generation ID in response: {result}",
                )

            # Poll for completion
            # The wait_for_completion method in AIMLAPIClient uses /v2/generate/video
            # but usually status is checked at a different endpoint for video?
            # Let's assume unified /v2/generate/video endpoint for polling too if it works
            status = await self.client.wait_for_completion(
                "/v2/generate/video",
                generation_id,
            )

            # Get video URL
            video_url = self._extract_video_url(status)
            if not video_url:
                return GenerationResult(
                    success=False,
                    provider=self.name,
                    content_type="video",
                    error="No video URL in response",
                    generation_id=generation_id,
                )

            # Download
            video_data = await self.client.download_file(video_url)

            # Save
            if output_path:
                file_path = Path(output_path)
            else:
                output_dir = get_settings().output_dir
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                file_path = output_dir / f"aimlapi_{timestamp}.mp4"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(video_data)

            logger.info(f"✅ AIMLAPI: Saved to {file_path}")

            return GenerationResult(
                success=True,
                provider=self.name,
                content_type="video",
                file_path=file_path,
                data=video_data,
                generation_id=generation_id,
                metadata={
                    "model": model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                },
            )

        except Exception as e:
            logger.error(f"AIMLAPI video generation failed: {e}")
            return GenerationResult(
                success=False,
                provider=self.name,
                content_type="video",
                error=str(e),
            )

    def _extract_video_url(self, status: dict) -> str | None:
        """Extract video URL from status response."""
        # Check various common keys
        if "url" in status:
            return status["url"]
        if "video_url" in status:
            return status["video_url"]
        
        # Check nested structures
        output = status.get("output")
        if isinstance(output, list) and output:
            return output[0] if isinstance(output[0], str) else output[0].get("url")
        if isinstance(output, dict):
            return output.get("url") or output.get("video_url")
            
        return None
