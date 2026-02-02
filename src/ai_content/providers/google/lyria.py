"""
Google Lyria RealTime music provider.

Uses WebSocket streaming for real-time music generation.
"""

import asyncio
import logging
import wave
import io
from datetime import datetime, timezone
from pathlib import Path

from ai_content.core.registry import ProviderRegistry
from ai_content.core.result import GenerationResult
from ai_content.core.exceptions import (
    ProviderError,
    AuthenticationError,
    GenerationError,
)
from ai_content.config import get_settings

logger = logging.getLogger(__name__)


@ProviderRegistry.register_music("lyria")
class GoogleLyriaProvider:
    """
    Google Lyria RealTime music provider.

    Features:
        - Real-time streaming generation
        - Weighted prompt support
        - BPM and temperature control
        - Instrumental only (no vocals)

    Example:
        >>> provider = GoogleLyriaProvider()
        >>> result = await provider.generate(
        ...     "smooth jazz fusion",
        ...     bpm=95,
        ...     duration_seconds=30,
        ... )
    """

    name = "lyria"
    supports_vocals = False
    supports_realtime = True
    supports_reference_audio = False

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
                    raise AuthenticationError("lyria")
                # Lyria RealTime requires v1alpha API version
                # https://ai.google.dev/gemini-api/docs/music-generation
                self._client = genai.Client(
                    api_key=api_key,
                    http_options={"api_version": "v1alpha"},
                )
            except ImportError:
                raise ProviderError(
                    "lyria",
                    "google-genai package not installed. Run: pip install google-genai",
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        bpm: int = 120,
        duration_seconds: int = 30,
        lyrics: str | None = None,
        reference_audio_url: str | None = None,
        output_path: str | None = None,
        temperature: float = 1.0,
    ) -> GenerationResult:
        """
        Generate music using Lyria RealTime streaming.

        Note:
            Lyria does not support vocals or lyrics. The lyrics parameter
            is ignored for compatibility with the MusicProvider protocol.
        """
        from google.genai import types

        client = self._get_client()

        logger.info(f"🎵 Lyria: Generating {duration_seconds}s at {bpm} BPM")
        logger.debug(f"   Prompt: {prompt[:50]}...")

        if lyrics:
            logger.warning("Lyria does not support vocals/lyrics. Ignoring lyrics parameter.")

        audio_chunks: list[bytes] = []
        capture_done = asyncio.Event()
        chunk_count = 0

        async def receive_audio(session):
            """Receive audio from Lyria stream in dedicated coroutine."""
            nonlocal chunk_count
            try:
                async for message in session.receive():
                    if hasattr(message, "server_content") and message.server_content:
                        if hasattr(message.server_content, "audio_chunks"):
                            for chunk in message.server_content.audio_chunks:
                                if hasattr(chunk, "data") and chunk.data:
                                    audio_chunks.append(chunk.data)
                                    chunk_count += 1
                    await asyncio.sleep(0)  # Yield control
                    if capture_done.is_set():
                        break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Audio receive error: {e}")

        try:
            async with client.aio.live.music.connect(model=self.settings.music_model) as session:
                logger.info("   ✓ Connection established")

                # Start receiver task first
                receive_task = asyncio.create_task(receive_audio(session))

                # Set weighted prompts
                await session.set_weighted_prompts(
                    prompts=[types.WeightedPrompt(text=prompt, weight=1.0)]
                )
                logger.info("   ✓ Prompt configured")

                # Configure generation
                await session.set_music_generation_config(
                    config=types.LiveMusicGenerationConfig(
                        bpm=bpm,
                        temperature=temperature,
                    )
                )
                logger.info(f"   ✓ Config set (BPM={bpm})")

                # Start streaming
                await session.play()
                logger.info(f"   ▶ Streaming for {duration_seconds}s...")

                # Wait for duration (robust: uses asyncio.sleep)
                await asyncio.sleep(duration_seconds)

                # Stop cleanly
                logger.info(f"   ⏸ Stopping... ({chunk_count} chunks received)")
                capture_done.set()
                await session.stop()
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"Lyria generation failed: {e}")
            return GenerationResult(
                success=False,
                provider=self.name,
                content_type="music",
                error=str(e),
            )

        if not audio_chunks:
            return GenerationResult(
                success=False,
                provider=self.name,
                content_type="music",
                error="No audio data received",
            )

        raw_audio_data = b"".join(audio_chunks)

        if not raw_audio_data:
            return GenerationResult(
                success=False,
                provider=self.name,
                content_type="music",
                error="No audio data received",
            )

        # Create WAV file in memory with proper header
        # Lyria RealTime uses 16-bit PCM, 48kHz, Stereo
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 2 bytes for 16-bit
                wav_file.setframerate(48000)  # 48kHz
                wav_file.writeframes(raw_audio_data)
            audio_data = wav_io.getvalue()

        # Save if output path provided
        file_path = None
        if output_path:
            file_path = Path(output_path)
        else:
            output_dir = get_settings().output_dir
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_path = output_dir / f"lyria_{timestamp}.wav"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(audio_data)

        logger.info(f"✅ Lyria: Saved to {file_path}")

        return GenerationResult(
            success=True,
            provider=self.name,
            content_type="music",
            file_path=file_path,
            data=audio_data,
            duration_seconds=duration_seconds,
            metadata={
                "bpm": bpm,
                "temperature": temperature,
                "prompt": prompt,
            },
        )
