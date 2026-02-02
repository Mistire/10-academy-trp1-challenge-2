"""
Configuration settings using Pydantic.

Supports loading from YAML files and environment variables.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GoogleSettings(BaseSettings):
    """Google API configuration."""

    api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Models - aligned with official Gemini API docs
    # Video: https://ai.google.dev/gemini-api/docs/video
    video_model: str = "veo-3.1-generate-preview"
    video_fast_model: str = "veo-3.1-fast-generate-preview"
    video_legacy_model: str = "veo-2.0-generate-001"

    # Image: https://ai.google.dev/gemini-api/docs/image-generation
    image_model: str = "gemini-3-pro-image-preview"  # Nano Banana Pro - 4K, complex
    image_fast_model: str = "gemini-2.5-flash-image"  # Nano Banana - fast, 1024px

    # Music: https://ai.google.dev/gemini-api/docs/music-generation
    music_model: str = "models/lyria-realtime-exp"

    # Video defaults
    video_aspect_ratio: str = "16:9"
    video_resolution: str = "1080p"
    video_duration_seconds: int = 8
    video_person_generation: str = "allow_adult"

    # Image defaults
    image_aspect_ratio: str = "16:9"
    image_count: int = 1

    # Music defaults
    music_bpm: int = 120
    music_duration_seconds: int = 30
    music_temperature: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class AIMLAPISettings(BaseSettings):
    """AIMLAPI configuration."""

    api_key: str = Field(default="", alias="AIMLAPI_KEY")
    base_url: str = "https://api.aimlapi.com"

    # Models
    music_model: str = "minimax/music-2.0"
    lyria_model: str = "google/lyria2"
    kling_model: str = "klingai/v2.1-master-text-to-video"
    wan_model: str = "alibaba/wan2.1-t2v-plus"
    video_model: str = "alibaba/wan2.1-t2v-plus"
    video_fast_model: str = "alibaba/wan2.1-t2v-turbo"

    # Timeouts (MiniMax music can take 15-30 minutes)
    request_timeout: int = 60
    poll_interval: int = 10
    max_poll_attempts: int = 180  # 30 minutes at 10s intervals

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class KlingSettings(BaseSettings):
    """Direct KlingAI API configuration."""

    api_key: str = Field(default="", alias="KLINGAI_API_KEY")
    secret_key: str = Field(default="", alias="KLINGAI_SECRET_KEY")
    base_url: str = "https://api.klingai.com"

    # Defaults
    model: str = "kling-v2-master"
    aspect_ratio: str = "16:9"
    duration: str = "5"
    mode: str = "std"

    # Timeouts (Kling can take 5-14 minutes)
    poll_interval: int = 30
    max_poll_attempts: int = 30

    model_config = SettingsConfigDict(
        env_prefix="KLINGAI_",
        extra="ignore",
    )


class PollinationsSettings(BaseSettings):
    """Pollinations.ai configuration."""

    api_key: str = Field(default="", alias="POLLINATIONS_API_KEY")
    base_url: str = "https://gen.pollinations.ai"

    # Models
    video_model: str = "seedance"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class Settings(BaseSettings):
    """
    Main application settings.

    Loads from environment variables and optional YAML config.

    Example:
        >>> settings = Settings()
        >>> print(settings.google.api_key)
        >>> print(settings.output_dir)
    """

    # Project
    project_name: str = "AI Content Generator"
    version: str = "0.1.0"

    # Paths
    output_dir: Path = Path("./exports")
    config_path: Path | None = None

    # Provider selection
    default_music_provider: str = "lyria"
    default_video_provider: str = "veo"
    default_image_provider: str = "imagen"

    # Enable/disable providers
    use_google: bool = True
    use_aimlapi: bool = True
    use_kling_direct: bool = True

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Nested settings
    google: GoogleSettings = Field(default_factory=GoogleSettings)
    aimlapi: AIMLAPISettings = Field(default_factory=AIMLAPISettings)
    kling: KlingSettings = Field(default_factory=KlingSettings)
    pollinations: PollinationsSettings = Field(default_factory=PollinationsSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance (lazy-loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Configured Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure(config_path: str | Path | None = None, **overrides) -> Settings:
    """
    Configure settings with optional YAML file and overrides.

    Args:
        config_path: Path to YAML config file
        **overrides: Direct setting overrides

    Returns:
        Configured Settings instance
    """
    global _settings

    kwargs = {}
    if config_path:
        from ai_content.config.loader import load_yaml_config

        kwargs = load_yaml_config(config_path)

    kwargs.update(overrides)
    _settings = Settings(**kwargs)
    return _settings
