"""
Video style presets.

Pre-configured prompt templates for common video styles.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoPreset:
    """
    Video generation preset.

    Attributes:
        name: Preset identifier
        prompt: Scene description prompt
        aspect_ratio: Video aspect ratio
        duration: Recommended duration in seconds
        style_keywords: Visual style keywords
    """

    name: str
    prompt: str
    aspect_ratio: str
    duration: int
    style_keywords: list[str]


# === Video Style Presets ===

NATURE = VideoPreset(
    name="nature",
    prompt="""A majestic lion slowly walks through tall savanna grass,
golden hour sunlight casting long shadows,
cinematic slow motion with a tracking dolly shot,
nature documentary style, 8K resolution, shallow depth of field,
professional color grading, David Attenborough aesthetic""",
    aspect_ratio="16:9",
    duration=5,
    style_keywords=["documentary", "wildlife", "golden-hour"],
)

URBAN = VideoPreset(
    name="urban",
    prompt="""Neon-lit streets of Tokyo at night, rain reflecting city lights,
cyberpunk aesthetic with holographic advertisements,
smooth steadicam following a lone figure walking,
Blade Runner inspired, moody and atmospheric,
4K cinematic, anamorphic lens flares""",
    aspect_ratio="16:9",
    duration=5,
    style_keywords=["cyberpunk", "urban", "neon"],
)

SPACE = VideoPreset(
    name="space",
    prompt="""A solitary astronaut floats in a space station observation deck,
slowly reaching toward Earth through the window,
soft Earth-glow lighting, reflective visor,
slow push-in camera movement, shallow depth of field,
Interstellar inspired, emotional and contemplative,
4K film grain, anamorphic lens""",
    aspect_ratio="16:9",
    duration=5,
    style_keywords=["sci-fi", "space", "contemplative"],
)

ABSTRACT = VideoPreset(
    name="abstract",
    prompt="""Flowing liquid metal morphing through impossible geometric shapes,
iridescent rainbow reflections on chrome surfaces,
macro lens extreme close-up, perfectly smooth motion,
satisfying abstract art, high-end commercial quality,
8K resolution, pristine studio lighting""",
    aspect_ratio="1:1",
    duration=5,
    style_keywords=["abstract", "commercial", "satisfying"],
)

OCEAN = VideoPreset(
    name="ocean",
    prompt="""Crystal clear turquoise ocean waves gently rolling,
underwater camera rises to break the surface,
sunbeams filtering through water, bioluminescent particles,
slow motion wave break, pristine beach paradise,
4K underwater cinematography, vibrant colors""",
    aspect_ratio="16:9",
    duration=5,
    style_keywords=["ocean", "underwater", "paradise"],
)

FANTASY = VideoPreset(
    name="fantasy",
    prompt="""Ancient dragon soaring over misty mountain peaks at sunset,
massive wings catching golden light, scales gleaming,
epic crane shot following the dragon's flight path,
high fantasy aesthetic, Lord of the Rings inspired,
8K cinematic, volumetric fog and god rays""",
    aspect_ratio="16:9",
    duration=5,
    style_keywords=["fantasy", "dragon", "epic"],
)

PORTRAIT = VideoPreset(
    name="portrait",
    prompt="""Close-up portrait of a person with striking features,
soft studio lighting with subtle rim light,
shallow depth of field, gentle movement,
fashion photography aesthetic, high-end commercial,
4K beauty cinematography, flawless skin detail""",
    aspect_ratio="9:16",
    duration=5,
    style_keywords=["portrait", "fashion", "beauty"],
)


# Registry of all presets
VIDEO_PRESETS: dict[str, VideoPreset] = {
    preset.name: preset
    for preset in [
        NATURE,
        URBAN,
        SPACE,
        ABSTRACT,
        OCEAN,
        FANTASY,
        PORTRAIT,
    ]
}


def get_preset(name: str) -> VideoPreset:
    """
    Get a video preset by name.

    Args:
        name: Preset name (e.g., "nature", "space")

    Returns:
        VideoPreset instance

    Raises:
        KeyError: If preset not found
    """
    if name not in VIDEO_PRESETS:
        available = list(VIDEO_PRESETS.keys())
        raise KeyError(f"Video preset '{name}' not found. Available: {available}")
    return VIDEO_PRESETS[name]


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(VIDEO_PRESETS.keys())
