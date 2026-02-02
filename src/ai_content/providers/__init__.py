"""
Providers module.

Import this module to register all available providers.
"""

# Import provider modules to trigger registration
from ai_content.providers import google
from ai_content.providers import aimlapi
from ai_content.providers import pollinations
from ai_content.providers import kling

# Re-export providers
from ai_content.providers.google import (
    GoogleLyriaProvider,
    GoogleVeoProvider,
    GoogleImagenProvider,
)
from ai_content.providers.aimlapi import (
    AIMLAPIClient,
    MiniMaxMusicProvider,
    AIMLAPIVideoProvider,
)
from ai_content.providers.pollinations import (
    PollinationsVideoProvider,
)
from ai_content.providers.kling import (
    KlingDirectProvider,
)

__all__ = [
    # Google
    "GoogleLyriaProvider",
    "GoogleVeoProvider",
    "GoogleImagenProvider",
    # AIMLAPI
    "AIMLAPIClient",
    "MiniMaxMusicProvider",
    "AIMLAPIVideoProvider",
    # Pollinations
    "PollinationsVideoProvider",
    # Kling
    "KlingDirectProvider",
]
