"""Centralized configuration for stereo camera and disparity profile.

Edit these values to change behavior across the app and tools in one place.
"""
from __future__ import annotations

# Physical camera mapping
LEFT_INDEX: int = 1
RIGHT_INDEX: int = 3

# If True, swap left/right frames before rectification and disparity
SWAP_LR: bool = True

# Default disparity profile name to load from ./disparity_profiles
PROFILE_NAME: str = "CDR"
