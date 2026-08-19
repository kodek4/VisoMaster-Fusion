"""AlphaFace experiment profiles shared by the UI and inference path."""

from typing import NamedTuple


class AlphaFaceProfile(NamedTuple):
    model_name: str
    fast_runtime: bool
    lean_crops: bool


ALPHAFACE_DEFAULT_PROFILE = "Baseline"
ALPHAFACE_PROFILE_OPTIONS = (
    ALPHAFACE_DEFAULT_PROFILE,
    "Exact",
    "Fused",
    "Fused FP16",
)

ALPHAFACE_PROFILES = {
    "Baseline": AlphaFaceProfile("AlphaFace", False, False),
    "Exact": AlphaFaceProfile("AlphaFace Exact", True, True),
    "Fused": AlphaFaceProfile("AlphaFace Fused", True, True),
    "Fused FP16": AlphaFaceProfile("AlphaFace Fused FP16", True, True),
}

ALPHAFACE_MODEL_NAMES = tuple(
    profile.model_name for profile in ALPHAFACE_PROFILES.values()
)
ALPHAFACE_FP16_MODEL_NAME = ALPHAFACE_PROFILES["Fused FP16"].model_name


def get_alphaface_profile(profile_name: object) -> AlphaFaceProfile:
    if isinstance(profile_name, str):
        return ALPHAFACE_PROFILES.get(
            profile_name, ALPHAFACE_PROFILES[ALPHAFACE_DEFAULT_PROFILE]
        )
    return ALPHAFACE_PROFILES[ALPHAFACE_DEFAULT_PROFILE]
