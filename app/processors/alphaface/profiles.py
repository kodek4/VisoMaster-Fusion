"""AlphaFace experiment profiles shared by the UI and inference path."""

from typing import NamedTuple


class AlphaFaceProfile(NamedTuple):
    model_name: str
    fast_runtime: bool
    lean_crops: bool


class AlphaFaceQualityProfile(NamedTuple):
    y_shift_256: float
    direct_256_crop: bool


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

ALPHAFACE_DEFAULT_QUALITY_PROFILE = "Baseline"
ALPHAFACE_QUALITY_PROFILE_OPTIONS = (
    ALPHAFACE_DEFAULT_QUALITY_PROFILE,
    "Training Matched",
    "Training Matched Direct",
)
ALPHAFACE_QUALITY_PROFILES = {
    "Baseline": AlphaFaceQualityProfile(0.0, False),
    "Training Matched": AlphaFaceQualityProfile(-14.0, False),
    "Training Matched Direct": AlphaFaceQualityProfile(-14.0, True),
}


def get_alphaface_profile(profile_name: object) -> AlphaFaceProfile:
    if isinstance(profile_name, str):
        return ALPHAFACE_PROFILES.get(
            profile_name, ALPHAFACE_PROFILES[ALPHAFACE_DEFAULT_PROFILE]
        )
    return ALPHAFACE_PROFILES[ALPHAFACE_DEFAULT_PROFILE]


def get_alphaface_quality_profile(profile_name: object) -> AlphaFaceQualityProfile:
    if isinstance(profile_name, str):
        return ALPHAFACE_QUALITY_PROFILES.get(
            profile_name,
            ALPHAFACE_QUALITY_PROFILES[ALPHAFACE_DEFAULT_QUALITY_PROFILE],
        )
    return ALPHAFACE_QUALITY_PROFILES[ALPHAFACE_DEFAULT_QUALITY_PROFILE]
