__version__ = "1.0.0"

from .frd import (
    DEFAULT_BIN_WIDTH,
    DEFAULT_NORMALIZE_SCALE,
    DEFAULT_VOXEL_ARRAY_SHIFT,
    EXCLUDE_FIRSTORDER,
    EXCLUDE_OPTIONS,
    EXCLUDE_SHAPE,
    EXCLUDE_TEXTURAL,
    EXCLUDE_WAVELET,
    FRD_VERSION_DEFAULT,
    FRD_VERSION_V0,
    FRD_VERSION_V1,
    NORM_REF_D1,
    NORM_REF_DEFAULT,
    NORM_REF_INDEPENDENT,
    NORM_REF_JOINT,
    V0_DEFAULT_IMAGE_TYPES,
    V0_DEFAULT_NORM_REF,
    V1_DEFAULT_IMAGE_TYPES,
    V1_DEFAULT_NORM_REF,
    compute_frd,
    detect_ood,
    interpret_frd,
    save_frd_stats,
)

__all__ = [
    "__version__",
    # Core API
    "compute_frd",
    "save_frd_stats",
    "interpret_frd",
    "detect_ood",
    # Version constants
    "FRD_VERSION_V0",
    "FRD_VERSION_V1",
    "FRD_VERSION_DEFAULT",
    # Image type defaults
    "V0_DEFAULT_IMAGE_TYPES",
    "V1_DEFAULT_IMAGE_TYPES",
    # Normalization reference modes
    "NORM_REF_JOINT",
    "NORM_REF_D1",
    "NORM_REF_INDEPENDENT",
    "NORM_REF_DEFAULT",
    "V0_DEFAULT_NORM_REF",
    "V1_DEFAULT_NORM_REF",
    # Feature exclusion
    "EXCLUDE_TEXTURAL",
    "EXCLUDE_WAVELET",
    "EXCLUDE_FIRSTORDER",
    "EXCLUDE_SHAPE",
    "EXCLUDE_OPTIONS",
    # PyRadiomics defaults
    "DEFAULT_BIN_WIDTH",
    "DEFAULT_NORMALIZE_SCALE",
    "DEFAULT_VOXEL_ARRAY_SHIFT",
]
