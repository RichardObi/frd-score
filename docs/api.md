# API Reference

## Core Functions

### `compute_frd`

::: frd_score.compute_frd
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `paths` | `list` | Two paths (directories, file lists, or `.npz` files) |
| `frd_version` | `str` | `"v0"` or `"v1"` (default) |
| `features` | `list[str]` \| `None` | Feature class names (e.g. `["firstorder", "glcm"]`). `None` = version default |
| `norm_type` | `str` \| `None` | `"minmax"` or `"zscore"`. `None` = version default |
| `norm_range` | `list[float]` \| `None` | `[min, max]` for normalisation. `None` = version default |
| `paths_masks` | `list[str]` \| `None` | Two mask directory paths |
| `resize_size` | `int` \| `tuple` \| `None` | Pixel resize target |
| `verbose` | `bool` | Enable detailed logging |
| `save_features` | `bool` | Save features to CSV |
| `norm_ref` | `str` \| `None` | `"joint"`, `"d1"`, or `"independent"`. `None` = version default |
| `num_workers` | `int` \| `None` | CPU workers. `None` = auto |
| `image_types` | `list[str]` \| `None` | PyRadiomics image types. `None` = version default |
| `use_paper_log` | `bool` | Use paper Eq. 3 log transform (v1 only) |
| `log_sigma` | `list[float]` \| `None` | LoG sigma values |
| `config_path` | `str` \| `None` | Custom PyRadiomics YAML config |
| `bin_width` | `int` \| `None` | PyRadiomics bin width |
| `normalize_scale` | `float` \| `None` | PyRadiomics normalise scale |
| `voxel_array_shift` | `float` \| `None` | PyRadiomics voxel array shift |
| `exclude_features` | `list[str]` \| `None` | Post-extraction exclusion: `"textural"`, `"wavelet"`, `"firstorder"`, `"shape"` |
| `match_sample_count` | `bool` | Subsample larger set to match smaller |
| `means_only` | `bool` | Skip covariance (mean-only distance) |
| `settings_dict` | `dict` \| `None` | Arbitrary PyRadiomics settings |
| `interpret` | `bool` | Run interpretability analysis |
| `interpret_dir` | `str` | Output directory for interpretability plots |

**Returns:** `float` — the FRD score.

---

### `save_frd_stats`

::: frd_score.save_frd_stats
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

**Parameters:** Same as `compute_frd()`, except `paths[0]` is the input directory and `paths[1]` is the output `.npz` file path.

---

### `interpret_frd`

::: frd_score.interpret_frd
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `feature_list` | `list[np.ndarray]` | Two arrays of shape `(n_samples, n_features)` |
| `feature_names` | `list[str]` | Feature names |
| `viz_dir` | `str` | Output directory |
| `run_tsne` | `bool` | Whether to produce t-SNE plot |

**Returns:** `dict` with `top_changed_features` and `n_features`.

---

### `detect_ood`

::: frd_score.detect_ood
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `feature_list` | `list[np.ndarray]` | `[D1_features, D2_features]` — D1 = reference, D2 = test |
| `detection_type` | `str` | `"image"` or `"dataset"` |
| `val_frac` | `float` | Fraction of D1 for threshold estimation |
| `use_val_set` | `bool` | Use hold-out validation split |
| `id_dist_assumption` | `str` | `"gaussian"`, `"t"`, or `"counting"` |
| `output_dir` | `str` | CSV output directory |
| `seed` | `int` \| `None` | Random seed |
| `filenames` | `list[str]` \| `None` | Filenames for CSV output |

**Returns:** `dict` — keys depend on `detection_type`:

- `"image"`: `threshold`, `scores`, `predictions`, `p_values`
- `"dataset"`: `nfrd`

---

### `calculate_frechet_distance`

::: frd_score.frd.calculate_frechet_distance
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

---

## Constants

### Quick import reference

```python
from frd_score import (
    # Core API
    compute_frd,
    save_frd_stats,
    interpret_frd,
    detect_ood,
    # Version constants
    FRD_VERSION_V0,           # "v0"
    FRD_VERSION_V1,           # "v1"
    FRD_VERSION_DEFAULT,      # "v1"
    # Image type defaults
    V0_DEFAULT_IMAGE_TYPES,   # ["Original"]
    V1_DEFAULT_IMAGE_TYPES,   # ["Original", "LoG", "Wavelet"]
    # Normalisation reference modes
    NORM_REF_JOINT,           # "joint"
    NORM_REF_D1,              # "d1"
    NORM_REF_INDEPENDENT,     # "independent"
    NORM_REF_DEFAULT,         # None (resolved per-version)
    V0_DEFAULT_NORM_REF,      # "joint"
    V1_DEFAULT_NORM_REF,      # "d1"
    # Feature exclusion
    EXCLUDE_TEXTURAL,         # "textural"
    EXCLUDE_WAVELET,          # "wavelet"
    EXCLUDE_FIRSTORDER,       # "firstorder"
    EXCLUDE_SHAPE,            # "shape"
    EXCLUDE_OPTIONS,          # {"textural", "wavelet", "firstorder", "shape"}
    # PyRadiomics defaults
    DEFAULT_BIN_WIDTH,        # 5
    DEFAULT_NORMALIZE_SCALE,  # 100
    DEFAULT_VOXEL_ARRAY_SHIFT,# 300
)
```
