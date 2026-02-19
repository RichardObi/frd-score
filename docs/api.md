# API Reference

## Core functions

### `compute_frd`

```python
frd_score.compute_frd(
    paths,
    frd_version="v1",
    features=None,
    norm_type=None,
    norm_range=None,
    paths_masks=None,
    resize_size=None,
    verbose=False,
    save_features=False,
    norm_ref=None,
    num_workers=None,
    image_types=None,
    use_paper_log=False,
    log_sigma=None,
    config_path=None,
    bin_width=None,
    normalize_scale=None,
    voxel_array_shift=None,
    exclude_features=None,
    match_sample_count=False,
    means_only=False,
    settings_dict=None,
    interpret=False,
    interpret_dir="outputs/interpretability_visualizations",
)
```

Compute the Fréchet Radiomics Distance between two image distributions. This is the main entry point.

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
| `exclude_features` | `list[str]` \| `None` | Post-extraction exclusion: `"textural"`, `"wavelet"`, `"firstorder"` |
| `match_sample_count` | `bool` | Subsample larger set to match smaller |
| `means_only` | `bool` | Skip covariance (mean-only distance) |
| `settings_dict` | `dict` \| `None` | Arbitrary PyRadiomics settings |
| `interpret` | `bool` | Run interpretability analysis |
| `interpret_dir` | `str` | Output directory for interpretability plots |

**Returns:** `float` — the FRD score.

---

### `save_frd_stats`

```python
frd_score.save_frd_stats(
    paths,
    frd_version="v1",
    # ... same parameters as compute_frd (except interpret/means_only)
)
```

Compute and save feature statistics (mean, covariance) for a single distribution to a `.npz` file.

**Parameters:** Same as `compute_frd()`, except `paths[0]` is the input directory and `paths[1]` is the output `.npz` file path.

---

### `interpret_frd`

```python
frd_score.interpret_frd(
    feature_list,
    feature_names,
    viz_dir="outputs/interpretability_visualizations",
    run_tsne=True,
)
```

Run interpretability analysis on extracted radiomic features.

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

```python
frd_score.detect_ood(
    feature_list,
    detection_type="image",
    val_frac=0.1,
    use_val_set=False,
    id_dist_assumption="gaussian",
    output_dir="outputs/ood_predictions",
    seed=None,
    filenames=None,
)
```

Out-of-distribution detection using normalised radiomics features.

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

## Constants

```python
from frd_score import (
    FRD_VERSION_V0,           # "v0"
    FRD_VERSION_V1,           # "v1"
    FRD_VERSION_DEFAULT,      # "v1"
    V0_DEFAULT_IMAGE_TYPES,   # ["Original"]
    V1_DEFAULT_IMAGE_TYPES,   # ["Original", "LoG", "Wavelet"]
    NORM_REF_JOINT,           # "joint"
    NORM_REF_D1,              # "d1"
    NORM_REF_INDEPENDENT,     # "independent"
    NORM_REF_DEFAULT,         # None (resolved per-version)
    V0_DEFAULT_NORM_REF,      # "joint"
    V1_DEFAULT_NORM_REF,      # "d1"
    EXCLUDE_TEXTURAL,         # "textural"
    EXCLUDE_WAVELET,          # "wavelet"
    EXCLUDE_FIRSTORDER,       # "firstorder"
    EXCLUDE_OPTIONS,          # {"textural", "wavelet", "firstorder"}
    DEFAULT_BIN_WIDTH,        # 5
    DEFAULT_NORMALIZE_SCALE,  # 100
    DEFAULT_VOXEL_ARRAY_SHIFT,# 300
)
```
