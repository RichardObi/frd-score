# Quick Start

## Compute FRD via CLI

```bash
# Default: FRDv1
python -m frd_score path/to/dataset_A path/to/dataset_B

# FRDv0
python -m frd_score path/to/dataset_A path/to/dataset_B --frd_version v0

# With masks
python -m frd_score path/to/dataset_A path/to/dataset_B \
    -m path/to/masks_A path/to/masks_B

# Verbose output
python -m frd_score path/to/dataset_A path/to/dataset_B -v
```

## Compute FRD via Python

```python
from frd_score import compute_frd

# From directories
frd_value = compute_frd(["path/to/dataset_A", "path/to/dataset_B"]) # (1)

# From file lists
frd_value = compute_frd([
    ["img1.png", "img2.png"],
    ["img3.png", "img4.png"],
])

# With all options
frd_value = compute_frd(
    ["path/to/dataset_A", "path/to/dataset_B"],
    frd_version="v1",          # (2)
    paths_masks=["path/to/masks_A", "path/to/masks_B"],  # (3)
    norm_type="zscore",
    norm_ref="d1",             # (4)
    resize_size=256,
    verbose=True,
    save_features=True,        # (5)
    num_workers=4,
)
```

1. Accepts directories, file lists, or `.npz` statistics files.
2. Default is `"v1"`. Use `"v0"` for backward compatibility.
3. Optional mask directories for localised feature extraction.
4. Normalisation reference: `"joint"`, `"d1"` (paper default), or `"independent"`.
5. Saves extracted features to a CSV file for inspection.

## Save and reuse statistics

Precomputing statistics avoids re-extracting features when comparing against the same reference dataset multiple times:

```bash
# Save statistics
python -m frd_score --save_stats path/to/reference path/to/reference_stats.npz

# Reuse
python -m frd_score path/to/reference_stats.npz path/to/test_data
```

```python
from frd_score import save_frd_stats, compute_frd

# Save
save_frd_stats(["path/to/reference", "reference_stats.npz"]) # (1)

# Reuse
frd_value = compute_frd(["reference_stats.npz", "path/to/test_data"]) # (2)
```

1. Extracts features and saves mean/covariance to a `.npz` file.
2. Loads precomputed statistics instead of re-extracting features.

!!! note
    When using `.npz` files, `norm_ref` must be `"independent"` since the original features are no longer available for joint or D1-referenced normalisation.

## What to expect

- **FRDv1 values** are log-transformed and typically range from ~2 to ~15 for medical imaging datasets.
- **FRDv0 values** are raw Fréchet distances and can vary widely depending on the datasets.
- **Lower is better** — a value closer to 0 means the two distributions are more similar.
- Feature extraction is the slowest step. Use `--num_workers` to parallelise, and `--save_stats` to cache results.

## Feature exclusion

Exclude feature categories after extraction for ablation studies or to remove uninformative features:

```bash
# Exclude shape features (often constant when no mask is used)
python -m frd_score path/to/dataset_A path/to/dataset_B --exclude_features shape

# Exclude multiple categories
python -m frd_score path/to/dataset_A path/to/dataset_B --exclude_features shape wavelet
```

```python
frd_value = compute_frd(
    ["path/to/dataset_A", "path/to/dataset_B"],
    exclude_features=["shape", "wavelet"],
)
```

Available categories: `textural` (GLCM/GLRLM/GLSZM/NGTDM/GLDM), `wavelet`, `firstorder`, `shape` (shape/shape2D).

!!! tip "When to exclude shape features"
    Shape features (`shape_*`, `shape2D_*`) are computed from the mask geometry, not pixel intensities. When no mask is provided, frd-score creates a full-image mask, so all images of the same dimensions get **identical shape values** (zero variance). Excluding them avoids polluting the FRD with uninformative features.
