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
frd_value = compute_frd(["path/to/dataset_A", "path/to/dataset_B"])

# From file lists
frd_value = compute_frd([
    ["img1.png", "img2.png"],
    ["img3.png", "img4.png"],
])

# With all options
frd_value = compute_frd(
    ["path/to/dataset_A", "path/to/dataset_B"],
    frd_version="v1",
    paths_masks=["path/to/masks_A", "path/to/masks_B"],
    norm_type="zscore",
    norm_ref="d1",
    resize_size=256,
    verbose=True,
    save_features=True,
    num_workers=4,
)
```

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
save_frd_stats(["path/to/reference", "reference_stats.npz"])

# Reuse
frd_value = compute_frd(["reference_stats.npz", "path/to/test_data"])
```

!!! note
    When using `.npz` files, `norm_ref` must be `"independent"` since the original features are no longer available for joint or D1-referenced normalisation.

## What to expect

- **FRDv1 values** are log-transformed and typically range from ~2 to ~15 for medical imaging datasets.
- **FRDv0 values** are raw Fréchet distances and can vary widely depending on the datasets.
- **Lower is better** — a value closer to 0 means the two distributions are more similar.
- Feature extraction is the slowest step. Use `--num_workers` to parallelise, and `--save_stats` to cache results.
