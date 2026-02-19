# FAQ & Troubleshooting

## Installation issues

### `ImportError: pyradiomics is required but not installed`

The PyPI release of pyradiomics is broken for Python ≥ 3.10. Install from GitHub:

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
```

See [pyradiomics #903](https://github.com/AIM-Harvard/pyradiomics/issues/903) for details.

### Build errors on Windows (`cl.exe not found`, `CMake Error`)

Pyradiomics has a C extension that requires a compiler. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **"Desktop development with C++"** workload, then retry:

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
```

### `ModuleNotFoundError: No module named 'matplotlib'`

Matplotlib is an **optional** dependency, only needed for interpretability analysis. Install it:

```bash
pip install matplotlib scikit-learn
```

---

## Usage issues

### Why are my FRDv0 and FRDv1 scores not comparable?

FRDv0 and FRDv1 use different feature sets, normalisation strategies, and output transforms:

| | FRDv0 | FRDv1 |
|---|---|---|
| Features | ~94 (Original only) | ~464 (Original + LoG + Wavelet) |
| Normalisation | min-max, joint | z-score, D1-referenced |
| Output | raw Fréchet distance | log-transformed |

Only compare scores computed with the same `frd_version`.

### What does `norm_ref` do and which should I use?

| Mode | Behaviour | When to use |
|---|---|---|
| `joint` | Normalise using stats from D1 ∪ D2 | FRDv0 default; general purpose |
| `d1` | Normalise both D1 and D2 using D1's stats only | FRDv1 default; matches the paper (Eq. 3) |
| `independent` | Each dataset normalised with its own stats | Required when using `.npz` files |

!!! warning
    When loading precomputed `.npz` statistics, `norm_ref` must be `"independent"` since the raw features are no longer available for joint or D1-referenced normalisation.

### Why are my shape features all the same?

Shape features (`shape_*`, `shape2D_*`) are computed from the **mask geometry**, not pixel intensities. When no mask is provided, frd-score creates a full-image mask — so all images of the same dimensions produce identical shape values (zero variance).

**Solution:** Use `--exclude_features shape` to remove them:

```bash
python -m frd_score data/real data/synthetic --exclude_features shape
```

```python
frd_value = compute_frd(paths, exclude_features=["shape"])
```

### How do I speed up feature extraction?

1. **Use multiple workers:** `--num_workers 8` (or set via Python: `num_workers=8`)
2. **Cache statistics:** Save with `--save_stats`, reuse the `.npz` file later
3. **Reduce feature scope:** Use fewer image types (`-I Original`) or feature groups (`-f firstorder glcm`)
4. **Resize images:** Use `--resize_size 128` to downsample large images

### What image formats are supported?

| Format | Dimensions | Extensions |
|---|---|---|
| PNG | 2D | `.png` |
| JPEG | 2D | `.jpg`, `.jpeg` |
| TIFF | 2D | `.tif`, `.tiff` |
| BMP | 2D | `.bmp` |
| NIfTI | 3D | `.nii`, `.nii.gz` |

All 2D images in a directory must be the same format. 3D NIfTI volumes are detected automatically.

### Can I compare 2D and 3D datasets?

No. Both datasets must have the same dimensionality. FRD extracts different feature sets for 2D and 3D images, so they cannot be compared directly.

---

## OOD detection issues

### What threshold is used for OOD detection?

By default, the 95th percentile of the reference distribution's L2 scores is used as the threshold. You can control the statistical assumption:

- `--id_dist_assumption gaussian` — parametric Gaussian fit (default)
- `--id_dist_assumption t` — Student's t-distribution
- `--id_dist_assumption counting` — non-parametric empirical percentile

### Should I use `--use_val_set`?

Using a held-out validation split (`--use_val_set --val_frac 0.2`) provides a less biased threshold estimate, but reduces the reference set size. It's recommended when you have ≥50 reference images.

---

## Interpretability issues

### The t-SNE plot is missing

`scikit-learn` is required for t-SNE. Install it:

```bash
pip install scikit-learn
```

If `scikit-learn` is not installed, t-SNE is skipped automatically with a warning.

### How do I interpret the feature ranking?

`interpret_frd()` ranks features by **squared mean difference** between the two distributions. The top features are those where the two datasets differ most. This helps identify:

- Which radiomic properties (texture, shape, intensity) drive the FRD score
- Whether differences are concentrated in a few features or spread across many
- Whether specific image filter types (LoG, Wavelet) contribute disproportionately
