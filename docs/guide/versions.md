# FRD Versions

The library ships two FRD formulations. Both compute a Fréchet distance between Gaussian-fitted radiomic feature distributions, but differ in extraction, normalisation, and post-processing.

## FRDv0 (Osuala et al., 2024)

The original formulation from [*Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models*](https://arxiv.org/abs/2403.13890).

| Setting | Value |
|---|---|
| Feature classes | `firstorder`, `glcm`, `glrlm`, `gldm`, `glszm`, `ngtdm`, `shape`, `shape2D` |
| Image types | `Original` only |
| Number of features | ~94 |
| Normalisation | Min-max to `[0, 7.457]` |
| Norm reference | `joint` (D1 ∪ D2) |
| Output transform | None (raw Fréchet distance) |

```python
frd_value = compute_frd(paths, frd_version="v0")
```

## FRDv1 (Konz, Osuala et al., 2026) — default

The extended formulation from [*Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets*](https://arxiv.org/abs/2412.01496).

| Setting | Value |
|---|---|
| Feature classes | `firstorder`, `glcm`, `glrlm`, `glszm`, `ngtdm` |
| Image types | `Original`, `LoG` (σ = 2, 3, 4, 5), `Wavelet` |
| Number of features | ~464 |
| Normalisation | Z-score to `[0, 1]` |
| Norm reference | `d1` (reference distribution only) |
| Output transform | `log(d²_F)` (log of squared Fréchet distance) |

```python
frd_value = compute_frd(paths, frd_version="v1")  # default
```

### Paper log transform

The default log transform computes `FRD = log(d²_F)`. The paper (Eq. 3) defines `FRD = log(d_F) = log(√d²_F) = 0.5 × log(d²_F)`. To use the paper formula:

```python
frd_value = compute_frd(paths, use_paper_log=True)
```

!!! warning
    `use_paper_log=True` produces values that are **half** of the default and are NOT directly comparable to previously published FRD values.

## Choosing a version

- Use **v1** (default) for new experiments. It is more robust and produces better-calibrated scores.
- Use **v0** if you need backward compatibility with prior FRDv0 measurements.

## Version-specific defaults

All version-specific defaults are resolved automatically, but can be overridden:

```python
# Override normalisation for v1
frd_value = compute_frd(
    paths,
    frd_version="v1",
    norm_type="minmax",          # override: use min-max instead of z-score
    norm_ref="joint",            # override: use joint normalisation
    image_types=["Original"],    # override: skip LoG and Wavelet
)
```

## Normalisation reference modes

| `norm_ref` | Description |
|---|---|
| `joint` | Normalise using statistics from D1 ∪ D2 concatenated. Default for v0. |
| `d1` | Normalise both D1 and D2 using D1's statistics only (paper Eq. 3). Default for v1. |
| `independent` | Each dataset normalised using only its own statistics. Not recommended for comparison. |
