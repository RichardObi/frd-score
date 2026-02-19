# CLI Reference

## Main command

```
python -m frd_score <path1> <path2> [OPTIONS]
```

`path1` and `path2` can be:

- Directories containing images (PNG, JPG, TIFF, BMP, NIfTI)
- Paths to `.npz` statistics files (created with `--save_stats`)
- Mixed: one `.npz` and one directory

### Extraction & normalisation flags

These flags are shared between the main command and the `ood` subcommand.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--frd_version` | `v0` \| `v1` | `v1` | FRD version to use |
| `-m`, `--paths_masks` | path path | — | Mask directories for localised extraction |
| `-f`, `--feature_groups` | str... | version default | PyRadiomics feature classes |
| `-I`, `--image_types` | str... | version default | Image filter types: `Original`, `LoG`, `Wavelet` |
| `-r`, `--resize_size` | int [int] | — | Resize images to N×N or W H |
| `-R`, `--norm_range` | float float | version default | Normalisation value range |
| `-T`, `--norm_type` | `minmax` \| `zscore` | version default | Normalisation strategy |
| `--norm_ref` | `joint` \| `d1` \| `independent` | version default | Normalisation reference distribution |
| `-v`, `--verbose` | flag | off | Enable detailed logging |
| `-w`, `--num_workers` | int | auto | CPU workers for multiprocessing |
| `--log_sigma` | float... | `2.0 3.0 4.0 5.0` | LoG sigma values |
| `--bin_width` | int | `5` | PyRadiomics bin width |
| `--normalize_scale` | float | `100` | PyRadiomics normalise scale |
| `--voxel_array_shift` | float | `300` | PyRadiomics voxel array shift |
| `--config_path` | path | — | Custom PyRadiomics YAML config (overrides above) |
| `--exclude_features` | str... | — | Post-extraction exclusion: `textural`, `wavelet`, `firstorder` |
| `--match_sample_count` | flag | off | Subsample larger dataset to match smaller |

### FRD-specific flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `-s`, `--save_stats` | flag | off | Save statistics to `.npz` instead of computing FRD |
| `-F`, `--save_features` | flag | off | Save features to CSV |
| `--use_paper_log` | flag | off | Use paper Eq. 3 log transform |
| `--means_only` | flag | off | Mean-only Fréchet distance (skip covariance) |
| `--interpret` | flag | off | Run interpretability analysis |
| `--interpret_dir` | path | `outputs/interpretability_visualizations` | Output dir for interpretation plots |

## OOD subcommand

```
python -m frd_score ood <reference_path> <test_path> [OPTIONS]
```

### OOD-specific flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--detection_type` | `image` \| `dataset` | `image` | Per-image or dataset-level OOD |
| `--val_frac` | float | `0.1` | Fraction of reference held out for threshold |
| `--use_val_set` | flag | off | Enable hold-out validation split |
| `--id_dist_assumption` | `gaussian` \| `t` \| `counting` | `gaussian` | Statistical model for in-distribution scores |
| `--output_dir` | path | `outputs/ood_predictions` | Output directory for OOD CSV |
| `--seed` | int | — | Random seed for reproducibility |

All shared extraction flags are also available.

## Examples

```bash
# Basic FRDv1
python -m frd_score data/real data/synthetic

# FRDv0 with masks and verbose
python -m frd_score data/real data/synthetic \
    --frd_version v0 \
    -m data/masks_real data/masks_synthetic \
    -v

# Custom feature extraction
python -m frd_score data/real data/synthetic \
    -f firstorder glcm \
    -I Original LoG \
    --log_sigma 1.0 2.0 3.0 \
    --bin_width 10

# Save and reuse statistics
python -m frd_score --save_stats data/reference ref_stats.npz
python -m frd_score ref_stats.npz data/generated

# OOD detection
python -m frd_score ood data/in_domain data/test_images \
    --seed 42 --use_val_set --val_frac 0.2

# Interpretability
python -m frd_score data/real data/synthetic \
    --interpret --interpret_dir results/interp/
```
