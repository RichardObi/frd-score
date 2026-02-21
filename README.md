<!---[![PyPI](https://img.shields.io/pypi/v/frd-score.svg)](https://pypi.org/project/frd-score/)--->
## NEWS 1/9/26: 🎉 the FRD paper has been accepted to [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841526000125) 🎉!

# FRD (Fréchet Radiomic Distance): A Metric Designed for Medical Image Distribution Comparison in the Age of Deep Learning

#### By [Nicholas Konz*](https://nickk124.github.io/), [Richard Osuala*](https://scholar.google.com/citations?user=0KkVRVQAAAAJ&hl=en), (* = equal contribution), [Preeti Verma](https://scholar.google.com/citations?user=6WN41lwAAAAJ&hl=en), [Yuwen Chen](https://scholar.google.com/citations?user=61s49p0AAAAJ&hl=en), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en), [Haoyu Dong](https://haoyudong-97.github.io/), [Yaqian Chen](https://scholar.google.com/citations?user=iegKFuQAAAAJ&hl=en), [Andrew Marshall](https://linkedin.com/in/andrewmarshall26), [Lidia Garrucho](https://github.com/LidiaGarrucho), [Kaisar Kushibar](https://scholar.google.es/citations?user=VeHqMi4AAAAJ&hl=en), [Daniel M. Lang](https://scholar.google.com/citations?user=AV04Hs4AAAAJ&hl=en), [Gene S. Kim](https://vivo.weill.cornell.edu/display/cwid-sgk4001), [Lars J. Grimm](https://scholars.duke.edu/person/lars.grimm), [John M. Lewin](https://medicine.yale.edu/profile/john-lewin/), [James S. Duncan](https://medicine.yale.edu/profile/james-duncan/), [Julia A. Schnabel](https://compai-lab.github.io/), [Oliver Diaz](https://sites.google.com/site/odiazmontesdeoca/home), [Karim Lekadir](https://www.bcn-aim.org/) and [Maciej A. Mazurowski](https://sites.duke.edu/mazurowski/).


[![PyPI](https://img.shields.io/pypi/v/frd-score.svg)](https://pypi.org/project/frd-score/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/frd-score.svg)](https://anaconda.org/conda-forge/frd-score)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://github.com/RichardObi/frd-score/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RichardObi/frd-score/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2412.01496-b31b1b.svg)](https://arxiv.org/abs/2412.01496)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://richardobi.github.io/frd/)

# Fréchet Radiomics Distance (FRD)

**[Project Website](https://richardobi.github.io/frd/)** · **[Paper (Medical Image Analysis)](https://www.sciencedirect.com/science/article/pii/S1361841526000125)** · **[arXiv](https://arxiv.org/abs/2412.01496)** · **[Evaluation Framework](https://github.com/mazurowski-lab/medical-image-similarity-metrics)** · **[Documentation](https://richardobi.github.io/frd-score/)** · **[API](https://richardobi.github.io/frd-score/api/)**


<p align="center">
  <img src="https://raw.githubusercontent.com/RichardObi/frd-score/main/docs/assets/teaser.png" alt="FRD overview" width="75%">
</p>

**FRD** measures the similarity of [radiomic](https://pyradiomics.readthedocs.io/) image features between two datasets by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between Gaussians fitted to the extracted and normalized features. The lower the FRD, the more similar the two datasets.

FRD supports both **2D** (PNG, JPG, TIFF, BMP) and **3D** (NIfTI `.nii.gz`) radiological images.

<p align="center">
  <img src="https://raw.githubusercontent.com/RichardObi/frd-score/main/docs/assets/radiomics_taxonomy.jpg" alt="Radiomics feature taxonomy" width="75%">
</p>

## Why FRD over FID, KID, CMMD, etc.?

FRD uses *standardised radiomic features* rather than pretrained deep features (as in FID, KID, CMMD). We show in our paper that this yields:

1. **Better alignment** with downstream task performance (e.g. segmentation).
2. **Improved stability** and computational efficiency for small-to-moderately-sized datasets.
3. **Improved interpretability**, because radiomic features are clearly defined and widely used in medical imaging.

## FRD Versions

| | FRDv0 (Osuala et al., 2024) | FRDv1 (Konz, Osuala et al., 2026) — **default** |
|---|---|---|
| Features | ~94 (Original only) | ~464 (Original + LoG + Wavelet) |
| Normalization | min-max, joint | z-score, D1-referenced |
| Output | raw Fréchet distance | log-transformed Fréchet distance |
| Feature classes | firstorder, glcm, glrlm, gldm, glszm, ngtdm, shape, shape2D | firstorder, glcm, glrlm, glszm, ngtdm |

## Installation

### From PyPI (recommended)

```bash
pip install frd-score
```

> **Note:** `frd-score` requires [pyradiomics](https://github.com/AIM-Harvard/pyradiomics), which must be installed separately from GitHub because the PyPI release is broken for Python ≥ 3.10 ([#903](https://github.com/AIM-Harvard/pyradiomics/issues/903)):

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
```

### From Conda (conda-forge)

```bash
conda install -c conda-forge frd-score
```

The conda-forge package includes all dependencies (including pyradiomics), so no additional installation steps are needed.

### From source

```bash
git clone https://github.com/RichardObi/frd-score.git
cd frd-score
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.10
- pyradiomics (installed from GitHub, see above)
- numpy, scipy, Pillow, SimpleITK, opencv-contrib-python-headless

<details>
<summary><strong>Windows users</strong></summary>

Building pyradiomics from source requires a C compiler and CMake. Install
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the "Desktop development with C++" workload, then retry the pip install.

</details>

## Quick Start

### CLI

```bash
# Compute FRD between two image folders (default: v1)
python -m frd_score path/to/dataset_A path/to/dataset_B

# Use FRDv0 instead
python -m frd_score path/to/dataset_A path/to/dataset_B --frd_version v0

# With masks
python -m frd_score path/to/dataset_A path/to/dataset_B -m path/to/masks_A path/to/masks_B

# Save precomputed statistics to .npz
python -m frd_score --save_stats path/to/dataset path/to/output.npz

# Re-use .npz file
python -m frd_score path/to/output.npz path/to/dataset_B
```

### Python API

```python
from frd_score import compute_frd

# Basic usage
frd_value = compute_frd(["path/to/dataset_A", "path/to/dataset_B"])

# With masks and options
frd_value = compute_frd(
    ["path/to/dataset_A", "path/to/dataset_B"],
    paths_masks=["path/to/masks_A", "path/to/masks_B"],
    frd_version="v1",
    verbose=True,
)

# From file lists
frd_value = compute_frd([
    ["img1.png", "img2.png", "img3.png"],
    ["img4.png", "img5.png", "img6.png"],
])
```

## CLI Reference

### Main command

```
python -m frd_score path1 path2 [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--frd_version` | `v0` or `v1` | `v1` |
| `-m`, `--paths_masks` | Two mask folder paths | None |
| `-f`, `--feature_groups` | Feature classes to extract (e.g. `firstorder glcm`) | version default |
| `-I`, `--image_types` | Image filter types (`Original`, `LoG`, `Wavelet`) | version default |
| `-r`, `--resize_size` | Resize images to N×N or W×H | None |
| `-R`, `--norm_range` | Normalization range `[min max]` | version default |
| `-T`, `--norm_type` | `minmax` or `zscore` | version default |
| `--norm_ref` | Normalization reference: `joint`, `d1`, `independent` | version default |
| `-v`, `--verbose` | Verbose logging | off |
| `-w`, `--num_workers` | CPU workers for multiprocessing | auto |
| `-s`, `--save_stats` | Save statistics to `.npz` | off |
| `-F`, `--save_features` | Save features to CSV | off |
| `--use_paper_log` | Use paper Eq. 3 log transform: `log(√d²)` | off |
| `--means_only` | Mean-only Fréchet distance (no covariance) | off |
| `--log_sigma` | LoG sigma values | `[2.0, 3.0, 4.0, 5.0]` |
| `--bin_width` | PyRadiomics bin width | `5` |
| `--normalize_scale` | PyRadiomics normalize scale | `100` |
| `--voxel_array_shift` | PyRadiomics voxel array shift | `300` |
| `--config_path` | Custom PyRadiomics YAML config | None |
| `--exclude_features` | Post-extraction exclusion: `textural`, `wavelet`, `firstorder`, `shape` | None |
| `--match_sample_count` | Subsample larger dataset to match smaller | off |
| `--interpret` | Run interpretability analysis | off |
| `--interpret_dir` | Output dir for interpretation plots | `outputs/interpretability_visualizations` |

### OOD subcommand

```bash
# Image-level OOD detection
python -m frd_score ood path/to/reference path/to/test

# Dataset-level nFRD
python -m frd_score ood path/to/reference path/to/test --detection_type dataset
```

| Flag | Description | Default |
|------|-------------|---------|
| `--detection_type` | `image` or `dataset` | `image` |
| `--val_frac` | Fraction of reference held out for threshold | `0.1` |
| `--use_val_set` | Enable hold-out validation split | off |
| `--id_dist_assumption` | `gaussian`, `t`, or `counting` | `gaussian` |
| `--output_dir` | Directory for OOD CSV output | `outputs/ood_predictions` |
| `--seed` | Random seed for reproducibility | None |

All shared extraction flags (`--frd_version`, `-f`, `-I`, `--norm_ref`, etc.) are also available in the `ood` subcommand.

## Python API Reference

### `compute_frd(paths, **kwargs)`

Main entry point. See [API docs](https://richardobi.github.io/frd-score/api/) for the full signature and parameter descriptions.

### `save_frd_stats(paths, **kwargs)`

Compute and save feature statistics to a `.npz` file for later re-use. Accepts the same parameters as `compute_frd()`.

### `interpret_frd(feature_list, feature_names, **kwargs)`

Run interpretability analysis on extracted features. Produces t-SNE plots and per-feature difference rankings. Requires `matplotlib` and `scikit-learn`.

### `detect_ood(feature_list, **kwargs)`

Out-of-distribution detection using normalized radiomics features. Supports per-image scoring (`detection_type="image"`) and dataset-level nFRD (`detection_type="dataset"`).

## Interpretability

<p align="center">
  <img src="https://raw.githubusercontent.com/RichardObi/frd-score/main/docs/assets/radiomic_interp.png" alt="Radiomic interpretability" width="65%">
</p>

FRD enables interpretable comparison of image sets. Use the `--interpret` flag or call `interpret_frd()` to:

- Rank the most-changed radiomic features between two distributions
- Visualise feature distributions via t-SNE
- Identify which images changed the most (for paired datasets)

```bash
python -m frd_score path/to/dataset_A path/to/dataset_B --interpret
```

## Out-of-Domain (OOD) Detection

FRD can detect whether newly acquired medical images come from the same domain as a reference set — useful for flagging potential distribution shifts (e.g. different scanners, protocols).

```bash
# Per-image OOD scores and p-values
python -m frd_score ood path/to/reference path/to/test_images

# Dataset-level OOD score (nFRD)
python -m frd_score ood path/to/reference path/to/test_images --detection_type dataset
```

Results are saved to `outputs/ood_predictions/ood_predictions.csv` with columns: `filename`, `ood_score`, `ood_prediction`, `p_value`.

## Citation

If you use this library in your research, please cite:

```bibtex
@article{konz2026frd,
    title     = {Fr\'{e}chet Radiomic Distance (FRD): A Versatile Metric for
                 Comparing Medical Imaging Datasets},
    author    = {Konz, Nicholas and Osuala, Richard and Verma, Preeti and
                 Chen, Yuwen and Gu, Hanxue and Dong, Haoyu and Chen, Yaqian
                 and Marshall, Andrew and Garrucho, Lidia and Kushibar, Kaisar
                 and Lang, Daniel M. and Kim, Gene S. and Grimm, Lars J. and
                 Lewin, John M. and Duncan, James S. and Schnabel, Julia A. and
                 Diaz, Oliver and Lekadir, Karim and Mazurowski, Maciej A.},
    journal   = {Medical Image Analysis},
    volume    = {110},
    pages     = {103943},
    year      = {2026},
    publisher = {Elsevier},
    doi       = {10.1016/j.media.2026.103943},
    url       = {https://www.sciencedirect.com/science/article/pii/S1361841526000125},
}
```

Earlier FRD work:

```bibtex
@article{osuala2024towards,
    title   = {Towards Learning Contrast Kinetics with Multi-Condition
               Latent Diffusion Models},
    author  = {Osuala, Richard and Lang, Daniel and Verma, Preeti and
               Joshi, Smriti and Tsirikoglou, Apostolia and Skorupko, Grzegorz
               and Kushibar, Kaisar and Garrucho, Lidia and Pinaya, Walter HL
               and Diaz, Oliver and others},
    journal = {arXiv preprint arXiv:2403.13890},
    year    = {2024},
}
```

## Links

- [API Documentation](https://richardobi.github.io/frd-score/api) — overview, benchmarks, datasets, FAQ
- [Change Log](https://richardobi.github.io/frd-score/changelog) — overview, benchmarks, datasets, FAQ
- [Project Website](https://richardobi.github.io/frd/) — overview, benchmarks, datasets, FAQ
- [Journal Article](https://www.sciencedirect.com/science/article/pii/S1361841526000125) — Medical Image Analysis, Vol. 110 (2026)
- [arXiv Preprint](https://arxiv.org/abs/2412.01496)
- [Evaluation Framework](https://github.com/mazurowski-lab/medical-image-similarity-metrics) — scripts for OOD detection, translation evaluation, and metric comparison
- [API Documentation](https://richardobi.github.io/frd-score/) — full docs hosted on GitHub Pages

## Acknowledgements

- [Preeti Verma](https://github.com/preeti-verma8600) — implementation of a script of an early frd version.
- [Nicholas Konz](https://nickk124.github.io/) — FRDv1 and [RaD](https://github.com/mazurowski-lab/RaD) repository
- [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) — radiomic feature extraction backend
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) — Fréchet distance implementation reference

## License

[Apache 2.0](LICENSE)
