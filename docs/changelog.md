# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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

## [Unreleased]

### Added

- **Medical image test fixtures** — 20 downscaled (128×128) grayscale PNGs from diverse modalities in `tests/data/medical_2d/{d1,d2}/`, providing realistic end-to-end test coverage.
- **`TestMedicalImages2D` test class** — 10 new tests exercising `compute_frd()` on real medical images: v0/v1 computation, same-distribution sanity check, `norm_ref` variants, `exclude_features`, `resize_size`, `means_only`, and `match_sample_count`.

## [1.1.0] — 2026-02-19

### Added

- **FRDv1 support** — extended feature extraction with ~464 features (Original + LoG + Wavelet), z-score normalisation, and log-transformed Fréchet distance. v1 is now the default.
- **Unified API** — `compute_frd()` supports both v0 and v1 via the `frd_version` parameter.
- **OOD detection** — `detect_ood()` function and `ood` CLI subcommand for image-level and dataset-level out-of-distribution detection.
- **Interpretability** — `interpret_frd()` with t-SNE visualisations and per-feature difference rankings.
- **Normalisation reference modes** — `norm_ref` parameter with `"joint"`, `"d1"`, and `"independent"` options.
- **Paper log transform** — `use_paper_log` flag for Eq. 3 consistency.
- **PyRadiomics extraction controls** — `bin_width`, `normalize_scale`, `voxel_array_shift`, `log_sigma`, `config_path`, `settings_dict`.
- **Feature exclusion** — `exclude_features` for post-extraction ablation (`"textural"`, `"wavelet"`, `"firstorder"`, `"shape"`). The `"shape"` option removes `shape_*` and `shape2D_*` features, which are often constant when no mask is used.
- **Sample matching** — `match_sample_count` to equalise dataset sizes.
- **Means-only mode** — `means_only` for small datasets.
- **MkDocs documentation** — hosted on GitHub Pages.
- **GitHub Actions CI** — automated testing and docs deployment.
- **`--version` flag** — print the installed frd-score version and exit.
- **Pyradiomics compatibility** — import guard with helpful error message for broken PyPI release ([#903](https://github.com/AIM-Harvard/pyradiomics/issues/903)).

### Changed

- **Python requirement** bumped from ≥3.8 to ≥3.10 (pyradiomics compatibility).
- **pyradiomics dependency** moved to documented install step (PyPI does not support git URLs in `install_requires`).
- Default FRD version changed from v0 to v1.
- `pyproject.toml` now declares build system.

### Fixed

- Pyradiomics import failure on Python ≥3.10 handled with clear error message.

## [1.0.0] — 2024

### Added

- Initial FRDv0 implementation.
- CLI via `python -m frd_score`.
- Min-max normalisation with FID-compatible range.
- Save/load `.npz` statistics.
- Multi-worker feature extraction.
- Mask support for localised feature extraction.
