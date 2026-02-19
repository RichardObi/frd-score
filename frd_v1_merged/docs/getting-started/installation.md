# Installation

## From PyPI

```bash
pip install frd-score
```

### Pyradiomics dependency

`frd-score` depends on [pyradiomics](https://github.com/AIM-Harvard/pyradiomics) for radiomic feature extraction. The PyPI release of pyradiomics is **broken for Python ≥ 3.10** ([#903](https://github.com/AIM-Harvard/pyradiomics/issues/903)), so it must be installed from GitHub:

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
```

!!! warning "Install pyradiomics first"
    Run the pyradiomics install command **before** or **after** `pip install frd-score`. If pyradiomics is not installed, importing `frd_score` will raise a helpful `ImportError` with the install command.

## From source (development)

```bash
git clone https://github.com/RichardObi/frd-score.git
cd frd-score
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
pip install -e ".[dev]"
```

This installs the development extras: `pytest`, `nox`, `black`, `isort`, `flake8`, and `nibabel`.

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pyradiomics | from GitHub master |
| numpy | any |
| scipy | ≥ 1.10.0 |
| Pillow | ≥ 10.3.0 |
| SimpleITK | ≥ 2.3.1 |
| opencv-contrib-python-headless | ≥ 4.8.1.78 |

### Optional

| Package | Purpose |
|---|---|
| `matplotlib` | Interpretability visualisations |
| `scikit-learn` | t-SNE in interpretability analysis |
| `nibabel` | NIfTI test fixtures (dev only) |

## Platform notes

### Windows

Building pyradiomics from source requires a C compiler and CMake. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **"Desktop development with C++"** workload, then retry:

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
```

### macOS (Apple Silicon)

No special steps needed. The standard install works on both Intel and Apple Silicon Macs.

## Verifying the installation

```python
import frd_score
print(frd_score.__version__)

from frd_score.frd import get_feature_extractor
extractor = get_feature_extractor(frd_version="v1", image_dim=2)
print("Extractor ready:", type(extractor).__name__)
```
