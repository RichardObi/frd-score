# Contributing

Thank you for your interest in contributing to FRD-Score!

## Development setup

```bash
git clone https://github.com/RichardObi/frd-score.git
cd frd-score
python -m venv frd_env && source frd_env/bin/activate
pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
pip install -e ".[dev]"
```

## Running tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test class
python -m pytest tests/test_frd.py::TestFRDv1_2D -v

# Fast unit tests only (no radiomics extraction)
python -m pytest tests/test_frd.py -k "TestExcludeFeatures or TestMeansOnly or TestMatchSampleCount" -v

# With short traceback
python -m pytest tests/ --tb=short -q
```

!!! tip "Test runtime"
    Integration tests that run full radiomics extraction can take several minutes each. Use `-k` to select specific test classes when iterating quickly.

!!! info "Medical image fixtures"
    `TestMedicalImages2D` uses real downscaled medical images in `tests/data/medical_2d/{d1,d2}/`. These 128×128 grayscale PNGs (~155 KB total) are committed to the repo and provide more realistic test coverage than synthetic noise images.

## Code style

The project uses [Black](https://github.com/psf/black) (line length 88) and [isort](https://pycqa.github.io/isort/) for formatting:

```bash
black src/ tests/
isort src/ tests/
```

Lint with [flake8](https://flake8.pycqa.org/):

```bash
flake8 src/ tests/
```

## Project structure

```
frd-score/
├── src/frd_score/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── frd.py               # Core implementation (~2200 lines)
│   ├── py.typed             # PEP 561 type marker
│   └── configs/
│       ├── extraction_2d.yaml
│       └── extraction_3d.yaml
├── tests/
│   ├── test_frd.py          # Test suite (87+ tests)
│   └── data/medical_2d/     # Real medical image fixtures (128×128 grayscale)
├── docs/                    # MkDocs documentation source
├── .github/workflows/
│   ├── ci.yml               # CI: tests on push/PR
│   ├── docs.yml             # Build & deploy docs to GitHub Pages
│   └── publish.yml          # Publish to PyPI on release
├── recipe/                  # conda-forge recipe
├── setup.py                 # Package configuration
├── pyproject.toml           # Build system & tool config
├── mkdocs.yml               # Documentation config
├── requirements.in          # Direct dependencies
└── requirements.txt         # Pinned dependencies
```

## Building documentation locally

```bash
pip install mkdocs-material
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Changes to `docs/` are live-reloaded.

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with tests.
3. Run `python -m pytest tests/ -v` and ensure all tests pass.
4. Run `black` and `isort` on your changes.
5. Open a pull request with a clear description of the change.

## Reporting issues

Please open an issue at [github.com/RichardObi/frd-score/issues](https://github.com/RichardObi/frd-score/issues) and include:

- Python version (`python --version`)
- OS and architecture
- Package versions: `pip list | grep -E "frd-score|pyradiomics|numpy|scipy"`
- Full error traceback
- Minimal reproduction script if possible
