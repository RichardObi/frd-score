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
python -m pytest tests/test_frd.py::TestComputeFRD -v

# With short traceback
python -m pytest tests/ --tb=short -q
```

## Code style

The project uses [Black](https://github.com/psf/black) (line length 88) and [isort](https://pycqa.github.io/isort/) for formatting:

```bash
black src/ tests/
isort src/ tests/
```

Lint with flake8:

```bash
flake8 src/ tests/
```

Or run everything via [nox](https://nox.thea.codes/):

```bash
nox -s lint
nox -s tests
```

## Project structure

```
frd-score/
├── src/frd_score/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entry point
│   ├── frd.py               # Core implementation
│   └── configs/
│       ├── extraction_2d.yaml
│       └── extraction_3d.yaml
├── tests/
│   └── test_frd.py          # Test suite (75+ tests)
├── docs/                    # MkDocs documentation
├── recipe/                  # conda-forge recipe
├── setup.py                 # Package configuration
├── pyproject.toml           # Build system & tool config
├── requirements.in          # Direct dependencies
├── requirements.txt         # Pinned dependencies
└── mkdocs.yml               # Documentation config
```

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes with tests.
3. Run `python -m pytest tests/ -v` and ensure all tests pass.
4. Run `black` and `isort` on your changes.
5. Open a pull request with a clear description.

## Reporting issues

Please include:

- Python version (`python --version`)
- OS and architecture
- `pip list | grep -E "frd-score|pyradiomics|numpy|scipy"`
- Full error traceback
- Minimal reproduction script if possible
