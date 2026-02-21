# coding: utf-8
import os

import setuptools



def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), "r") as f:
        return f.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.0.1"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setuptools.setup(
        name="frd-score",
        version=get_version(os.path.join("src", "frd_score", "__init__.py")),
        author="Richard Osuala, Nick Konz, Preeti Verma",
        description="Fréchet Radiomics Distance (FRD) — a metric for comparing medical image distributions",
        long_description=read("README.md"),
        license="Apache-2.0",
        long_description_content_type="text/markdown",
        url="https://github.com/RichardObi/frd-score",
        download_url=f"https://github.com/RichardObi/frd-score/archive/refs/tags/v{get_version(os.path.join('src', 'frd_score', '__init__.py'))}.tar.gz",
        project_urls={
            "Bug Tracker": "https://github.com/RichardObi/frd-score/issues",
            "Documentation": "https://richardobi.github.io/frd-score/",
            "Project Website": "https://richardobi.github.io/frd/",
            "Changelog": "https://richardobi.github.io/frd-score/changelog/",
            "Source": "https://github.com/RichardObi/frd-score",
        },
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        package_data={
            "frd_score": ["configs/*.yaml", "py.typed"],
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Healthcare Industry",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Topic :: Scientific/Engineering :: Medical Science Apps.",
            "Topic :: Scientific/Engineering :: Image Processing",
        ],
        keywords=['Radiomics', 'Frechet', 'Distance', 'medical', 'imaging', 'radiology', 'generative', 'synthetic', 'evaluation'],
        python_requires=">=3.10",
        entry_points={
            "console_scripts": [
                "frd-score = frd_score.frd:main",
            ],
        },
        install_requires=[
            # NOTE: pyradiomics is intentionally NOT listed here.
            # PyPI does not allow git+https URLs in install_requires,
            # and the PyPI release (<=3.1.0) is broken for Python >=3.10.
            # Users must install pyradiomics separately from GitHub:
            #   pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master
            # See: https://github.com/AIM-Harvard/pyradiomics/issues/903
            "numpy",
            "Pillow>=10.3.0",
            "scipy>=1.10.0",
            "opencv_contrib_python_headless>=4.8.1.78",
            "SimpleITK>=2.3.1",
        ],
        extras_require={
            "dev": [
                "flake8",
                "flake8-bugbear",
                "flake8-isort",
                "black==24.3.0",
                "isort",
                "nox",
                "pytest>=8.1.1",
                "nibabel>=3.2.1",
            ],
            "docs": [
                "mkdocs-material>=9.0",
                "mkdocstrings[python]>=0.25.0",
                "mike>=2.0",
            ],
        },
    )
