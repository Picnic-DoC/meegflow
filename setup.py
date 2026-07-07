#!/usr/bin/env python3
"""Setup script for MEEGFlow."""

import os
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="meegflow",
    version=os.environ.get("MEEGFLOW_VERSION", "0.0.1"),
    description="A modular, configuration-driven extensible M/EEG preprocessing pipeline using MNE-Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Laouen Belloli",
    url="https://github.com/Picnic-DoC/meegflow",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mne>=1.5.0",
        "mne-bids>=0.18.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "PyYAML>=6.0",
        "rich>=13.0.0",
        "h5py>=3.0.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        # 'local' execution backend (in-process/local Dask cluster).
        "dask": ["dask[distributed]>=2024.1.0"],
        # 'slurm' | 'pbs' | 'sge' | 'lsf' | 'htcondor' execution backends.
        "dask-jobqueue": ["dask[distributed]>=2024.1.0", "dask-jobqueue>=0.8.2"],
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "meegflow=meegflow.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
