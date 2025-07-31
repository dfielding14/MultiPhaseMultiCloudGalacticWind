"""
Setup script for multiphasegalacticwind package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multiphasegalacticwind",
    version="0.1.0",
    author="Drummond B. Fielding",
    author_email="dfielding@flatironinstitute.org",
    description="Multiphase galactic wind model for MCMC fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dfielding14/MultiPhaseMultiCloudGalacticWind",
    packages=find_packages(),
    package_data={
        "multiphasegalacticwind": ["data/*.npz"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.3",
    ],
    extras_require={
        "plotting": ["cmasher>=1.6"],
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
)