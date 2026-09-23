from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent

setup(
    name="wp21_ml_utils",
    version="0.2.0",
    description="Custom TensorFlow/HGQ2 layers and models for the Global Trigger",
    long_description=(HERE / "README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    url="https://github.com/brightneedle/wp21_ml_utils.git",
    author="Noah Clarke Hall",
    author_email="noah.clarkehall@cern.ch",
    license="LGPL-3.0-only",
    license_files=["LICENSE"],
    packages=["wp21_ml_utils"],
    python_requires=">=3.10",
    install_requires=["pyyaml", "tensorflow>=2.16", "HGQ2>=0.1.8"],
    extras_require={
        "dev": [
            "pytest",
            "matplotlib",
            "pre-commit",
            "scipy",
            "twine",
            "hls4ml>=1.2",
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
