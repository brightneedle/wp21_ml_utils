from setuptools import setup

setup(
    name="wp21_ml_utils",
    version="0.1.0",
    description="Custom TensorFlow/HGQ2 layers and models for the Global Trigger",
    url="https://github.com/brightneedle/wp21_ml_utils.git",
    author="Noah Clarke Hall",
    author_email="noah.clarkehall@cern.ch",
    license="BSD 2-clause",
    packages=["wp21_ml_utils"],
    install_requires=[
        "tensorflow[and-cuda]==2.18",
        "HGQ2==0.1.8",
        "monotonic-nn",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
