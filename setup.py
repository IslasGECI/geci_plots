
from setuptools import setup, find_packages

setup(
    name="geci_plots",
    version="0.1.0",
    packages=["geci_plots"],
    install_requires = [
        "numpy",
        "pandas",
        "matplotlib",
    ],
    author="Fernando Alvarez",
    python_requires='>=3.6'
)
