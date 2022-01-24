from setuptools import setup

setup(
    name="geci_plots",
    version="0.4.1",
    packages=["geci_plots"],
    package_data={'': ['geci_styles.mplstyle']},
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
    ],
    author="Ciencia de Datos • GECI",
)
