"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import find_packages, setup

setup(
    name="prediction_utils",
    version="0.6.0",
    description="Extractors and Models for predictions on OMOP CDM datasets",
    url="https://github.com/som-shahlab/prediction_utils",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas>=1.0.0",
        "matplotlib",
        "torch",
        "pyyaml",
        "pyarrow",
        "sqlalchemy",
        "dask>=2.14.0",
        "scipy",
        "sklearn",
        "configargparse",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "pandas-gbq",
        "pytest",
        "tqdm",
        "joblib",
        "seaborn"
    ],
)
