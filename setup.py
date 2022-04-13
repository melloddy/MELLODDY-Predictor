import pathlib

from setuptools import find_packages
from setuptools import setup

about: dict = dict()
version_path = pathlib.Path(__file__).parent / "model_manipulation_software" / "__version__.py"
with version_path.open("r", encoding="utf-8") as fp:
    exec(fp.read(), about)

setup(
    name="model_manipulation_software",
    version=about["__version__"],
    packages=find_packages(),
    install_requires=[
        "numpy==1.22.3",
        "pandas==1.4.1",
        "jsonschema==4.4.0",
        "pandera==0.9.0",
        "scipy==1.8.0",
        "rdkit-pypi==2021.9.5.1",
        "dask==2022.3.0",
        "torch==1.8.1",
        "pandas==1.4.1",
    ],
    extras_require={
        "gitlab": [
            "melloddy_tuner@git+ssh://git@git.infra.melloddy.eu/wp1/data_prep.git@develop",
            "sparsechem@git+ssh://git@git.infra.melloddy.eu/wp2/sparsechem.git@master",
        ],
    },
)
