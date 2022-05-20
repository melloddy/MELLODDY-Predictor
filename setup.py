import pathlib

from setuptools import find_packages
from setuptools import setup

about: dict = dict()
version_path = pathlib.Path(__file__).parent / "model_manipulation_software" / "__version__.py"
with version_path.open("r", encoding="utf-8") as fp:
    exec(fp.read(), about)

setup(
    name="model_manipulation_software",
    python_requires=">=3.8.0",
    version=about["__version__"],
    packages=find_packages(),
    install_requires=[
        "numpy==1.*",
        "pandas==1.*",
        "jsonschema",
        "pandera",
        "scipy==1.*",
        "rdkit-pypi==2021.03.5",
        "dask",
        "pandas==1.*",
    ],
    extras_require={
        "gitlab": [
            "melloddy_tuner@git+ssh://git@git.infra.melloddy.eu/wp1/data_prep.git@develop",
            "sparsechem@git+ssh://git@git.infra.melloddy.eu/wp2/sparsechem.git@master",
        ],
    },
)
