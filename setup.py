# Copyright 2022 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

from setuptools import find_packages
from setuptools import setup

about: dict = dict()
version_path = pathlib.Path(__file__).parent / "melloddy_predictor" / "__version__.py"
with version_path.open("r", encoding="utf-8") as fp:
    exec(fp.read(), about)

setup(
    name="melloddy_predictor",
    python_requires=">=3.8.0,<3.10.0",  # rdkit-pypi==2021.03.5 incompatible with 3.10.0 (Issue #4)
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
        "github": [
            "melloddy_tuner@git+https://git@github.com/MELLODDY/MELLODDY-TUNER.git@master",
            "sparsechem@git+https://git@github.com/MELLODDY/SparseChem.git@master",
            "protobuf==3.20.*",
        ],
        "doc": [
            "pdoc3==0.10.*",
        ],
    },
)
