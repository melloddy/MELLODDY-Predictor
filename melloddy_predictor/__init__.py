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

"""
This module is used to generate predictions from a melloddy model.

To use it, you have to create a `PreparedData` object and a `Model` object, and then call the `Model.predict` method.

Example:
    You can see an example in `examples/example.py` and run it with:

        $ python example.py

    Or you can use the Jupyter Notebook provided in the `examples/` folder.

"""

from melloddy_predictor.model import Model
from melloddy_predictor.prepared_data import PreparedData

__all__ = ["PreparedData", "Model"]

__pdoc__ = {
    "model": False,
    "prepared_data": False,
    "predictions": False,
}
