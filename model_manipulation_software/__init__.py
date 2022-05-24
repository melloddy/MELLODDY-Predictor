"""
This module is used to generate predictions from a melloddy model.

To use it, you have to create a `PreparedData` object and a `Model` object, and then call the `model.predict` method.

Example:
    You can see an example in `examples/example.py` and run it with:

        $ python example.py

    Or you can use the Jupyter Notebook provided in the `examples/` folder.

"""

from model_manipulation_software.model import Model
from model_manipulation_software.prepared_data import PreparedData

__all__ = ["PreparedData", "Model"]

__pdoc__ = {
    "model": False,
    "prepared_data": False,
    "predictions": False,
}
