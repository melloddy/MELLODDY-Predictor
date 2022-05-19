"""
This module is used to generate predictions from a melloddy model.

To use it, you have to create a `PredictionSystem` object, and then call the `PredictionSystem.predict` method.

Example:
    You can see an example in `examples/example.py` and run it with:

        $ python example.py

    Or you can use the Jupyter Notebook provided in the `examples/` folder.

"""

from model_manipulation_software.model import Model
from model_manipulation_software.prediction_system import PredictionSystem

__all__ = ["PredictionSystem", "Model"]

__pdoc__ = {
    "model": False,
    "prediction_system": False,
    "predictions": False,
}
