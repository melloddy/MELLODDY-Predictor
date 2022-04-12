import os
import pathlib

import pytest

from model_manipulation_software import PredictionSystem

TEST_FILE_DIR = os.path.dirname(__file__)


@pytest.mark.slow
def test_prediction_system():
    models_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/models")
    encryption_key = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_key.json")
    preparation_parameter = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_parameters.json")
    smiles_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples.csv")

    predictor = PredictionSystem(
        model_folder=models_path,
        encryption_key=encryption_key,
        preparation_parameters=preparation_parameter,
    )

    cls_pred, reg_pred = predictor.predict("example_hybrid_model", smiles_path)

    # This is definitely not a great test but it is better than nothing
    assert cls_pred[0][0] == pytest.approx(0.996014)
    assert reg_pred[0][0] == pytest.approx(4.2191267)
