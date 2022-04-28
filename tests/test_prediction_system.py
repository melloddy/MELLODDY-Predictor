import os
import pathlib

import pytest

from model_manipulation_software import PredictionSystem

TEST_FILE_DIR = os.path.dirname(__file__)


# TODO parametrize with cls and hybrid models
@pytest.mark.slow
def test_prediction_system_cls():
    models_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/models")
    encryption_key = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_key.json")
    preparation_parameter = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_parameters.json")
    smiles_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples.csv")

    predictor = PredictionSystem(
        model_folder=models_path,
        encryption_key=encryption_key,
        preparation_parameters=preparation_parameter,
    )

    cls_pred, reg_pred, _ = predictor.predict("example_cls_model", smiles_path)

    print(cls_pred.head(10))

    # This is definitely not a great test but it is better than nothing
    # cls model:
    assert cls_pred.iloc[0][1] == pytest.approx(0.528646)
    assert reg_pred.shape[1] == 1


# TODO: test df_failed, hybrid
