import os
import pathlib

import pytest

from model_manipulation_software import PredictionSystem

TEST_FILE_DIR = os.path.dirname(__file__)
MODELS_PATH = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/models")
ENCRYPTION_KEY = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_key.json")
PREPARATION_PARAMETER = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_parameters.json")
SMILES_PATH = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples.csv")


# TODO parametrize with cls and hybrid models
@pytest.mark.slow
def test_prediction_system_cls():
    predictor = PredictionSystem(
        model_folder=MODELS_PATH,
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
    )

    cls_pred, reg_pred, _ = predictor.predict("example_cls_model", SMILES_PATH)

    # This is definitely not a great test but it is better than nothing
    # cls model:
    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert cls_pred.iloc[99][0] == pytest.approx(0.5745349526405334)
    assert reg_pred.shape[1] == 0


def test_prediction_on_subset_of_tasks():
    predictor = PredictionSystem(
        model_folder=MODELS_PATH,
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
    )

    cls_pred, _, _ = predictor.predict("example_cls_model", SMILES_PATH, [24])

    assert cls_pred.iloc[0][1] == 0
    assert cls_pred.iloc[0][24] == pytest.approx(0.34870466589927673)


# TODO: test df_failed, hybrid
