import os
import pathlib

import pandas as pd
import pytest

from model_manipulation_software import PredictionSystem

TEST_FILE_DIR = os.path.dirname(__file__)
MODELS_PATH = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/models")
ENCRYPTION_KEY = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_key.json")
PREPARATION_PARAMETER = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_parameters.json")
SMILES_PATH = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples.csv")


@pytest.mark.parametrize(
    ["model", "expected_shapes", "expected_values"],
    [
        (
            "example_cls_model",
            [(100, 2952), (100, 0)],  # TODO: directly use class_output_size from hps
            [pytest.approx(0.528646), None],
        ),
        ("example_clsaux_model", [(100, 3466), (100, 0)], [pytest.approx(0.639690), None]),
        ("example_reg_model", [(100, 0), (100, 1587)], [None, pytest.approx(5.645558)]),
        (
            "example_hyb_model",
            [(100, 2952), (100, 1587)],
            [pytest.approx(0.730933), pytest.approx(5.949892)],
        ),
    ],
)
@pytest.mark.slow
def test_prediction_system(model, expected_values, expected_shapes):
    predictor = PredictionSystem(
        model_folder=MODELS_PATH,
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
    )

    cls_pred, reg_pred, failing_smiles = predictor.predict(model, SMILES_PATH)

    # This is definitely not a great test but it is better than nothing
    assert failing_smiles.empty  # no failing data
    assert cls_pred.shape == expected_shapes[0]
    assert reg_pred.shape == expected_shapes[1]

    assert cls_pred.columns.is_unique
    assert cls_pred.index.is_unique  # input_compound_id

    assert reg_pred.columns.is_unique
    assert reg_pred.index.is_unique

    if expected_values[0] is not None:
        assert cls_pred.iloc[0][0] == expected_values[0]
    if expected_values[1] is not None:
        assert reg_pred.iloc[0][0] == expected_values[1]


def test_prediction_on_subset_of_tasks():
    predictor = PredictionSystem(
        model_folder=MODELS_PATH,
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
    )

    cls_pred, _, _ = predictor.predict("example_cls_model", SMILES_PATH, [24])

    assert cls_pred.iloc[0][1] == 0
    assert cls_pred.iloc[0][24] == pytest.approx(0.34870466589927673)


def test_failing_smiles(tmp_path):
    models_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/models")
    encryption_key = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_key.json")
    preparation_parameter = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_parameters.json")
    smiles_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples_failing.csv")

    smiles = pd.read_csv(smiles_path)
    failing_smiles_path = tmp_path / "failing_smiles.csv"
    smiles.to_csv(failing_smiles_path, columns=["input_compound_id", "smiles"])

    predictor = PredictionSystem(
        model_folder=models_path,
        encryption_key=encryption_key,
        preparation_parameters=preparation_parameter,
    )

    cls_pred, reg_pred, failing_smiles = predictor.predict("example_cls_model", failing_smiles_path)

    # This is definitely not a great test but it is better than nothing
    # cls model:
    assert failing_smiles.shape == (1, 2)
    assert failing_smiles["input_compound_id"][0] == 1376019
    assert "number of non-H atoms 155 exceeds limit of 100 for smiles" in str(failing_smiles["error_message"][0])
    assert cls_pred.shape == (99, 2952)
    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert reg_pred.shape == (99, 0)
