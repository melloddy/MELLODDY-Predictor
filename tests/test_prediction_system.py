import os
import pathlib

import melloddy_tuner.utils.helper  # type: ignore
import pandas as pd
import pytest

from melloddy_predictor import Model
from melloddy_predictor import PreparedData

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
    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
    )

    failing_smiles = prepared_data.failed_compounds

    model = Model(MODELS_PATH / model)

    cls_pred, reg_pred = model.predict(prepared_data)

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

    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
    )

    model = Model(MODELS_PATH / "example_cls_model")

    cls_pred, _ = model.predict(prepared_data, classification_tasks=[24])

    assert cls_pred.iloc[0][1] == 0
    assert cls_pred.iloc[0][24] == pytest.approx(0.34870466589927673)


def test_prediction_on_multiple_models():
    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
    )

    model = Model(MODELS_PATH / "example_cls_model")
    model2 = Model(MODELS_PATH / "example_hyb_model")

    cls_pred, _ = model.predict(prepared_data)
    _, reg_pred = model2.predict(prepared_data)

    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert reg_pred.iloc[0][0] == pytest.approx(5.949892)


def test_failing_smiles():
    smiles_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples_failing.csv")

    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(smiles_path))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
    )

    failing_smiles = prepared_data.failed_compounds

    model = Model(MODELS_PATH / "example_cls_model")

    cls_pred, reg_pred = model.predict(prepared_data)

    # This is definitely not a great test but it is better than nothing
    # cls model:
    assert failing_smiles.shape == (1, 2)
    assert failing_smiles["input_compound_id"][0] == 1376019
    assert "number of non-H atoms 155 exceeds limit of 100 for smiles" in str(failing_smiles["error_message"][0])
    assert cls_pred.shape == (99, 2952)
    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert reg_pred.shape == (99, 0)


def test_load_on_demand():
    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
    )

    # test init True
    model = Model(MODELS_PATH / "example_cls_model", load_on_demand=True)
    assert not hasattr(model, "_model")
    cls_pred, _ = model.predict(prepared_data)
    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert not hasattr(model, "_model")

    # test set False post init
    model.load_on_demand = False
    assert model._model
    cls_pred, _ = model.predict(prepared_data)
    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert model._model

    # test init False
    model = Model(MODELS_PATH / "example_cls_model", load_on_demand=False)
    assert model._model
    cls_pred, _ = model.predict(prepared_data)
    assert cls_pred.iloc[0][0] == pytest.approx(0.528646)
    assert model._model
