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
NUM_CPU = 1


@pytest.mark.parametrize(
    ["model", "is_reg", "is_cls"],
    [
        ("example_cls_model", False, True),
        ("example_clsaux_model", False, True),
        ("example_reg_model", True, False),
        ("example_hyb_model", True, True),
    ],
)
@pytest.mark.slow
def test_prediction_system(model, is_reg, is_cls):
    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
        num_cpu=NUM_CPU,
    )

    failing_smiles = prepared_data.failed_compounds

    model = Model(MODELS_PATH / model)

    cls_pred, reg_pred = model.predict(prepared_data)

    if is_reg:
        assert model._regr_output_size > 0
        assert all(pd.notna(reg_pred))  # all predictions are not NaN
        assert not reg_pred.eq(0).any().any()  # no zero values
        # test that we apply the inverse normalization to the regression predictions
        assert (
            reg_pred.iloc[0][0] > 1
        )  # first value become > 1 when the inverse_normalization is applied (observed with the example models)

    if is_cls:
        assert model._class_output_size > 0
        assert all(pd.notna(cls_pred))
        assert not cls_pred.eq(0).any().any()

    assert failing_smiles.empty  # no failing data

    assert cls_pred.shape == (df.shape[0], model._class_output_size)
    assert reg_pred.shape == (df.shape[0], model._regr_output_size)

    assert cls_pred.columns.is_unique
    assert cls_pred.index.is_unique  # input_compound_id

    assert reg_pred.columns.is_unique
    assert reg_pred.index.is_unique


def test_prediction_on_subset_of_tasks():

    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
        num_cpu=NUM_CPU,
    )

    model = Model(MODELS_PATH / "example_cls_model")

    index = 24
    cls_pred, _ = model.predict(prepared_data, classification_tasks=[index])

    assert cls_pred.drop(cls_pred.columns[index], axis=1).eq(0).all().all()  # all predictions are 0 except for index
    assert not cls_pred.iloc[:, index].eq(0).any()  # no zero values


def test_prediction_on_multiple_models():
    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
        num_cpu=NUM_CPU,
    )

    model = Model(MODELS_PATH / "example_cls_model")
    model2 = Model(MODELS_PATH / "example_hyb_model")

    cls_pred, _ = model.predict(prepared_data)
    _, reg_pred = model2.predict(prepared_data)

    assert model2._regr_output_size > 0
    assert all(pd.notna(reg_pred))  # all predictions are not NaN
    assert not reg_pred.eq(0).any().any()  # no zero values

    assert model._class_output_size > 0
    assert all(pd.notna(cls_pred))
    assert not cls_pred.eq(0).any().any()

    assert cls_pred.shape == (df.shape[0], model._class_output_size)
    assert reg_pred.shape == (df.shape[0], model2._regr_output_size)


def test_failing_smiles():
    smiles_path = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/data/T2_100samples_failing.csv")

    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(smiles_path))
    error_message = "number of non-H atoms 155 exceeds limit of 100"

    with pytest.warns() as warnings:
        prepared_data = PreparedData(
            encryption_key=ENCRYPTION_KEY,
            preparation_parameters=PREPARATION_PARAMETER,
            smiles=df,
            num_cpu=NUM_CPU,
        )

    assert any([error_message in str(w.message) for w in warnings])

    failing_smiles = prepared_data.failed_compounds

    model = Model(MODELS_PATH / "example_cls_model")

    cls_pred, reg_pred = model.predict(prepared_data)

    assert failing_smiles.shape == (1, 2)
    assert failing_smiles["input_compound_id"][0] == 1376019
    assert error_message in str(failing_smiles["error_message"][0])
    assert cls_pred.shape == (df.shape[0] - 1, model._class_output_size)

    assert reg_pred.shape == (df.shape[0] - 1, model._regr_output_size)


def test_load_on_demand():
    df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(SMILES_PATH))

    prepared_data = PreparedData(
        encryption_key=ENCRYPTION_KEY,
        preparation_parameters=PREPARATION_PARAMETER,
        smiles=df,
        num_cpu=NUM_CPU,
    )

    # test init True
    model = Model(MODELS_PATH / "example_cls_model", load_on_demand=True)
    assert not hasattr(model, "_model")
    cls_pred, _ = model.predict(prepared_data)
    assert cls_pred.shape == (df.shape[0], model._class_output_size)
    assert not hasattr(model, "_model")

    # test set False post init
    model.load_on_demand = False
    assert model._model
    cls_pred, _ = model.predict(prepared_data)
    assert cls_pred.shape == (df.shape[0], model._class_output_size)
    assert model._model

    # test init False
    model = Model(MODELS_PATH / "example_cls_model", load_on_demand=False)
    assert model._model
    cls_pred, _ = model.predict(prepared_data)
    assert cls_pred.shape == (df.shape[0], model._class_output_size)
    assert model._model
