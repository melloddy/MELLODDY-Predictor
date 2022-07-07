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
from types import SimpleNamespace
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import sparsechem  # type: ignore
import torch
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse import lil_matrix
from torch.utils.data import DataLoader

from melloddy_predictor.predictions import ClassificationPrediction
from melloddy_predictor.predictions import RegressionPrediction
from melloddy_predictor.prepared_data import PreparedData  # type: ignore


class Model:
    """
    A sparsechem model and its configuration

    Args:
        path (pathlib.Path): the path of the model's folder.
            The directory structure inside this folder should be the following:

                my_model/
                ├─ hyperparameters.json
                ├─ model.pth
                ├─ T8c.csv
                ├─ T8r.csv

            The `hyperparameters` file should contain at least a `conf` key with informations about the model.
            The T8 metadata files should be provided based on the kind of the model:

            * CLS model: `T8c.csv`.
            * REG model: `T8r.csv`.
            * HYB model: `T8c.csv` and `T8r.csv`.

            The model should be compatible with `sparsechem` `0.9.6+`. If it is not, you can convert it with
            [this script](https://git.infra.melloddy.eu/wp2/sparsechem/-/blob/convert_v0.9.5_to_v0.9.6/examples/chembl/convert.py).

        device (str): device on which we should load the model (cpu, cuda:0 .. cuda:x). Defaults to "cpu".
        load_on_demand (bool): If `False`, the model will be loaded upon initialization and stay in memory
            until `Model.unload()` is called or the model object is deleted.
            If set to `True` (default), the model will be loaded into memory only when calling `Model.predict()`,
            and unloaded before returning the results.

    Raises:
        FileNotFoundError: path / "hyperparameters.json" not found
        FileNotFoundError: path / "model.pth" not found
    """  # noqa: E501

    _internal_conf: SimpleNamespace
    _model: Optional[sparsechem.SparseFFN]

    def __init__(self, path: pathlib.Path, device: str = "cpu", load_on_demand: bool = True) -> None:
        self.path = path
        self._conf_path = self.path / "hyperparameters.json"
        self._model_path = self.path / "model.pth"
        self._device = device
        self._dropout = 0
        self._y_cat_columns = None
        self._load_on_demand = load_on_demand

        if not os.path.isfile(self._conf_path):
            raise FileNotFoundError(self._conf_path)
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(self._model_path)

        if not self._load_on_demand:
            self.load()

    @property
    def device(self) -> str:
        """
        cf class docstring
        """
        return self._device

    @device.setter
    def device(self, device: str):
        """
        cf class docstring
        """
        if not self._model:
            self._device = device
        else:
            raise Exception("cannot switch device when model is loaded")

    @property
    def load_on_demand(self) -> bool:
        """
        cf class docstring
        """
        return self._load_on_demand

    @load_on_demand.setter
    def load_on_demand(self, load_on_demand: bool):
        """
        cf class docstring
        """
        self._load_on_demand = load_on_demand
        if not self._load_on_demand:
            self.load()

    @property
    def dropout(self) -> int:
        """
        Used as an argument of `sparsechem.predict_sparse()` in `Model.predict()`
        """
        return self._dropout

    @dropout.setter
    def dropout(self, dropout: int):
        """
        Used as an argument of `sparsechem.predict_sparse()` in `Model.predict()`
        """
        self._dropout = dropout

    @property
    def y_cat_columns(self):
        """
        Used as an argument of `sparsechem.predict_sparse()` in `Model.predict()`
        """
        return self._y_cat_columns

    @y_cat_columns.setter
    def y_cat_columns(self, columns):
        """
        Used as an argument of `sparsechem.predict_sparse()` in `Model.predict()`
        """
        self._y_cat_columns = columns

    @property
    def _conf(self) -> SimpleNamespace:
        """
        The configuration of the model, which contains the "conf" values of the "hyperparameters.json" as well as
            - "model_type"
            - "class_output_size"
            - "regr_output_size"

        Returns:
            SimpleNamespace: Namespace sent by sparsechem
        """
        if not hasattr(self, "_internal_conf") or not self._internal_conf:
            self._internal_conf: SimpleNamespace = sparsechem.load_results(str(self._conf_path), two_heads=True)["conf"]
        return self._internal_conf

    @property
    def _stats(self) -> SimpleNamespace:
        """
        The stats of the model, which contains the "stats" values of the "hyperparameters.json", used for
        `inverse_normalization` in `sparsechem` (only for reg and hyb models)

        Returns:
            dict : stats
        """
        if not hasattr(self, "_internal_stats") or not self._internal_stats:
            self._internal_stats: SimpleNamespace = sparsechem.load_results(str(self._conf_path), two_heads=True)[
                "stats"
            ]
        return self._internal_stats

    @property
    def _class_output_size(self) -> str:
        return self._conf.class_output_size

    @property
    def _regr_output_size(self) -> str:
        return self._conf.regr_output_size

    @property
    def _fold_inputs(self) -> str:
        return self._conf.fold_inputs

    @property
    def _input_transform(self) -> str:
        return self._conf.input_transform

    @property
    def _reg_metadata(self) -> pd.DataFrame:
        return self._load_metadata("T8r.csv")

    @property
    def _cls_metadata(self) -> pd.DataFrame:
        return self._load_metadata("T8c.csv")

    def _load_metadata(self, filename) -> pd.DataFrame:
        metadata_file = self.path / filename
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(metadata_file)
        data = pd.read_csv(metadata_file)
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def load(self) -> None:
        """Loads the model on the specified device"""
        if not hasattr(self, "_model") or not self._model:
            self._model = sparsechem.SparseFFN(self._conf).to(self._device)
            state = torch.load(self._model_path, map_location=torch.device(self._device))
            self._model.load_state_dict(state)

    def unload(self) -> None:
        """Remove the model from the device"""
        del self._model

    def predict(
        self,
        prepared_data: PreparedData,
        classification_tasks: Optional[List[int]] = None,
        regression_tasks: Optional[List[int]] = None,
        batch_size: int = 4000,
        num_workers: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predict on the test data (Smiles) using the model.

        Args:
            prepared_data (PreparedData): The data prepared by melloddy_tuner.
            classification_tasks: A list of tasks indexes (`cont_classification_task_id` from the `metadata file`) for
                which you want to predict. If not set it will predict on all classification tasks. If you don't want
                to predict on any tasks you can send an empty list.
            regression_tasks: A list of tasks indexes (`cont_regression_task_id` from the `metadata file`) for which
                you want to predict. If not set it will predict on all regression tasks. If you don't want to predict
                on any tasks you can sent an empty list.
            batch_size: How many data samples should be loaded per batch
                (see `torch.utils.data.DataLoader` for more details).
            num_workers: How many subprocess we should use for data loading
                (see `torch.utils.data.DataLoader` for more details).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: `cls_pred` and `reg_pred`
            - `cls_pred`: the prediction dataframe for classification tasks: the columns are the tasks
                (`input_assay_id`_`threshold` from the `classification metadata` file) and the rows are the compounds
                ids (`input_compound_id` from the `smiles` file).
            - `reg_pred`: the prediction dataframe for regression tasks: the columns are the tasks (`input_assay_id`
                from the `regression metadata` file) and the rows are the compounds ids (`input_compound_id` from the
                `smiles` file).
        """

        data = sparsechem.fold_transform_inputs(
            prepared_data.data, folding_size=self._fold_inputs, transform=self._input_transform
        )

        classification_mask = prediction_mask((data.shape[0], int(self._class_output_size)), classification_tasks)
        regression_mask = prediction_mask((data.shape[0], int(self._regr_output_size)), regression_tasks)

        dataset_te = sparsechem.ClassRegrSparseDataset(x=data, y_class=classification_mask, y_regr=regression_mask)
        loader = DataLoader(
            dataset=dataset_te,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset_te.collate,
        )

        self.load()

        cls_pred, reg_pred = sparsechem.predict_sparse(
            net=self._model,
            loader=loader,
            dev=self._device,
            dropout=self._dropout,
            progress=False,
            y_cat_columns=self._y_cat_columns,
        )

        # apply inverse_normalization on the regression predictions if they are not empty (reg/hyb models)
        if reg_pred.shape[1] != 0:
            reg_pred = sparsechem.inverse_normalization(
                reg_pred, mean=np.array(self._stats["mean"]), variance=np.array(self._stats["var"]), array=True
            )

        cls_pred = ClassificationPrediction(cls_pred.toarray())
        reg_pred = RegressionPrediction(reg_pred.toarray())

        if cls_pred.has_tasks():
            cls_pred.map_task_ids(self._cls_metadata)
        if reg_pred.has_tasks():
            reg_pred.map_task_ids(self._reg_metadata)

        if self._load_on_demand:
            self.unload()

        for pred in [cls_pred, reg_pred]:
            pred.map_compound_ids(prepared_data.compound_ids)

        return cls_pred.dataframe, reg_pred.dataframe


def prediction_mask(shape: Tuple[int, int], tasks_ids: Optional[List[int]]) -> csr_matrix:
    """produce a mask that will be used for prediction

    Based on tasks_ids, we fill the columns with 1 to produce a mask that will generate prediction for this columns
    if tasks_ids is set to None we predict for all columns. If set to [] we don't predict for any column.
    """
    mask = lil_matrix(shape, dtype=np.float32)

    if tasks_ids is None:
        mask[:, :] = 1
    elif len(tasks_ids) != 0:
        mask[:, tasks_ids] = 1

    return mask.tocsr()
