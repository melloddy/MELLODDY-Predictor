import os
import pathlib
from types import SimpleNamespace
from typing import Tuple

import pandas as pd
import sparsechem  # type: ignore
import torch
from torch.utils.data import DataLoader

from model_manipulation_software.predictions import ClassificationPrediction
from model_manipulation_software.predictions import Prediction
from model_manipulation_software.predictions import RegressionPrediction


class Model:
    """
    A sparsechem model and its configuration

    Args:
        path (pathlib.Path): the path of the model's folder.
            The directory structure inside this folder should be the following:

                my_model/
                ├─ hyperparameters.json
                ├─ model.pth
                ├─ T8_cls.csv
                ├─ T8_reg.csv

            The `hyperparameters` file should contain at least a `conf` key with informations about the model.
            The T8 metadata files should be provided based on the kind of the model:

            * CLS model: `T8_cls`.
            * REG model: `T8_reg`.
            * HYB model: `T8_cls` and `T8_reg`.

            The model should be compatible with `sparsechem` `0.9.6+`. If it is not, you can convert it with
            [this script](https://git.infra.melloddy.eu/wp2/sparsechem/-/blob/convert_v0.9.5_to_v0.9.6/examples/chembl/convert.py).

    Raises:
        FileNotFoundError: path / "hyperparameters.json" not found
        FileNotFoundError: path / "model.pth" not found
    """  # noqa: E501

    _internal_conf: SimpleNamespace
    _model: sparsechem.SparseFFN

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._conf_path = self.path / "hyperparameters.json"
        self._model_path = self.path / "model.pth"

        if not os.path.isfile(self._conf_path):
            raise FileNotFoundError(self._conf_path)
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(self._model_path)

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
    def class_output_size(self) -> str:
        return self._conf.class_output_size

    @property
    def regr_output_size(self) -> str:
        return self._conf.regr_output_size

    @property
    def fold_inputs(self) -> str:
        return self._conf.fold_inputs

    @property
    def input_transform(self) -> str:
        return self._conf.input_transform

    @property
    def _reg_metadata(self) -> pd.DataFrame:
        return self._load_metadata("T8_reg.csv")

    @property
    def _cls_metadata(self) -> pd.DataFrame:
        return self._load_metadata("T8_cls.csv")

    def _load_metadata(self, filename) -> pd.DataFrame:
        metadata_file = self.path / filename
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(metadata_file)
        data = pd.read_csv(metadata_file)
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()

    def load(self, device: str = "cpu") -> None:
        """Loads the model on the specified device

        Args:
            device (str, optional): device on which we should load the model (cpu, cuda:0 .. cuda:x). Defaults to "cpu".
        """
        if not hasattr(self, "_model") or not self._model:
            self._device = device
            self._model = sparsechem.SparseFFN(self._conf).to(self._device)
            state = torch.load(self._model_path, map_location=torch.device(self._device))
            self._model.load_state_dict(state)

    def predict(self, data: DataLoader) -> Tuple[Prediction, Prediction]:
        """Predicts using the model

        Args:
            data (DataLoader): data on which we should run predictions

        Returns:
            Tuple[np.ndarray, np.ndarray]: first item is classification matrix and second is regression matrix
        """
        cls, reg = sparsechem.predict_sparse(
            net=self._model, loader=data, dev=self._device, dropout=0, progress=False, y_cat_columns=None
        )

        cls = ClassificationPrediction(cls.toarray())
        reg = RegressionPrediction(reg.toarray())

        if cls.has_tasks():
            cls.map_task_ids(self._cls_metadata)
        if reg.has_tasks():
            reg.map_task_ids(self._reg_metadata)

        return cls, reg
