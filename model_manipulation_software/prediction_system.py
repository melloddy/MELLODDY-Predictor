import os
import pathlib
import tempfile
from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import Tuple

import melloddy_tuner.tunercli  # type: ignore
import melloddy_tuner.utils.helper  # type: ignore
import numpy as np
import sparsechem  # type: ignore
import torch
from torch.utils.data import DataLoader

STRUCTURE_FILE = "structure_file"
CONFIG_FILE = "config_file"
KEY_FILE = "key_file"
OUTPUT_DIR = "output_dir"
RUN_NAME = "run_name"
NUMBER_CPU = "number_cpu"
REF_HASH = "ref_hash"
NON_INTERACTIVE = "non_interactive"
PREPARATION_RUN_NAME = "mms"


class ModelUnknownError(Exception):
    pass


class Model:
    _conf: SimpleNamespace
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
    def conf(self) -> SimpleNamespace:
        if not hasattr(self, "_conf") or not self._conf:
            self._conf: SimpleNamespace = sparsechem.load_results(str(self._conf_path), two_heads=True)["conf"]
        return self._conf

    @property
    def class_output_size(self) -> str:
        return self.conf.class_output_size

    @property
    def regr_output_size(self) -> str:
        return self.conf.regr_output_size

    @property
    def fold_inputs(self) -> str:
        return self.conf.fold_inputs

    @property
    def input_transform(self) -> str:
        return self.conf.input_transform

    def load(self, device: str = "cpu") -> None:
        """Loads the model on the specified device

        Args:
            device (str, optional): device on which we should load the model (cpu, cuda:0 .. cuda:x). Defaults to "cpu".
        """
        if not hasattr(self, "_model") or not self._model:
            self._device = device
            self._model = sparsechem.SparseFFN(self.conf).to(self._device)
            state = torch.load(self._model_path, map_location=torch.device(self._device))
            self._model.load_state_dict(state)

    def predict(self, data: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts using the model

        Args:
            data (DataLoader): data on which we should run predictions

        Returns:
            Tuple[np.ndarray, np.ndarray]: first item is classification matrix and second is regression matrix
        """
        return sparsechem.predict(
            net=self._model, loader=data, dev=self._device, dropout=0, progress=False, y_cat_columns=None
        )


class PredictionSystem:
    """Prediction system exposes predictions for SMILES from model given on init"""

    _device: str
    _tuner_encryption_key: pathlib.Path
    _tuner_configuration_parameters: pathlib.Path
    _task_metadata: pathlib.Path
    _reg_processed_metadata: pathlib.Path
    _cls_processed_metadata: pathlib.Path
    _models: Dict[str, Model]

    def __init__(
        self,
        model_folder: pathlib.Path,
        permutation_key: pathlib.Path,
        preparation_parameters: pathlib.Path,
        task_metadata: pathlib.Path,
        reg_processed_metadata: pathlib.Path,
        cls_processed_metadata: pathlib.Path,
        device: str = "cpu",
    ):
        self._device = device
        self._tuner_encryption_key = permutation_key
        self._tuner_configuration_parameters = preparation_parameters
        self._task_metadata = task_metadata
        self._reg_processed_metadata = reg_processed_metadata
        self._cls_processed_metadata = cls_processed_metadata

        if not os.path.isdir(model_folder):
            raise NotADirectoryError(f"{model_folder} is not a directory")

        self._models = {}

        for dirname in os.listdir(model_folder):
            self._models[dirname] = Model(model_folder / dirname)

    def _build_data_preparation_args(
        self,
        data_file: pathlib.Path,
        output_dir: pathlib.Path,
    ) -> Namespace:
        namespace: Dict[str, Any] = {}

        namespace[STRUCTURE_FILE] = str(data_file)
        namespace[CONFIG_FILE] = str(self._tuner_configuration_parameters)
        namespace[KEY_FILE] = str(self._tuner_encryption_key)
        namespace[OUTPUT_DIR] = str(output_dir)
        namespace[RUN_NAME] = PREPARATION_RUN_NAME
        namespace[NUMBER_CPU] = 1
        namespace[NON_INTERACTIVE] = True
        namespace[REF_HASH] = None

        return Namespace(**namespace)

    def _get_model(self, model_name: str) -> Model:
        try:
            model = self._models[model_name]
        except KeyError:
            raise ModelUnknownError("Requested model does not exist")
        return model

    def predict(self, model_name: str, smiles: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
        model = self._get_model(model_name)

        with tempfile.TemporaryDirectory("mms") as data_dir:
            output_dir = pathlib.Path(data_dir)
            melloddy_tuner.tunercli.do_prepare_prediction(
                self._build_data_preparation_args(
                    smiles,
                    output_dir,
                )
            )
            data_path = output_dir / PREPARATION_RUN_NAME / "matrices" / "pred_x.npz"
            data = sparsechem.load_sparse(str(data_path))
            data = sparsechem.fold_transform_inputs(
                data, folding_size=model.fold_inputs, transform=model.input_transform
            )

        y_class = sparsechem.load_check_sparse(filename=None, shape=(data.shape[0], model.class_output_size))
        y_regr = sparsechem.load_check_sparse(filename=None, shape=(data.shape[0], model.regr_output_size))

        dataset_te = sparsechem.ClassRegrSparseDataset(x=data, y_class=y_class, y_regr=y_regr)
        loader = DataLoader(
            dataset=dataset_te, batch_size=4000, num_workers=4, pin_memory=True, collate_fn=dataset_te.collate
        )

        model.load(self._device)

        return model.predict(loader)
