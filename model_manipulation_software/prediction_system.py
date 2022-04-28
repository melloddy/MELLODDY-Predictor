import os
import pathlib
from types import SimpleNamespace
from typing import Dict
from typing import Tuple

import melloddy_tuner.tunercli  # type: ignore
import melloddy_tuner.utils.helper  # type: ignore
import numpy as np
import pandas as pd
import sparsechem  # type: ignore
import torch
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader

STRUCTURE_FILE = "structure_file"
CONFIG_FILE = "config_file"
KEY_FILE = "key_file"
RUN_NAME = "run_name"
NUMBER_CPU = "number_cpu"
REF_HASH = "ref_hash"
NON_INTERACTIVE = "non_interactive"
PREPARATION_RUN_NAME = "mms"


class ModelUnknownError(Exception):
    pass


class Model:
    """
    A sparsechem model and its configuration

    Args:
        path (pathlib.Path): the path of the model's folder. Contains the files `hyperparameters.json` and `model.pth`.
        "hyperparameters.json" should contain at least a "conf" dict with "input_transform" and "fold_inputs"

    Raises:
        FileNotFoundError: path / "hyperparameters.json" not found
        FileNotFoundError: path / "model.pth" not found
    """

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
        """
        The configuration of the model, which contains the "conf" values of the "hyperparameters.json" as well as
            - "model_type"
            - "class_output_size"
            - "regr_output_size"

        Returns:
            SimpleNamespace: Namespace sent by sparsechem
        """
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

    @property
    def reg_metadata(self) -> pd.DataFrame:
        return self._load_metadata("T8_reg.csv")

    @property
    def cls_metadata(self) -> pd.DataFrame:
        return self._load_metadata("T8_cls.csv")

    def _load_metadata(self, filename) -> pd.DataFrame:
        metadata_file = self.path / filename
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(metadata_file)
        data = pd.read_csv(metadata_file)
        return data if isinstance(data, pd.DataFrame) else DataFrame()

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
    """
    Prediction system exposes predictions for SMILES from model given on init

    Args:
        model_folder (pathlib.Path): The path of the folder which contains all the models.
            Each model is a folder `model_name` (used in the `predict` function) which contains the files
            `hyperparameters.json` and `model.pth`.
        permutation_key (pathlib.Path): Path of the encryption key `json` used to shuffle the bits of the descriptors
            (fingerprints) in `melloddy_tuner`.
            Ex: `inputs/config/example_key.json`.
        preparation_parameters (pathlib.Path): Path of the parameters `json` to be used to prepare the dataset with
            `melloddy_tuner`.
            Ex: `inputs/config/example_parameters.json`.
            More details in `melloddy_tuner` `README.md`, `# Parameter definitions`.
        device (str, optional): device used to load the model for the predictions. Defaults to "cpu".

    Raises:
        NotADirectoryError: `model_folder` is not a directory
        ModelUnknownError: Requested model does not exist
    """

    _device: str
    _tuner_encryption_key: pathlib.Path
    _tuner_configuration_parameters: pathlib.Path
    _models: Dict[str, Model]

    def __init__(
        self,
        model_folder: pathlib.Path,
        encryption_key: pathlib.Path,
        preparation_parameters: pathlib.Path,
        device: str = "cpu",
    ):
        if not os.path.isdir(model_folder):
            raise NotADirectoryError(f"{model_folder} is not a directory")
        if not os.path.isfile(encryption_key):
            raise FileNotFoundError(encryption_key)
        if not os.path.isfile(preparation_parameters):
            raise FileNotFoundError(preparation_parameters)

        self._device = device

        self._models = {}
        for dirname in os.listdir(model_folder):
            self._models[dirname] = Model(model_folder / dirname)

        self._tuner_encryption_key = encryption_key
        self._tuner_configuration_parameters = preparation_parameters

    def _get_model(self, model_name: str) -> Model:
        try:
            model = self._models[model_name]
        except KeyError:
            raise ModelUnknownError("Requested model does not exist")
        return model

    def predict(self, model_name: str, smiles: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Predict on the test data (Smiles) with a given model.

        Args:
            model_name (str): the folder name of the model, which should be in the `model_folder` given at the init.
                Contains the files `hyperparameters.json` and `model.pth`.
            smiles (pathlib.Path): The test data. Path of the T2 structure input file (Smiles) in `csv` format.

        Returns:
            Tuple[np.ndarray, np.ndarray]: cls_pred and reg_pred, the predictions matrixes for classification and
            regression tasks. For each array, the columns are the tasks and the rows are the samples.
        """
        model = self._get_model(model_name)

        df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(smiles))
        data, df_failed, compound_mapping = melloddy_tuner.tunercli.do_prepare_prediction_online(
            input_structure=df,
            key_path=self._tuner_encryption_key,
            config_file=self._tuner_configuration_parameters,
            num_cpu=1,
        )
        compound_ids = compound_mapping["input_compound_id"].reset_index().drop("index", axis=1)

        data = sparsechem.fold_transform_inputs(data, folding_size=model.fold_inputs, transform=model.input_transform)

        y_class = sparsechem.load_check_sparse(filename=None, shape=(data.shape[0], model.class_output_size))
        y_regr = sparsechem.load_check_sparse(filename=None, shape=(data.shape[0], model.regr_output_size))

        dataset_te = sparsechem.ClassRegrSparseDataset(x=data, y_class=y_class, y_regr=y_regr)
        loader = DataLoader(
            dataset=dataset_te, batch_size=4000, num_workers=4, pin_memory=True, collate_fn=dataset_te.collate
        )

        model.load(self._device)

        cls_pred, reg_pred = model.predict(loader)

        if cls_pred.shape[1] != 0:
            cls_pred_df = map_task_ids(metadata=model.cls_metadata, pred=cls_pred, pb_type="classification")
        else:
            cls_pred_df = pd.DataFrame(cls_pred)

        cls_pred_df = map_compound_ids(compound_ids, cls_pred_df)

        if reg_pred.shape[1] != 0:
            reg_pred_df = map_task_ids(metadata=model.reg_metadata, pred=cls_pred, pb_type="regression")
        else:
            reg_pred_df = pd.DataFrame(reg_pred)

        reg_pred_df = map_compound_ids(compound_ids, reg_pred_df)

        return cls_pred_df, reg_pred_df, df_failed


def map_task_ids(metadata: pd.DataFrame, pred: np.ndarray, pb_type: str):
    metadata_df = metadata.dropna(subset=[f"cont_{pb_type}_task_id"])
    if metadata_df is not None:
        tasks = metadata_df["input_assay_id"].astype(str) + "_" + metadata_df["threshold"].astype(str)
        return pd.DataFrame(pred, columns=tasks)
    else:
        raise RuntimeError("Cannot map an empty tasks DataFrame")


def map_compound_ids(compound_ids: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([compound_ids, predictions], axis=1)
