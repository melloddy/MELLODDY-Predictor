import os
import pathlib
from typing import Dict
from typing import Tuple

import melloddy_tuner.tunercli  # type: ignore
import melloddy_tuner.utils.helper  # type: ignore
import pandas as pd
import sparsechem  # type: ignore
from torch.utils.data import DataLoader

from model_manipulation_software.model import Model

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

        for pred in [cls_pred, reg_pred]:
            pred.map_compound_ids(compound_ids)

        return cls_pred.dataframe, reg_pred.dataframe, df_failed
