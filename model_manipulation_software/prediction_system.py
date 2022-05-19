import os
import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import melloddy_tuner.tunercli  # type: ignore
import melloddy_tuner.utils.helper  # type: ignore
import numpy as np
import pandas as pd
import sparsechem  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse import lil_matrix  # type: ignore
from torch.utils.data import DataLoader

from model_manipulation_software.model import Model


class ModelUnknownError(Exception):
    pass


class PredictionSystem:
    """
    Initialize the prediction system of the model manipulation software.

    Args:
        model_folder (pathlib.Path): The path of the folder which contains all the models.
            Each `model` is a folder `model_name` (used in the `predict` function) which contains:
            - a `configuration` file `hyperparameters.json`
            - a `model checkpoint` file `model.pth`
            - `metadata` file(s) `T8_cls.csv` and/or `T8_reg.csv`
        encryption_key (pathlib.Path): Path of the encryption key `json` used to shuffle the bits of the descriptors
            (fingerprints) in `melloddy_tuner`.
            Ex: `inputs/config/example_key.json`.
        preparation_parameters (pathlib.Path): Path of the parameters `json` to be used to prepare the dataset with
            `melloddy_tuner`.
            Ex: `inputs/config/example_parameters.json`.
            More details in `melloddy_tuner`'s `README.md`, Section `# Parameter definitions`.
        device (str, optional): device used to load the model for the predictions. Defaults to `cpu`.

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
        encryption_key: pathlib.Path,
        preparation_parameters: pathlib.Path,
        device: str = "cpu",
    ):
        if not os.path.isfile(encryption_key):
            raise FileNotFoundError(encryption_key)
        if not os.path.isfile(preparation_parameters):
            raise FileNotFoundError(preparation_parameters)

        self._device = device

        self._models = {}

        self._tuner_encryption_key = encryption_key
        self._tuner_configuration_parameters = preparation_parameters

    def predict(
        self,
        model: Model,
        smiles: pd.DataFrame,
        classification_tasks: Optional[List[int]] = None,
        regression_tasks: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Predict on the test data (Smiles) with a given model.

        Args:
            model (Model): the model you want to use to perform predictions
            smiles (pd.DataFrame): The test data. A loaded T2 structure.
            classification_tasks: A list of tasks indexes (`cont_classification_task_id` from the `metadata file`) for
                which you want to predict. If not set it will predict on all classification tasks. If you don't want
                to predict on any tasks you can send an empty list.
            regression_tasks: A list of tasks indexes (`cont_regression_task_id` from the `metadata file`) for which
                you want to predict. If not set it will predict on all regression tasks. If you don't want to predict
                on any tasks you can sent an empty list.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: `cls_pred`, `reg_pred` and `failed_smiles`.
            - `cls_pred`: the prediction dataframe for classification tasks: the columns are the tasks
                (`input_assay_id`_`threshold` from the `classification metadata` file) and the rows are the compounds
                ids (`input_compound_id` from the `smiles` file).
            - `reg_pred`: the prediction dataframe for regression tasks: the columns are the tasks (`input_assay_id`
                from the `regression metadata` file) and the rows are the compounds ids (`input_compound_id` from the
                `smiles` file).
            - `failed_smiles`: the smiles which can't be processed. The rows are the compounds ids (`input_compound_id`
                from the `smiles` file), and the column `error_message` contains the error returned by `melloddy_tuner`
        """
        data, df_failed, compound_mapping = melloddy_tuner.tunercli.do_prepare_prediction_online(
            input_structure=smiles,
            key_path=str(self._tuner_encryption_key),
            config_file=str(self._tuner_configuration_parameters),
            num_cpu=1,
        )
        compound_ids = compound_mapping["input_compound_id"].reset_index().drop("index", axis=1)
        assert compound_ids["input_compound_id"].is_unique

        data = sparsechem.fold_transform_inputs(data, folding_size=model.fold_inputs, transform=model.input_transform)

        classification_mask = prediction_mask((data.shape[0], int(model.class_output_size)), classification_tasks)
        regression_mask = prediction_mask((data.shape[0], int(model.regr_output_size)), regression_tasks)

        dataset_te = sparsechem.ClassRegrSparseDataset(x=data, y_class=classification_mask, y_regr=regression_mask)
        loader = DataLoader(
            dataset=dataset_te, batch_size=4000, num_workers=4, pin_memory=True, collate_fn=dataset_te.collate
        )

        model.load(self._device)

        cls_pred, reg_pred = model.predict(loader)

        for pred in [cls_pred, reg_pred]:
            pred.map_compound_ids(compound_ids)

        return cls_pred.dataframe, reg_pred.dataframe, df_failed


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
