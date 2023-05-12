from collections import OrderedDict
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import sparsechem as sc
import torch
from scipy.sparse import csr_matrix


def csr_to_torch_coo(csr_mat: csr_matrix) -> torch.Tensor:
    """Converts from scipy sparse csr matrix to a torch sparsae_coo_tensor to be
    submitted to the sparsechem network Sparsewchem requires.

    Args:
        csr_matx (scipy.sparse.csr_matrix) sprse csr matrix to convert

    Returns:
        torch.sparse_coo_tensor
    """
    coo_mat = csr_mat.tocoo()
    return torch.sparse_coo_tensor(
        indices=torch.Tensor([coo_mat.row, coo_mat.col]),
        values=coo_mat.data,
        size=coo_mat.shape,
        dtype=torch.float,
    )


class ScModelType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1
    HYBRID = 2


class PredictorSingle:
    """This class handles predictions for single instances.

    It bypasses a lot of mechansims for batched data loading
    """

    def __init__(
        self,
        model,
        conf,
        class_task_map=None,
        regr_task_map=None,
        dropout=False,
        device="cpu",
    ):
        """Inititialze the predictor object.

        Args:
            model:                       filename of the model pytorch model file
            conf:                        filename of the the corresponding configuration file for the model
            class_task_map:              a dictionary or pandas series having classification task labels as
                as key or index, resp, and continuous classification task IDs (column indexes of the prediction
                matrix) as values
            regr_task_map:               a dictionary or pandas series having regression task labels as as key
                or index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values
            dropout(bool):               whether to apply dropout or nor
            device:                      device to run on, per default cpu
        """
        results_loaded = sc.load_results(conf, two_heads=True)
        self.conf = results_loaded["conf"]
        self.device = device
        self.net = sc.SparseFFN(self.conf).to(self.device)
        self.inverse_normalization = False
        state_dict = OrderedDict(torch.load(model, map_location=torch.device(self.device)))

        if self.conf.model_type == "federated":
            state_dict_new = OrderedDict()
            state_dict_new["net.0.net_freq.weight"] = state_dict["0.0.net_freq.weight"]
            state_dict_new["net.0.net_freq.bias"] = state_dict["0.0.net_freq.bias"]
            state_dict_new["net.2.net.2.weight"] = state_dict["1.net.2.weight"]
            state_dict_new["net.2.net.2.bias"] = state_dict["1.net.2.bias"]
            state_dict = state_dict_new

        # apply model weights
        self.net.load_state_dict(state_dict)
        # set model into evaluation mode
        self.net.eval()
        # apply dropout, if chosen
        self.dropout = dropout
        if self.dropout:
            self.net.apply(sc.utils.enable_dropout)
        # if inverse normalization is done load the stats
        if "stats" in results_loaded:
            self.inverse_normalization = True
            stats = results_loaded["stats"]
            self.reg_mean = np.array(stats["mean"])
            self.reg_var = np.array(stats["var"])
            self.reg_stddev = np.sqrt(self.reg_var)

        if self.net.cat_id_size is not None:
            raise NotImplementedError("Predictions for models with a catalog head are not yet implemented")
        if self.net.class_output_size > 0:
            if self.net.regr_output_size > 0:
                self.model_type: ScModelType = ScModelType.HYBRID
            else:
                self.model_type = ScModelType.CLASSIFICATION
        elif self.net.regr_output_size > 0:
            self.model_type = ScModelType.REGRESSION  # pylint: disable=redefined-variable-type

        self.has_task_maps = False
        if (class_task_map is not None) or (regr_task_map is not None):
            self.set_tasks_maps(class_task_map, regr_task_map)

    def set_tasks_maps(self, class_task_map: dict = None, regr_task_map: dict = None):
        """Set the task maps stored in the object.

        Args:
            class_task_map:              a dictionary or pandas series having classification task labels
                as as key or index, resp, and continuous classification task IDs (column indexes of the prediction
                matrix) as values
            regr_task_map:               a dictionary or pandas series having regression task labels as as
                key or index, resp, and continuous regression task IDs (column indexes of the prediction
                matrix) as values
        """
        class_task_map, regr_task_map, mapped_tasks_type = self.__validate_maps__(class_task_map, regr_task_map)
        self.class_task_map = class_task_map
        self.regr_task_map = regr_task_map
        self.mapped_tasks_type = mapped_tasks_type
        self.has_task_maps = True

    def get_mapped_task_names(self):
        if not self.has_task_maps:
            return None
        if self.mapped_tasks_type == ScModelType.CLASSIFICATION:
            return self.class_task_map.index.values
        if self.mapped_tasks_type == ScModelType.REGRESSION:
            return self.regr_task_map.index.values
        if self.mapped_tasks_type == ScModelType.HYBRID:
            return np.concatenate([self.class_task_map.index.values, self.regr_task_map.index.values])
        return None

    def get_model_type(self):
        return self.model_type

    def get_mapped_task_type(self):
        if self.has_task_maps:
            return self.mapped_tasks_type
        return None

    def get_num_class_tasks(self):
        return self.conf.class_output_size

    def get_num_regr_tasks(self):
        return self.conf.regr_output_size

    def get_num_tasks(self):
        return self.conf.class_output_size + self.conf.regr_output_size

    def get_num_tasks_by_type(self, task_type):
        if task_type == ScModelType.CLASSIFICATION:
            return self.get_num_class_tasks()
        if task_type == ScModelType.REGRESSION:
            return self.get_num_regr_tasks()
        raise ValueError(f"Non permitted task type {task_type}")

    def __validate_maps__(
        self, class_task_map: Union[dict, pd.Series, None], regr_task_map: Union[dict, pd.Series, None]
    ):
        class_task_map = self.__validate_map__(class_task_map, ScModelType.CLASSIFICATION)
        regr_task_map = self.__validate_map__(regr_task_map, ScModelType.REGRESSION)
        # test for non overlap of task labels between classification and regression
        if class_task_map is not None:
            if regr_task_map is not None:
                mapped_tasks_type: ScModelType = ScModelType.HYBRID
            else:
                mapped_tasks_type = ScModelType.CLASSIFICATION
        elif regr_task_map is not None:
            mapped_tasks_type = ScModelType.REGRESSION  # pylint: disable=redefined-variable-type
        else:
            raise ValueError("Task maps for both classification and regression are None")
        if mapped_tasks_type == ScModelType.HYBRID:
            if (
                class_task_map is not None
                and regr_task_map is not None
                and isinstance(class_task_map, pd.Series)
                and isinstance(regr_task_map, pd.Series)
                and class_task_map.index.intersection(regr_task_map.index).size > 0
            ):
                raise ValueError(
                    "classification and regression task map have task labels in common, this is not permitted"
                )
        return class_task_map, regr_task_map, mapped_tasks_type

    def __validate_map__(self, task_map: Union[dict, pd.Series, None], task_type):
        if task_map is not None:
            if self.get_num_tasks_by_type(task_type) == 0:
                raise ValueError(
                    "A {0} task map has been provided for a model without {0} tasks".format(  # pylint: disable=consider-using-f-string # noqa: E501
                        task_type.name
                    )
                )
            if isinstance(task_map, pd.Series):
                pass
            elif isinstance(task_map, dict):
                task_map = pd.Series(task_map)
            else:
                raise TypeError(
                    "{0} task_map needs be either of type pandas.Series or be a dictionary, but is a {1}".format(  # pylint: disable=consider-using-f-string,line-too-long # noqa: E501
                        task_type.name, type(task_map)
                    )
                )
            if not task_map.dtype == int:
                raise TypeError(f"The {task_type.name} task_map needs to have values of type int")
            if task_map.max() >= self.get_num_tasks_by_type(task_type):
                raise ValueError(
                    "The maximum value of {0} task_map exceeps the number of {0} outputs ({1})".format(  # pylint: disable=consider-using-f-string # noqa: E501
                        task_type.name, self.get_num_tasks_by_type(task_type)
                    )
                )
            if not task_map.is_unique:
                raise ValueError(f"the task indexes in {task_type.name} task map are not unique")
            if not task_map.index.is_unique:
                raise ValueError(f"the task labels in {task_type.name} task map are not unique")
        return task_map

    def predict_from_csr(self, x_csr: csr_matrix) -> tuple:
        """Feed the input csr matrix in on the go to the neural net for prediction, by
        passing the torch data loader This is meant to be used for small batches Returns
        a dense numpy array for classification and regression tasks.

        Args:
            x_csr(scipy.sparse.csr_matrix) : a scipy sparse csr matrix with fingerprint features

        Returns:
            Tupe(np.array, np.array) with classifcation and regression predictions
        """
        X = csr_to_torch_coo(x_csr)
        y_class_array, y_regr_array = self.predict_from_tensor(X)
        return y_class_array, y_regr_array

    def predict_from_tensor(self, X: torch.Tensor) -> tuple:
        """Feed the input torch sparse coo_tensor in on the go to the neural net for
        prediction, by passing the torch data loader This is meant to be used for small
        batches.

        Args:
            X(torch.sparse_coo_tensor) : a torch sparse coo tensor matrix with fingerprint features

        Returns:
            Tupe(np.array, np.array) with classifcation and regression predictions
        """
        # don't compute gradients
        with torch.no_grad():
            if self.net.cat_id_size is None:
                y_class, y_regr = self.net.forward(X.to(self.device))
            else:
                y_class, y_regr, _ = self.net.forward(X.to(self.device))
            y_class_array = torch.sigmoid(y_class).cpu().numpy()
            y_regr_array = y_regr.cpu().numpy()
            if self.inverse_normalization:
                # y_regr_array  = sc.inverse_normalization(csr_matrix(y_regr_array) , mean=self.reg_mean, \
                #                                         variance=self.reg_var, array=True)
                y_regr_array = y_regr_array * self.reg_stddev + self.reg_mean
        return y_class_array, y_regr_array

    @staticmethod
    def extract_tasks(y_array, task_map):
        return pd.Series(y_array[0, task_map.values], index=task_map.index)

    def predict_decorated_series_from_tensor(
        self,
        X: torch.Tensor,
        class_task_map: Union[dict, pd.Series] = None,
        regr_task_map: Union[dict, pd.Series] = None,
        limit_to_type: ScModelType = None,
    ) -> pd.Series:
        """This runs the prediction on the input tensor expected to have single row and
        extracts the desired tasks based on the information in the task maps that have
        been passed either on predictor intitialization, or with the call of this
        function (having precedence). It extracts from the raw prediction the tasks of
        interstest as specified through the task map(s) and warps them into a series
        having the task lables of the task maps index as series index. Predictions for
        tasks, which index is not listed in the tasks maps(s) are not included in the
        returned series.

        Args:
            X(torch.Tensor) : a torch sparse coo tensor matrix with fingerprint features
            class_task_map: a dictionary or pandas series having classification task labels as as key
                or index, resp, and continuous classification task IDs (column indexes of the prediction
                matrix) as values
            regr_task_map: a dictionary or pandas series having regression task labels as as key or
                index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values
            limit_to_type: if not None, only tasks of the specified type are returned

        Returns:
            pd.Series of predictions with task labels as index

        Raises:
            ValueError: if the input tensor has more than one row
            ValueError: if the task maps are not of type dict or pandas.Series
            ValueError: if the task maps have values of type other than int
        """
        # This function expects receiving a tensor witrh a single row, as we also only return results for the first row
        if X.size(0) != 1:
            raise ValueError(
                f"This function expects only single row tensor, but the tensor passed has size of {X.size(0)}"
            )

        # if task maps are passed as arguments we use them
        if class_task_map is not None or regr_task_map is not None:
            class_task_map, regr_task_map, mapped_tasks_type = self.__validate_maps__(class_task_map, regr_task_map)
        # otherwise we fall back to the tasks maps passed upon intialization, if peresent
        elif self.has_task_maps:
            class_task_map = self.class_task_map
            regr_task_map = self.regr_task_map
            mapped_tasks_type = self.mapped_tasks_type
        # if those don't exist, the function cannpot proceed
        else:
            raise ValueError(
                "Task maps must be passed either at intialization time of the predictor or when calling the prediction"
                " functions"
            )

        y_class_array, y_regr_array = self.predict_from_tensor(X)

        if mapped_tasks_type == ScModelType.HYBRID and limit_to_type is not None:
            if limit_to_type in [ScModelType.CLASSIFICATION, ScModelType.REGRESSION]:
                mapped_tasks_type = limit_to_type
            else:
                raise ValueError(f"Not permitted type {limit_to_type} has been provided for limit_to_type")

        if mapped_tasks_type == ScModelType.CLASSIFICATION:
            results = self.extract_tasks(y_class_array, class_task_map)
        elif mapped_tasks_type == ScModelType.REGRESSION:
            results = self.extract_tasks(y_regr_array, regr_task_map)
        elif mapped_tasks_type == ScModelType.HYBRID:
            results = pd.concat(
                [
                    self.extract_tasks(y_class_array, class_task_map),
                    self.extract_tasks(y_regr_array, regr_task_map),
                ]
            )

        return results

    def predict_decorated_series_from_csr(
        self,
        x_csr: csr_matrix,
        class_task_map: Union[dict, pd.Series] = None,
        regr_task_map: Union[dict, pd.Series] = None,
        limit_to_type: ScModelType = None,
    ) -> pd.Series:
        """This runs the prediction on the input csr_matrix expected to have single row
        and extracts the desired tasks based on the information in the task maps that
        have been passed either on predictor initialization, or with the call of this
        function (having precedence). It extracts from the raw prediction the tasks of
        interest as specified through the task map(s) and warps them into a series
        having the task labels of the task maps index as series index. Predictions for
        tasks, which index is not listed in the tasks maps(s) are not included in the
        returned series.

        Args:
            x_csr(csr_matrix) : a torch sparse coo tensor matrix with fingerprint features
                class_task_map: a dictionary or pandas series having classification task labels as as key
                or index, resp, and continuous classification task IDs (column indexes of the prediction matrix)
                as values
            class_task_map(Union[dict, pd.Series], optional): a dictionary or pandas series having classification
                task labels as as key
                or index, resp, and continuous classification task IDs (column indexes of the prediction matrix)
            regr_task_map(Union[dict, pd.Series], optional): a dictionary or pandas series having regression task
                labels as as key or
                index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values
            limit_to_type: if not None, only tasks of the specified type are returned
        Returns:
            pd.Series of predictions with task labels as index
        """
        X = csr_to_torch_coo(x_csr)
        return self.predict_decorated_series_from_tensor(X, class_task_map, regr_task_map, limit_to_type=limit_to_type)

    def predict_trunk_from_tensor(self, X: torch.Tensor) -> np.ndarray:
        """This function computes the last hidden layer of the model.

        Args:
            X (torch.sparse_coo_tensor) : fingerprint features as tporch sparse_coo_tensor

        Returns:
            numpy.ndarray of hidden layer values
        """

        with torch.no_grad():
            return self.net.forward(X.to(self.device), trunk_embeddings=True).cpu().numpy()


def t8df_to_task_map(
    t8_df: pd.DataFrame,
    task_type: str,
    name_column: str = "input_assay_id",
    threshold_multi_ix=False,
) -> pd.Series:
    """This function extracts from a t8 type dataframe (or a selected slice thereof) a
    task_map for the predictor object.

    Args:
        t8_df (pandas.DataFrame): dataframe to extract the task map from
        task_type (str):          either "classification" or "regression"
        name_column (str) :       column in datafarem to use as task labels
        threshold_multi_ix (bool): Whether to create a multi index with class_labela nd threshold
            as index columns. Default False

    Returns:
        pandas.Series: task map with task labels as index and task IDs as values

    Raises:
        ValueError: if task_type is not "classification" or "regression"
        ValueError: if task index column is not present in the task dataframe
        ValueError: if null value task indices are present in data frame
        ValueError: if duplicate task indices are present in data frame
        ValueError: if task name column is not present in the task dataframe
    """
    temp_df = t8_df.copy()
    if task_type not in ("classification", "regression"):
        raise ValueError(f'Task type must be either "classification" or "regression", passed type is {task_type}')
    task_id_column = f"cont_{task_type}_task_id"
    if task_id_column not in temp_df:
        raise ValueError(f'task index column "{task_id_column}" is not present in the task dataframe')
    if temp_df[task_id_column].isnull().any():
        raise ValueError("Null value continuous task indices are present in data frame")
    if temp_df[task_id_column].duplicated().any():
        raise ValueError("Duplicate continuous task indices are present in data frame")
    if name_column not in temp_df:
        raise ValueError(f'task name column "{name_column}" is not present in the task dataframe')
    if task_type == "classification":
        if threshold_multi_ix:
            temp_df["task_labels"] = temp_df.apply(lambda x: f"assay_{x[name_column]}_class", axis=1)
        else:
            temp_df["task_labels"] = temp_df.apply(
                lambda x: f"assay_{x[name_column]}_class_{x['threshold']:0.2f}",
                axis=1,
            )
    else:
        temp_df["task_labels"] = temp_df.apply(lambda x: f"assay_{x[name_column]}_value", axis=1)
    temp_df = temp_df.set_index("task_labels")
    if threshold_multi_ix:
        temp_df = temp_df.set_index("threshold", append=True)
    if not temp_df.index.is_unique:
        raise ValueError(
            "task labels are not unique, try to use a different name column, and/or make use of concat_threshold or "
            " option"
        )
    return temp_df[task_id_column].astype(int)
