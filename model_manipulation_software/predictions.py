from typing import Union

import numpy as np
import pandas as pd


class PredictionMappingError(Exception):
    pass


class Prediction:
    def __init__(self, pred_matrix: np.ndarray):
        self.pred = pd.DataFrame(pred_matrix)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.pred

    def has_tasks(self) -> bool:
        return self.pred.shape[1] != 0

    def map_task_ids(self, metadata: pd.DataFrame):
        del metadata  # This class does not propose an implementation for this
        raise NotImplementedError()

    def map_compound_ids(self, compound_ids: Union[pd.DataFrame, pd.Series]) -> None:
        self.pred = pd.concat([compound_ids, self.pred], axis=1)
        self.pred.set_index("input_compound_id", drop=True, inplace=True)


class ClassificationPrediction(Prediction):
    def map_task_ids(self, metadata: pd.DataFrame):
        filtered_metadata = metadata.dropna(subset=["cont_classification_task_id"])
        if filtered_metadata is not None:
            tasks = filtered_metadata["input_assay_id"].astype(str) + "_" + filtered_metadata["threshold"].astype(str)
            if not tasks.is_unique:
                raise PredictionMappingError("Tasks are not unique")
            if not len(self.pred.columns) == len(tasks):
                raise PredictionMappingError(
                    "The amount of tasks in the T8c file is different from the amout of tasks outputed by the model."
                )
            self.pred.columns = tasks


class RegressionPrediction(Prediction):
    def map_task_ids(self, metadata: pd.DataFrame):
        filtered_metadata = metadata.dropna(subset=["cont_regression_task_id"])
        if filtered_metadata is not None:
            tasks = filtered_metadata["input_assay_id"].astype(str)
            if not tasks.is_unique:
                raise PredictionMappingError("Tasks are not unique")
            if not len(self.pred.columns) == len(tasks):
                raise PredictionMappingError(
                    "The amount of tasks in the T8c file is different from the amout of tasks outputed by the model."
                )
            self.pred.columns = tasks
