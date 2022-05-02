import numpy as np
import pandas as pd


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

    def _map_task_ids(self, metadata: pd.DataFrame, problem_type: str) -> None:
        filtered_metadata = metadata.dropna(subset=[f"cont_{problem_type}_task_id"])
        if filtered_metadata is not None:
            tasks = filtered_metadata["input_assay_id"].astype(str) + "_" + filtered_metadata["threshold"].astype(str)
            self.pred.columns = tasks

    def map_compound_ids(self, compound_ids: pd.DataFrame) -> None:
        self.pred = pd.concat([compound_ids, self.pred], axis=1)


class ClassificationPrediction(Prediction):
    pb_type: str = "classification"

    def map_task_ids(self, metadata: pd.DataFrame):
        return super()._map_task_ids(metadata, self.pb_type)


class RegressionPrediction(Prediction):
    pb_type: str = "regression"

    def map_task_ids(self, metadata: pd.DataFrame) -> None:
        return super()._map_task_ids(metadata, self.pb_type)
