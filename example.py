from pathlib import Path

import melloddy_tuner.utils.helper  # type: ignore
import pandas as pd

from model_manipulation_software import Model
from model_manipulation_software import PredictionSystem

predictor = PredictionSystem(
    encryption_key=Path("inputs/config/example_key.json"),
    preparation_parameters=Path("inputs/config/example_parameters.json"),
)
smiles_path = Path("inputs/data/T2_100samples.csv")
df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(smiles_path))
cls_pred, reg_pred, failed_smiles = predictor.predict(
    model=Model(Path("inputs/models/example_cls_model")), smiles=df, classification_tasks=[0, 1, 2]
)

print("\n cls_pred dataframe : \n")
print(cls_pred.head(10))
