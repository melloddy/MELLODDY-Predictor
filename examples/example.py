from pathlib import Path

import melloddy_tuner.utils.helper  # type: ignore
import pandas as pd

from model_manipulation_software import Model
from model_manipulation_software import PreparedData

smiles_path = Path("inputs/data/T2_100samples.csv")
df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(smiles_path))

prepared_data = PreparedData(
    encryption_key=Path("inputs/config/example_key.json"),
    preparation_parameters=Path("inputs/config/example_parameters.json"),
    smiles=df,
)

model = Model(Path("inputs/models/example_cls_model"))

cls_pred, reg_pred = model.predict(prepared_data=prepared_data, classification_tasks=[0, 1, 2])

print("\n cls_pred dataframe : \n")
print(cls_pred.head(10))
