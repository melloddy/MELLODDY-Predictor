from pathlib import Path

from model_manipulation_software import PredictionSystem

predictor = PredictionSystem(
    model_folder=Path("inputs/models"),
    encryption_key=Path("inputs/config/example_key.json"),
    preparation_parameters=Path("inputs/config/example_parameters.json"),
)

cls_pred, reg_pred, failed_smiles = predictor.predict(
    model_name="example_cls_model", smiles=Path("inputs/data/T2_100samples.csv"), classification_tasks=[0, 1, 2]
)

print("\n cls_pred dataframe : \n")
print(cls_pred.head(10))
