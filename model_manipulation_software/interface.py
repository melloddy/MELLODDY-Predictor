import pathlib

from model_manipulation_software.prediction_system import PredictionSystem


def main():
    models_path = pathlib.Path("/Users/apicosson/Documents/workspace/melloddy/mms/files/models")
    permutation_key = pathlib.Path("/Users/apicosson/Documents/workspace/melloddy/mms/files/example_key.json")
    preparation_parameter = pathlib.Path(
        "/Users/apicosson/Documents/workspace/melloddy/mms/files/example_parameters.json"
    )
    task_metadata = pathlib.Path("/Users/apicosson/Documents/workspace/melloddy/mms/")
    smiles_path = pathlib.Path("/Users/apicosson/Documents/workspace/melloddy/mms/files/T2_100samples.csv")

    predictor = PredictionSystem(
        model_folder=models_path,
        permutation_key=permutation_key,
        preparation_parameters=preparation_parameter,
        task_metadata=task_metadata,
        reg_processed_metadata="",
        cls_processed_metadata="",
    )

    cls_pred, reg_pred = predictor.predict("cls_mod_1", smiles_path)

    print(cls_pred[:10])


if __name__ == "__main__":
    main()
