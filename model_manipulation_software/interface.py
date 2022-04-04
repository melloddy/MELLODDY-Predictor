import pathlib

from model_manipulation_software.prediction_system import PredictionSystem


def main():

    models_path = pathlib.Path("inputs/models")
    encryption_key = pathlib.Path("inputs/config/example_key.json")
    preparation_parameter = pathlib.Path("inputs/config/example_parameters.json")
    smiles_path = pathlib.Path("inputs/data/T2_100samples.csv")

    predictor = PredictionSystem(
        model_folder=models_path,
        encryption_key=encryption_key,
        preparation_parameters=preparation_parameter,
    )

    cls_pred, reg_pred = predictor.predict("example_hybrid_model", smiles_path)

    print(cls_pred[:10])


if __name__ == "__main__":
    main()
