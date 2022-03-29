from argparse import Namespace

import melloddy_tuner.tunercli
import numpy as np
import sparsechem
import torch
from torch.utils.data import DataLoader

STRUCTURE_FILE = "structure_file"
CONFIG_FILE = "config_file"
KEY_FILE = "key_file"
OUTPUT_DIR = "output_dir"
RUN_NAME = "run_name"
NUMBER_CPU = "number_cpu"
REF_HASH = "ref_hash"
NON_INTERACTIVE = "non_interactive"


def build_args(
    structure_file: str,
    config_file: str,
    key_file: str,
    output_dir: str,
    run_name: str,
    number_cpu: int = 1,
    ref_hash: str = "",
    non_interactive: bool = True,
) -> Namespace:
    namespace = {}

    namespace[STRUCTURE_FILE] = structure_file
    namespace[CONFIG_FILE] = config_file
    namespace[KEY_FILE] = key_file
    namespace[OUTPUT_DIR] = output_dir
    namespace[RUN_NAME] = run_name
    namespace[NUMBER_CPU] = number_cpu
    namespace[NON_INTERACTIVE] = non_interactive
    namespace[REF_HASH] = ref_hash if ref_hash else None

    return Namespace(**namespace)


def main():
    args = build_args(
        "/Users/apicosson/Documents/workspace/melloddy/mms/files/T2_100samples.csv",
        "/Users/apicosson/Documents/workspace/melloddy/mms/files/example_parameters.json",
        "/Users/apicosson/Documents/workspace/melloddy/mms/files/example_key.json",
        "/Users/apicosson/Documents/workspace/melloddy/mms/out",
        "test",
        4,
    )

    melloddy_tuner.tunercli.do_prepare_prediction(args)

    results_loaded = sparsechem.load_results(
        "/Users/apicosson/Documents/workspace/melloddy/mms/files/cp1_98_ce324027-b6b3-4c5f-a9d6-3e7468659230_9c85b1f5-8849-499c-8241-459953e38d2c/export/hyperparameters.json",
        two_heads=True,
    )
    conf = results_loaded["conf"]

    data = sparsechem.load_sparse("/Users/apicosson/Documents/workspace/melloddy/mms/out/test/matrices/pred_x.npz")
    data = sparsechem.fold_transform_inputs(data, folding_size=conf.fold_inputs, transform=conf.input_transform)

    device = "cpu"
    net = sparsechem.SparseFFN(conf).to(device)
    state_dict = torch.load(
        "/Users/apicosson/Documents/workspace/melloddy/mms/files/cp1_98_ce324027-b6b3-4c5f-a9d6-3e7468659230_9c85b1f5-8849-499c-8241-459953e38d2c/export/model.pth",
        map_location=torch.device(device),
    )

    y_class = sparsechem.load_check_sparse(None, (data.shape[0], conf.class_output_size))
    y_regr = sparsechem.load_check_sparse(None, (data.shape[0], conf.regr_output_size))

    net.load_state_dict(state_dict)
    dataset_te = sparsechem.ClassRegrSparseDataset(x=data, y_class=y_class, y_regr=y_regr)
    loader_te = DataLoader(
        dataset_te,
        batch_size=4000,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset_te.collate,
    )

    class_out, regr_out = sparsechem.predict(net, loader_te, dev=device, dropout=0, progress=True, y_cat_columns=None)

    print(class_out[:10])

    filename = "out/pred-hidden.npy"
    np.save(filename, class_out)
