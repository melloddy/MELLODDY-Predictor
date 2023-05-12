import os
import pathlib

import numpy as np
import pandas as pd
import pytest
from melloddy_tuner.utils.single_row_prep2pred import KeyProviderFromJsonFile
from melloddy_tuner.utils.single_row_prep2pred import SingleRowPreparator
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
from scipy.sparse import load_npz

from melloddy_predictor.predictor_single import PredictorSingle
from melloddy_predictor.predictor_single import ScModelType
from melloddy_predictor.predictor_single import t8df_to_task_map

TEST_FILE_DIR = os.path.dirname(__file__)
MODELS_PATH = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/models")
ENCRYPTION_KEY = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_key.json")
PREPARATION_PARAMETER = pathlib.Path(f"{TEST_FILE_DIR}/../inputs/config/example_parameters.json")


@pytest.fixture
def input_smiles_df():
    return pd.read_csv(os.path.join(TEST_FILE_DIR, "begin_to_end_test/T2_100samples.csv"))


@pytest.fixture
def ref_output_xdata():
    return load_npz(os.path.join(TEST_FILE_DIR, "begin_to_end_test/mt_output/pred_x.npz"))


@pytest.fixture
def ref_row_mapping_table():
    return pd.read_csv(os.path.join(TEST_FILE_DIR, "begin_to_end_test/mt_output/mapping_table.csv"))


@pytest.fixture
def kprovider():
    return KeyProviderFromJsonFile(ENCRYPTION_KEY)


@pytest.fixture
def srprep(kprovider):
    return SingleRowPreparator(key_provider=kprovider, params=PREPARATION_PARAMETER)


@pytest.fixture
def ref_output_ydata():
    y_refs_class = {
        "cls": np.load(os.path.join(TEST_FILE_DIR, "begin_to_end_test/sc_output/cls_model-class.npy")),
        "clsaux": np.load(os.path.join(TEST_FILE_DIR, "begin_to_end_test/sc_output/clsaux_model-class.npy")),
        "hyb": np.load(os.path.join(TEST_FILE_DIR, "begin_to_end_test/sc_output/hyb_model-class.npy")),
    }
    y_refs_regr = {
        "reg": np.load(os.path.join(TEST_FILE_DIR, "begin_to_end_test/sc_output/reg_model-regr.npy")),
        "hyb": np.load(os.path.join(TEST_FILE_DIR, "begin_to_end_test/sc_output/hyb_model-regr.npy")),
    }
    return {"class": y_refs_class, "regr": y_refs_regr}


@pytest.fixture
def ref_output_trunk():
    return {
        mtype: np.load(os.path.join(TEST_FILE_DIR, "begin_to_end_test/sc_output/trunk_{}.npy".format(mtype)))
        for mtype in ["cls", "clsaux", "reg", "hyb"]
    }


@pytest.fixture
def class_task_map():
    return {"class_570": 570, "class_581": 581, "class_2276": 2276}


@pytest.fixture
def regr_task_map():
    return {"regr_633": 633, "regr_740": 740, "regr_2": 2}


@pytest.fixture
def ref_name_arrays(class_task_map, regr_task_map):
    return {
        "cls": pd.Series(class_task_map).index.values,
        "clsaux": pd.Series(class_task_map).index.values,
        "reg": pd.Series(regr_task_map).index.values,
        "hyb": np.concatenate([pd.Series(class_task_map).index.values, pd.Series(regr_task_map).index.values]),
    }


@pytest.fixture
def ref_model_types():
    return {
        "cls": ScModelType.CLASSIFICATION,
        "clsaux": ScModelType.CLASSIFICATION,
        "reg": ScModelType.REGRESSION,
        "hyb": ScModelType.HYBRID,
    }


@pytest.fixture
def test_preds(class_task_map, regr_task_map):
    return {
        "cls": PredictorSingle(
            model=os.path.join(MODELS_PATH, "example_cls_model/model.pth"),
            conf=os.path.join(MODELS_PATH, "example_cls_model/hyperparameters.json"),
            class_task_map=class_task_map,
        ),
        "clsaux": PredictorSingle(
            model=os.path.join(MODELS_PATH, "example_clsaux_model/model.pth"),
            conf=os.path.join(MODELS_PATH, "example_clsaux_model/hyperparameters.json"),
            class_task_map=class_task_map,
        ),
        "reg": PredictorSingle(
            model=os.path.join(MODELS_PATH, "example_reg_model/model.pth"),
            conf=os.path.join(MODELS_PATH, "example_reg_model/hyperparameters.json"),
            regr_task_map=regr_task_map,
        ),
        "hyb": PredictorSingle(
            model=os.path.join(MODELS_PATH, "example_hyb_model/model.pth"),
            conf=os.path.join(MODELS_PATH, "example_hyb_model/hyperparameters.json"),
            class_task_map=class_task_map,
            regr_task_map=regr_task_map,
        ),
    }


@pytest.fixture
def ref_hyb_res_slice_df():
    return pd.read_csv(
        os.path.join(TEST_FILE_DIR, "begin_to_end_test/selected_hyb_tasks.csv"), index_col="input_compound_id"
    )


@pytest.fixture
def input_failing_smiles_df():
    return pd.read_csv(os.path.join(TEST_FILE_DIR, "begin_to_end_test/T2_100samples_failing.csv"))


@pytest.fixture
def ix_rename_map(ref_row_mapping_table):
    return ref_row_mapping_table.set_index("cont_descriptor_vector_id")["input_compound_id"]


@pytest.fixture
def get_benzene_x_csr(srprep):
    return srprep.descriptor_calc.calculate_single_csr("c1ccccc1")


@pytest.fixture
def get_benzene_y_ref():
    return {
        "cls": pd.Series({"class_570": 0.516933, "class_581": 0.433307, "class_2276": 0.565609}, dtype="float32"),
        "clsaux": pd.Series({"class_570": 0.412029, "class_581": 0.489868, "class_2276": 0.504993}, dtype="float32"),
        "reg": pd.Series({"regr_633": 5.097863, "regr_740": 5.743073, "regr_2": 7.306094}, dtype="float64"),
        "hyb": pd.Series(
            {
                "class_570": 0.821179,
                "class_581": 0.209964,
                "class_2276": 0.560037,
                "regr_633": 5.118069,
                "regr_740": 5.721944,
                "regr_2": 7.383655,
            },
            dtype="float64",
        ),
    }


@pytest.fixture
def cls_t8df_head():
    int_cols = [
        "cont_classification_task_id",
        "classification_task_id",
        "num_total_actives",
        "num_fold_min_actives",
        "num_total_inactives",
        "num_fold_min_inactives",
        "n_tasks",
        "retained_tasks",
    ]
    T8c = pd.read_csv(os.path.join(MODELS_PATH, "example_cls_model/T8c.csv"))
    T8c[int_cols] = T8c[int_cols].astype("Int64")
    return T8c[T8c["cont_classification_task_id"] < 10]


@pytest.fixture
def test_pred_multi_ix(cls_t8df_head):
    multi_ix_task_map = t8df_to_task_map(cls_t8df_head, task_type="classification", threshold_multi_ix=True)
    return PredictorSingle(
        model=os.path.join(MODELS_PATH, "example_cls_model/model.pth"),
        conf=os.path.join(MODELS_PATH, "example_cls_model/hyperparameters.json"),
        class_task_map=multi_ix_task_map,
    )


def test_dense_tasks_prediction(srprep, input_smiles_df, ref_output_xdata, ref_output_ydata, ix_rename_map, test_preds):
    # generate x-data
    x_tensors_test = {
        k: srprep.process_smiles(smi) for k, smi in input_smiles_df.set_index("input_compound_id")["smiles"].items()
    }
    x_dataframe_test = (
        pd.concat({ix: pd.Series(v.to_dense().numpy()[0]) for ix, v in x_tensors_test.items()}).unstack().astype("int8")
    )
    x_dataframe_ref = pd.DataFrame(ref_output_xdata.todense()).rename(index=ix_rename_map)
    # assert equivalence of X-data
    assert_frame_equal(x_dataframe_test.sort_index(), x_dataframe_ref.sort_index())
    # generate y-data
    res = {}
    res["class"] = {}
    res["regr"] = {}
    for mtype, my_pred in test_preds.items():
        for ix, x_tens in x_tensors_test.items():
            y_class_array, y_regr_array = my_pred.predict_from_tensor(x_tens)
            if y_class_array.shape[1] > 0:
                if mtype not in res["class"]:
                    res["class"][mtype] = {}
                res["class"][mtype][ix] = y_class_array
            if y_regr_array.shape[1] > 0:
                if mtype not in res["regr"]:
                    res["regr"][mtype] = {}
                res["regr"][mtype][ix] = y_regr_array
    #
    test_output_ydf = {
        ttype: {
            mtype: pd.concat({i: pd.Series(j[0]) for i, j in y_arrays.items()}).unstack()
            for mtype, y_arrays in ydata.items()
        }
        for ttype, ydata in res.items()
    }
    # prepare referency y dataframes
    ref_output_ydf = {
        ttype: {mtype: pd.DataFrame(y).rename(index=ix_rename_map) for mtype, y in y_refs.items()}
        for ttype, y_refs in ref_output_ydata.items()
    }
    # test bequivalence for classification
    for mtype, y_dataframe_ref in ref_output_ydf["class"].items():
        assert_frame_equal(test_output_ydf["class"][mtype].sort_index(), y_dataframe_ref.sort_index())
    # test fro regression deactivates as inverso normalization notr yet prersent for dense prediction
    # for mtype, y_dataframe_ref in ref_output_ydf["regr"].items():
    #    assert_frame_equal(test_output_ydf["regr"][mtype], y_dataframe_ref)


def test_named_task_predictions(
    srprep,
    input_smiles_df,
    test_preds,
    class_task_map,
    regr_task_map,
    ref_output_ydata,
    ref_hyb_res_slice_df,
    ix_rename_map,
):
    y_res_slice = {}
    for k, smi in input_smiles_df.set_index("input_compound_id")["smiles"].items():
        x = srprep.process_smiles(smi)
        y = test_preds["hyb"].predict_decorated_series_from_tensor(x)
        y_res_slice[k] = y
    test_hyb_res_slice_df = pd.concat(y_res_slice).unstack()
    test_hyb_res_slice_df.index.names = ["input_compound_id"]
    assert_frame_equal(test_hyb_res_slice_df, ref_hyb_res_slice_df)

    # manually inverse normalize the reference
    # this should not be necessary, once https://github.com/melloddy/SparseChem/issues/5 has been resolved
    ref_output_ydata = ref_output_ydata.copy()
    ref_output_ydata["regr"] = {
        mtype: y * test_preds[mtype].reg_stddev + test_preds[mtype].reg_mean
        for mtype, y in ref_output_ydata["regr"].items()
    }

    y_refs_selected_class_tasks = ref_output_ydata["class"]["hyb"][:, np.array(list(class_task_map.values()))]
    y_refs_selected_regr_tasks = ref_output_ydata["regr"]["hyb"][:, np.array(list(regr_task_map.values()))]
    y_refs_select_class_df = pd.DataFrame(y_refs_selected_class_tasks, columns=list(class_task_map.keys())).rename(
        index=ix_rename_map
    )
    y_refs_select_regr_df = pd.DataFrame(y_refs_selected_regr_tasks, columns=list(regr_task_map.keys())).rename(
        index=ix_rename_map
    )
    ref_hyb_res_slice_df_reconstructed = pd.concat([y_refs_select_class_df, y_refs_select_regr_df], axis=1)
    ref_hyb_res_slice_df_reconstructed.index.names = ["input_compound_id"]
    assert_frame_equal(
        test_hyb_res_slice_df.sort_index().astype("float32"),
        ref_hyb_res_slice_df_reconstructed.sort_index().astype("float32"),
    )


def test_failing_predictions(srprep, input_failing_smiles_df, test_preds):
    with pytest.raises(ValueError):
        y_res_slice = {}
        for k, smi in input_failing_smiles_df.set_index("input_compound_id")["smiles"].items():
            x = srprep.process_smiles(smi)
            y = test_preds["hyb"].predict_decorated_series_from_tensor(x)
            y_res_slice[k] = y


def test_get_mapped_task_names(test_preds, ref_name_arrays):
    for mtype, my_pred in test_preds.items():
        assert (my_pred.get_mapped_task_names() == ref_name_arrays[mtype]).all()


def test_get_model_type(test_preds, ref_model_types):
    for mtype, my_pred in test_preds.items():
        assert my_pred.get_model_type() == ref_model_types[mtype]


def test_limit_to_type(srprep, test_preds):
    x = srprep.process_smiles("c1ccccc1")
    # provoke failure with invalid type
    with pytest.raises(ValueError):
        y = test_preds["hyb"].predict_decorated_series_from_tensor(x, limit_to_type=5)
    # now test a valid type
    y = test_preds["hyb"].predict_decorated_series_from_tensor(x, limit_to_type=ScModelType.REGRESSION)
    y_ref = pd.Series({"regr_633": 5.118069, "regr_740": 5.721944, "regr_2": 7.383655})
    assert_series_equal(y, y_ref)


def test_csr_predictions(get_benzene_x_csr, get_benzene_y_ref, test_preds):
    for mtype, my_pred in test_preds.items():
        y_test = my_pred.predict_decorated_series_from_csr(get_benzene_x_csr)
        assert_series_equal(y_test, get_benzene_y_ref[mtype])


def test_trunk_output(test_preds, srprep, input_smiles_df, ref_output_trunk):
    for mtype, my_pred in test_preds.items():
        assert np.allclose(
            np.concatenate(
                [
                    my_pred.predict_trunk_from_tensor(srprep.process_smiles(smi))
                    for k, smi in input_smiles_df.set_index("input_compound_id")["smiles"].items()
                ]
            ),
            ref_output_trunk[mtype],
            rtol=1e-3,
        )


def test_task_map_generator(cls_t8df_head):
    task_map_test1 = t8df_to_task_map(cls_t8df_head, task_type="classification")
    labels = {
        "assay_517_class_7.00": 0,
        "assay_924_class_6.50": 1,
        "assay_924_class_7.00": 2,
        "assay_924_class_7.50": 3,
        "assay_1160_class_6.50": 4,
        "assay_1160_class_7.00": 5,
        "assay_1512_class_7.50": 6,
        "assay_1512_class_8.00": 7,
        "assay_1512_class_8.50": 8,
        "assay_1520_class_8.00": 9,
    }
    task_map_ref1 = pd.Series(labels, name="cont_classification_task_id", dtype="int64").rename_axis("task_labels")
    assert_series_equal(task_map_test1, task_map_ref1)

    task_map_test2 = t8df_to_task_map(cls_t8df_head, task_type="classification", threshold_multi_ix=True)
    labels2 = {
        "assay_517_class": {7.0: 0},
        "assay_924_class": {6.5: 1, 7.0: 2, 7.5: 3},
        "assay_1160_class": {6.5: 4, 7.0: 5},
        "assay_1512_class": {7.5: 6, 8.0: 7, 8.5: 8},
        "assay_1520_class": {8.0: 9},
    }
    task_map_ref2 = pd.concat(
        {key: pd.Series(val, name="cont_classification_task_id", dtype="int64") for key, val in labels2.items()}
    ).rename_axis(["task_labels", "threshold"])
    assert_series_equal(task_map_test2, task_map_ref2)


def test_multi_ix_predictions(srprep, test_pred_multi_ix):
    x = srprep.process_smiles("c1ccccc1")
    y_multi_ix_test = test_pred_multi_ix.predict_decorated_series_from_tensor(x)
    values_multi_ix = {
        "assay_517_class": {7.0: 0.531071},
        "assay_924_class": {6.5: 0.583757, 7.0: 0.542668, 7.5: 0.474523},
        "assay_1160_class": {6.5: 0.530777, 7.0: 0.428757},
        "assay_1512_class": {7.5: 0.472368, 8.0: 0.367206, 8.5: 0.306637},
        "assay_1520_class": {8.0: 0.499579},
    }
    y_multi_ix_ref = pd.concat(
        {key: pd.Series(val, dtype="float32") for key, val in values_multi_ix.items()}
    ).rename_axis(["task_labels", "threshold"])
    assert_series_equal(y_multi_ix_test, y_multi_ix_ref)
