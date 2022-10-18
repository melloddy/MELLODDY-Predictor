import os
import pathlib

import pandas as pd
import numpy as np

import pytest


from melloddy_tuner.utils.single_row_prep2pred import SingleRowPreparator
from  melloddy_predictor.predictor_single import PredictorSingle



from pandas._testing import assert_frame_equal
from scipy.sparse import save_npz, load_npz

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
    return pd.read_csv(os.path.join(TEST_FILE_DIR,"begin_to_end_test/mt_output/mapping_table.csv"))

@pytest.fixture
def srprep():
    return SingleRowPreparator(secret = ENCRYPTION_KEY, params = PREPARATION_PARAMETER)

@pytest.fixture
def ref_output_ydata():
    y_refs_class = {"cls":np.load(os.path.join(TEST_FILE_DIR,"begin_to_end_test/sc_output/cls_model-class.npy")),\
                    "clsaux": np.load(os.path.join(TEST_FILE_DIR,"begin_to_end_test/sc_output/clsaux_model-class.npy")),\
                    "hyb": np.load(os.path.join(TEST_FILE_DIR,"begin_to_end_test/sc_output/hyb_model-class.npy"))}
    y_refs_regr = {"reg": np.load(os.path.join(TEST_FILE_DIR,"begin_to_end_test/sc_output/reg_model-regr.npy")),\
                    "hyb": np.load(os.path.join(TEST_FILE_DIR,"begin_to_end_test/sc_output/hyb_model-regr.npy"))}
    return {"class" : y_refs_class, "regr" : y_refs_regr}

@pytest.fixture
def class_task_map():
    return {'class_570':570,'class_581':581,'class_2276':2276}

@pytest.fixture
def regr_task_map():
    return {"regr_633":633,"regr_740":740,"regr_2":2}

@pytest.fixture
def test_preds(class_task_map, regr_task_map):
    return {'cls' : PredictorSingle(model= os.path.join(MODELS_PATH,"example_cls_model/model.pth"), conf=os.path.join(MODELS_PATH,"example_cls_model/hyperparameters.json"), class_task_map = class_task_map),\
           'clsaux' : PredictorSingle(model= os.path.join(MODELS_PATH,"example_clsaux_model/model.pth"), conf=os.path.join(MODELS_PATH,"example_clsaux_model/hyperparameters.json"), class_task_map = class_task_map),\
           'reg' : PredictorSingle(model= os.path.join(MODELS_PATH,"example_reg_model/model.pth"), conf=os.path.join(MODELS_PATH,"example_reg_model/hyperparameters.json"), regr_task_map = regr_task_map),\
           'hyb' : PredictorSingle(model= os.path.join(MODELS_PATH,"example_hyb_model/model.pth"), conf=os.path.join(MODELS_PATH,"example_hyb_model/hyperparameters.json"), class_task_map = class_task_map, regr_task_map = regr_task_map)
          }

@pytest.fixture
def ref_hyb_res_slice_df():
    return pd.read_csv(os.path.join(TEST_FILE_DIR,"begin_to_end_test/selected_hyb_tasks.csv"),index_col = "input_compound_id")

@pytest.fixture
def input_failing_smiles_df():
    return pd.read_csv(os.path.join(TEST_FILE_DIR, "begin_to_end_test/T2_100samples_failing.csv"))

@pytest.fixture
def ix_rename_map(ref_row_mapping_table):
    return ref_row_mapping_table.set_index("cont_descriptor_vector_id")["input_compound_id"]


def test_dense_tasks_prediction(srprep, input_smiles_df, ref_output_xdata, ref_output_ydata, ix_rename_map, test_preds):
    #generate x-data
    x_tensors_test = {k:srprep.process_smiles(smi) for k,smi in input_smiles_df.set_index("input_compound_id")["smiles"].items()}
    x_dataframe_test = pd.concat({ix:pd.Series(v.to_dense().numpy()[0]) for ix,v in x_tensors_test.items()}).unstack().astype("int8")
    x_dataframe_ref = pd.DataFrame(ref_output_xdata.todense()).rename(index=ix_rename_map)
    #assert equivalence of X-data
    assert_frame_equal(x_dataframe_test.sort_index(), x_dataframe_ref.sort_index())
    #generate y-data
    res = {}
    res["class"] = {}
    res["regr"] = {}    
    for mtype, my_pred in test_preds.items():
        for ix, x_tens in x_tensors_test.items():
            y_class_array, y_regr_array = my_pred.predict_from_tensor(x_tens)
            if y_class_array.shape[1] > 0:
                if not mtype in res["class"]:
                    res["class"][mtype] = {}
                res["class"][mtype][ix] = y_class_array
            if y_regr_array.shape[1] > 0:
                if not mtype in res["regr"]:
                    res["regr"][mtype] = {}
                res["regr"][mtype][ix] = y_regr_array
    #
    test_output_ydf = {ttype : {mtype : pd.concat({i: pd.Series(j[0]) for i,j in y_arrays.items()}).unstack() for mtype, y_arrays in ydata.items()} for ttype, ydata in res.items()}
    #prepare referency y dataframes
    ref_output_ydf = {ttype : {mtype : pd.DataFrame(y).rename(index=ix_rename_map) for mtype, y in y_refs.items()} for ttype, y_refs in ref_output_ydata.items()}
    #test bequivalence for classification
    for mtype, y_dataframe_ref in ref_output_ydf["class"].items():
        assert_frame_equal(test_output_ydf["class"][mtype].sort_index(), y_dataframe_ref.sort_index())
    #test fro regression deactivates as inverso normalization notr yet prersent for dense prediction
    #for mtype, y_dataframe_ref in ref_output_ydf["regr"].items():
    #    assert_frame_equal(test_output_ydf["regr"][mtype], y_dataframe_ref)

def test_named_task_predictions(srprep, input_smiles_df, test_preds, class_task_map, regr_task_map, ref_output_ydata, ref_hyb_res_slice_df, ix_rename_map):
    y_res_slice = {}
    for k,smi in input_smiles_df.set_index("input_compound_id")["smiles"].items():
        x = srprep.process_smiles(smi)
        y = test_preds["hyb"].predict_decorated_series_from_tensor(x)
        y_res_slice[k] = y
    test_hyb_res_slice_df = pd.concat(y_res_slice).unstack()
    test_hyb_res_slice_df.index.names = ['input_compound_id']
    assert_frame_equal(test_hyb_res_slice_df, ref_hyb_res_slice_df)

    #manually inverse normalize the reference 
    #this should not be necessary, once https://github.com/melloddy/SparseChem/issues/5 has been resolved
    ref_output_ydata = ref_output_ydata.copy()
    ref_output_ydata["regr"]= {mtype : y*test_preds[mtype].reg_stddev + test_preds[mtype].reg_mean for mtype,y in ref_output_ydata["regr"].items()}
    
    y_refs_selected_class_tasks = ref_output_ydata["class"]["hyb"][:,np.array(list(class_task_map.values()))]
    y_refs_selected_regr_tasks = ref_output_ydata["regr"]["hyb"][:,np.array(list(regr_task_map.values()))]
    y_refs_select_class_df = pd.DataFrame(y_refs_selected_class_tasks, columns = list(class_task_map.keys())).rename(index=ix_rename_map)
    y_refs_select_regr_df = pd.DataFrame(y_refs_selected_regr_tasks, columns = list(regr_task_map.keys())).rename(index=ix_rename_map)
    ref_hyb_res_slice_df_reconstructed = pd.concat([y_refs_select_class_df, y_refs_select_regr_df],axis=1)
    ref_hyb_res_slice_df_reconstructed.index.names = ['input_compound_id']
    assert_frame_equal(test_hyb_res_slice_df.sort_index().astype("float32"), ref_hyb_res_slice_df_reconstructed.sort_index().astype("float32"))

def test_failing_predictions(srprep, input_failing_smiles_df, test_preds):
    with pytest.raises(ValueError):
        y_res_slice = {}
        for k,smi in input_failing_smiles_df.set_index("input_compound_id")["smiles"].items():
            x = srprep.process_smiles(smi)
            y = test_preds["hyb"].predict_decorated_series_from_tensor(x)
            y_res_slice[k] = y
