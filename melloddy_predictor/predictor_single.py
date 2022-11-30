import sparsechem as sc
import scipy.io
import numpy as np
import types
import pandas as pd
import torch
import sys
import argparse
from scipy.sparse import csr_matrix
from scipy.special import expit
from collections import OrderedDict
from enum import Enum
import warnings


def csr_to_torch_coo(csr_mat: csr_matrix) -> torch.Tensor:
    """
    Converts from scipy sparse csr matrix to a torch sparsae_coo_tensor to be submitted to the sparsechem network
    Sparsewchem requires 
    
    Args:
        csr_matx (scipy.sparse.csr_matrix) sprse csr matrix to convert
        
    Returns:
        torch.sparse_coo_tensor 
    """
    coo_mat = csr_mat.tocoo()
    return torch.sparse_coo_tensor(indices = np.array([coo_mat.row,coo_mat.col]), values = coo_mat.data, size = coo_mat.shape, dtype=torch.float)


class ScModelType(Enum):
    classification = 0
    regression = 1
    hybrid = 2

  
class PredictorSingle:
    """
    This class handles predictions for single instances.
    It bypasses a lot of mechansims for batched data loading
    """

    def __init__(self, model, conf, class_task_map=None, regr_task_map=None, dropout = False, device = "cpu"):
        """
        Inititialze the predictor object
        
        Args:
            model:                       filename of the model pytorch model file
            conf:                        filename of the the corresponsing configuration file for the model
            class_task_map:              a dictionary or pandas series having classification task labels as as key or index, resp, and continuous classification task IDs (column indexes of the prediction matrix) as values
            regr_task_map:               a dictionary or pandas series having regression task labels as as key or index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values 
            dropout(bool):               whether to apply dropout or nor
            device:                      device to run on, per dafault cpu
        
        """
        results_loaded = sc.load_results(conf, two_heads=True)
        self.conf  = results_loaded["conf"]
        self.device = device
        self.net = sc.SparseFFN(self.conf).to(self.device)
        state_dict = torch.load(model, map_location=torch.device(self.device))

        if self.conf.model_type == "federated":
            state_dict_new = OrderedDict()
            state_dict_new["net.0.net_freq.weight"] = state_dict["0.0.net_freq.weight"]
            state_dict_new["net.0.net_freq.bias"]   = state_dict["0.0.net_freq.bias"]
            state_dict_new["net.2.net.2.weight"]    = state_dict["1.net.2.weight"]
            state_dict_new["net.2.net.2.bias"]      = state_dict["1.net.2.bias"]
            state_dict = state_dict_new

        #apply model weights
        self.net.load_state_dict(state_dict)
        #set model into evaluation mode
        self.net.eval()
        #apply dropout, if chosen
        self.dropout = dropout
        if self.dropout:
            self.net.apply(sc.utils.enable_dropout) 
        # if inverse normalization is done load the stats
        self.inverse_normalization = self.conf.inverse_normalization
        if self.inverse_normalization:
            stats = results_loaded["stats"]
            self.reg_mean = np.array(stats["mean"])
            self.reg_var = np.array(stats["var"])
            self.reg_stddev = np.sqrt(self.reg_var)
        
        if self.net.cat_id_size is not None:
            raise NotImplementedError("Predictions for models with a catalog head are not yet implemented")
        if self.net.class_output_size > 0:
            if self.net.regr_output_size > 0:
                self.model_type = ScModelType.hybrid
            else:
                self.model_type = ScModelType.classification
        elif self.net.regr_output_size > 0:
            self.model_type = ScModelType.regression

        self.has_task_maps = False
        if (class_task_map is not None) or (regr_task_map is not None):
            self.set_tasks_maps(class_task_map, regr_task_map)
        

    def set_tasks_maps(self, class_task_map=None, regr_task_map=None):
        """
        Set the task maps stored in the object
        
        Args:
            class_task_map:              a dictionary or pandas series having classification task labels as as key or index, resp, and continuous classification task IDs (column indexes of the prediction matrix) as values
            regr_task_map:               a dictionary or pandas series having regression task labels as as key or index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values 
        
        """
        class_task_map, regr_task_map, mapped_tasks_type = self.__validate_maps__(class_task_map, regr_task_map)
        self.class_task_map = class_task_map
        self.regr_task_map = regr_task_map
        self.mapped_tasks_type = mapped_tasks_type
        self.has_task_maps = True


    def get_mapped_task_names(self):
        if not self.has_task_maps:
            return None
        elif self.mapped_tasks_type == ScModelType.classification:
            return class_task_map.index.values
        elif self.mapped_tasks_type == ScModelType.regression:
            return regr_task_map.index.values
        elif self.mapped_tasks_type == ScModelType.hybrid:
            return self.mapped_task_type_info.index.values

    def get_model_type(self):
        return self.model_type

    def get_mapped_task_type(self):
        if self.has_task_maps:
            return self.mapped_tasks_type
        else:
            return None

    def get_num_class_tasks(self):
        return self.conf.class_output_size

    def get_num_regr_tasks(self):
        return self.conf.regr_output_size

    def get_num_tasks(self):
        return self.conf.class_output_size + self.conf.regr_output_size

    def get_num_tasks_by_type(self, type):
        if type == ScModelType.classification:
            return self.get_num_class_tasks()
        elif type == ScModelType.regression:
            return self.get_num_regr_tasks()
        else:
            raise ValueError("Non permitted task type {}".format(type))

    def __validate_maps__(self,class_task_map, regr_task_map):
        class_task_map = self.__validate_map__(class_task_map, ScModelType.classification)
        regr_task_map = self.__validate_map__(regr_task_map, ScModelType.regression)
        #test for non overlap of task labels between classification and regression
        if class_task_map is not None:
            if regr_task_map is not None:
                mapped_tasks_type = ScModelType.hybrid
            else:
                mapped_tasks_type = ScModelType.classification
        else:
            if regr_task_map is not None:
                mapped_tasks_type = ScModelType.regression
            else:
                raise ValueError("Task maps for both classification and regression are None")
        if mapped_tasks_type == ScModelType.hybrid:
            if class_task_map.index.intersection(regr_task_map.index).size > 0:
                raise ValueError("classification and regression task map have task labels in common, this is not permitted")
        return class_task_map, regr_task_map, mapped_tasks_type
    

    def __validate_map__(self, task_map, task_type):
        if task_map is not None:
            if self.get_num_tasks_by_type(task_type) == 0:
                raise ValueError("A {0} task map has been provided for a model without {0} tasks".format(task_type.name))
            if type(task_map) == pd.Series:
                pass
            elif type(task_map) == dict:
                task_map = pd.Series(task_map)
            else:
                raise TypeError("{0} task_map needs be either of type pandas.Series or be a dictionary, but is a {1}".\
                                 format(task_type.name, type(task_map)))
            if not task_map.dtype == int:
                raise TypeError("The {0} task_map needs to have values of type int".format(task_type.name))
            if task_map.max() >= self.get_num_tasks_by_type(task_type):
                raise ValueError("The maximum value of {0} task_map exceeps the number of {0} outputs ({1})".\
                                 format(task_type.name,self.get_num_tasks_by_type(task_type)))
            if not task_map.is_unique:
                raise ValueError("the task indexes in {0} task map are not unique".format(task_type.name))
            if not task_map.index.is_unique:
                raise ValueError("the task labels in {0} task map are not unique".format(task_type.name))
        return task_map
                                                                 

    def predict_from_csr(self, x_csr: csr_matrix) -> tuple:
        """
        Feed the input csr matrix in on the go to the neural net for prediction, by passing the torch data loader
        This is meant to be used for small batches Returns a dense numpy array for classification and regression tasks
        
        Args:
            x_csr(scipy.sparse.csr_matrix) : a scipy sparse csr matrix with fingerprint features
            
        Returns:
            Tupe(np.array, np.array) with classifcation and regression predictions
        """
        X = csr_to_torch_coo(x_csr)
        y_class_array, y_regr_array = self.predict_from_tensor(X)
        return y_class_array, y_regr_array

    def predict_from_tensor(self, X: torch.Tensor) -> tuple:
        """
        Feed the input torch sparse coo_tensor in on the go to the neural net for prediction, by passing the torch data loader
        This is meant to be used for small batches 
        
        Args:
            X(torch.sparse_coo_tensor) : a torch sparse coo tensor matrix with fingerprint features
            
        Returns:
            Tupe(np.array, np.array) with classifcation and regression predictions
        """
        #don't compute gradients       
        with torch.no_grad():
            if self.net.cat_id_size is None:
                y_class, y_regr = self.net(X.to(self.device))
            else:
                y_class, y_regr, yc_cat = self.net(X.to(self.device))
            y_class_array = torch.sigmoid(y_class).cpu().numpy()
            y_regr_array =  y_regr.cpu().numpy()
            if self.inverse_normalization:
                #y_regr_array  = sc.inverse_normalization(csr_matrix(y_regr_array) , mean=self.reg_mean, \
                #                                         variance=self.reg_var, array=True)
                y_regr_array = y_regr_array * self.reg_stddev + self.reg_mean
        return y_class_array, y_regr_array

    @staticmethod
    def extract_tasks(y_array, task_map):
         return pd.Series(y_array[0,task_map.values],index = task_map.index)
    
    def predict_decorated_series_from_tensor(self, X: torch.Tensor, class_task_map=None, regr_task_map=None, limit_to_type = None) -> pd.Series:
        """
        This runs the prediction on the input tensor expected to have single row and extracts the desired tasks based on the information in the task 
        maps that have been passed either on predictor intitialization, or with the call of this function (having precedence). 
        It extracts from the raw prediction the tasks of interstest as specified through the task map(s) and warps them into a 
        series having the task lables of the task maps index as series index. Predictions for tasks, which index is not listed in the tasks maps(s) 
        are not included in the returned series
        
        Args:
             X(torch.sparse_coo_tensor) : a torch sparse coo tensor matrix with fingerprint features
             class_task_map: a dictionary or pandas series having classification task labels as as key or index, resp, and continuous classification task IDs (column indexes of the prediction matrix) as values
             regr_task_map: a dictionary or pandas series having regression task labels as as key or index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values 
        Returns:
            pd.Series of predictions with task labels as index        
        """
        #This function expects receiving a tensor witrh a single row, as we also only return results for the first row 
        if X.size(0) != 1:
            raise ValueError("This function expects only single row tensor, but the tensor passed has size of {0}".format(X.size(0)))

        #if task maps are passed as arguments we use them
        if (class_task_map is not None) or (regr_task_map is not None):
            class_task_map, regr_task_map, mapped_tasks_type = self.__validate_maps__(class_task_map, regr_task_map)
        #otherwise we fall back to the tasks maps passed upon intialization, if peresent
        elif self.has_task_maps:
            class_task_map = self.class_task_map
            regr_task_map = self.regr_task_map
            mapped_tasks_type = self.mapped_tasks_type
        #if those don't exist, the function cannpot proceed
        else:
            raise ValueError("Task maps must be passed either at intialization time of the predictor or when calling the prediction functions")
               
        y_class_array, y_regr_array = self.predict_from_tensor(X)
        
        if mapped_tasks_type == ScModelType.hybrid and limit_to_type is not None:
            if mapped_type in [ScModelType.classification,ScModelType.regression]:
                mapped_tasks_type = limit_to_type
            else:
                raise ValueError("Not permitted type {0} has been provided for limit_to_type".format(limit_to_type))

        if mapped_tasks_type == ScModelType.classification:
             results = self.extract_tasks(y_class_array, class_task_map)
        elif mapped_tasks_type == ScModelType.regression:
             results = self.extract_tasks(y_regr_array, regr_task_map)
        elif mapped_tasks_type == ScModelType.hybrid:
             results = pd.concat([self.extract_tasks(y_class_array, class_task_map),self.extract_tasks(y_regr_array, regr_task_map)])

        return results

    
    def predict_decorated_series_from_csr(self, x_csr: csr_matrix, class_task_map=None, regr_task_map=None, limit_to_type = None) -> pd.Series:
        """
        This runs the prediction on the input csr_matrix expected to have single row and extracts the desired tasks based on the information in the task 
        maps that have been passed either on predictor intitialization, or with the call of this function (having precedence). 
        It extracts from the raw prediction the tasks of interstest as specified through the task map(s) and warps them into a 
        series having the task lables of the task maps index as series index. Predictions for tasks, which index is not listed in the tasks maps(s) 
        are not included in the returned series
        
        Args:
             X(torch.sparse_coo_tensor) : a torch sparse coo tensor matrix with fingerprint features
             class_task_map: a dictionary or pandas series having classification task labels as as key or index, resp, and continuous classification task IDs (column indexes of the prediction matrix) as values
             regr_task_map: a dictionary or pandas series having regression task labels as as key or index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values 
        Returns:
            pd.Series of predictions with task labels as index        
        """
        X = csr_to_torch_coo(x_csr)
        return self.predict_decorated_series_from_tensor(X, class_task_map, regr_task_map, limit_to_type = limit_to_type)
            

    def predict_last_hidden_from_tensor(self, X: torch.Tensor) -> np.ndarray:
        """
        This function computes the last hidden layer of the model
        
        Args:
            X (torch.sparse_coo_tensor) : fingerprint features as tporch sparse_coo_tensor
        
        Returns:
            numpy.ndarray of hidden layer values
        """
        
        with torch.no_grad():
            return self.net(X.to(self.device), last_hidden=True).cpu().numpy()

    
    def predict_last_hidden_from_csr(self, x_csr):
        """
        This function computes the last hidden layer of the model
        
        Args:
            x_csr (scipy.sparse.csr_matrix) : fingerprint features as csr_matrix
        
        Returns:
            numpy.ndarray of hidden layer values
        """
        X = csr_to_torch_coo(x_csr)
        return self.predict_hidden_from_tensor(X)        
                

    def predict_trunk_from_tensor(self, X: torch.Tensor) -> np.ndarray:
        """
        This function computes the last hidden layer of the model
        
        Args:
            X (torch.sparse_coo_tensor) : fingerprint features as tporch sparse_coo_tensor
        
        Returns:
            numpy.ndarray of hidden layer values
        """
        
        with torch.no_grad():
            return self.net(X.to(self.device), trunk_embeddings=True).cpu().numpy()

    
    def predict_trunk_from_csr(self, x_csr):
        """
        This function computes the last hidden layer of the model
        
        Args:
            x_csr (scipy.sparse.csr_matrix) : fingerprint features as csr_matrix
        
        Returns:
            numpy.ndarray of hidden layer values
        """
        X = csr_to_torch_coo(x_csr)
        return self.predict_hidden_from_tensor(X)
        
        


def t8df_to_task_map(t8_df: pd.DataFrame, task_type: str, name_column : str = "input_assay_id", concat_task_tye : bool = False, threshold_multi_ix = False, concat_threshold : bool = True) -> pd.Series:
    """
    This function extracts from a t8 type dataframe (or a selected slice thereof) a task_map for the predictor object
    
    Args:
        t8_df (pandas.DataFrame): dataframe to extarct thge task map from
        task_type (str):          either "classification" or "regression"
        name_column (str) :       column in datafarem to use as task labels
        concat_task_type (bool):  Prepend task_label with the task_type, this can be usefull for hybdrid models where there can be indetically names tasks
        concat_threshold (bool):  If set to true for classification tasks the threshold value will be appended to the task name
    
    """
    temp_df = t8_df.copy()
    if not task_type in ["classification","regression"]:
        raise ValueError("Task type must be either \"classification\" or \"regression\", passed type is {0}".format(task_type)) 
    task_id_column = "cont_{0}_task_id".format(task_type)
    if not task_id_column in temp_df:
        raise ValueError("task index column \"{0}\" is not present in the task dataframe".format(task_id_column))
    if temp_df[task_id_column].isnull().any():
        raise ValueError("Null value task indices are present in data frame")    
    task_ids = temp_df[task_id_column].astype(int)
    if not name_column in temp_df:
        raise ValueError("task name column \"{0}\" is not present in the task dataframe".format(name_column))
    temp_df["task_labels"] = temp_df[task_id_column].astype(str)
    if concat_task_tye:
        temp_df["task_labels"] = task_type + '_' + temp_df["task_labels"]
    if task_type == "classification" and (concat_threshold or threshold_multi_ix):
        if not "threshold" in temp_df:
            raise ValueError("option \"concat_threshold\" was chosen, but \"threshold\" column not present in dataframe")
        if concat_threshold:
            temp_df["task_labels"] = temp_df["task_labels"] +  "_" + temp_df["threshold"].astype(str)
        elif threshold_multi_ix:
            temp_df["threshold"] = temp_df["threshold"].astype(float)
    if task_type == "classification" and threshold_multi_ix:
        temp_df = temp_df.set_index(["task_labels","threshold"])
    else:
        temp_df = temp_df.set_index("task_labels")
    if not temp_df.index.is_unique:
        raise ValueError("task labels are not unique, try to use a diffeent name column, and/or make use of concat_threshold or  option")
    return tempd_df[task_id_column]



