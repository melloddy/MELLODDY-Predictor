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
        
        self.has_task_maps = False
        if (class_task_map is not None) or (class_task_map is not None):
            self.set_tasks_maps(class_task_map, regr_task_map)
        

    def set_tasks_maps(self, class_task_map=None, regr_task_map=None):
        """
        Set the task maps stored in the object
        
        Args:
            class_task_map:              a dictionary or pandas series having classification task labels as as key or index, resp, and continuous classification task IDs (column indexes of the prediction matrix) as values
            regr_task_map:               a dictionary or pandas series having regression task labels as as key or index, resp, and continuous regression task IDs (column indexes of the prediction matrix) as values 
        
        """
        class_task_map, regr_task_map = self.__validate_maps__(class_task_map, regr_task_map)
        self.class_task_map = class_task_map
        self.regr_task_map = regr_task_map
        self.has_task_maps = True

        
    def __validate_maps__(self,class_task_map, regr_task_map):
        class_task_map = self.__validate_map__(class_task_map, "classification", self.conf.class_output_size)
        regr_task_map = self.__validate_map__(regr_task_map, "regression", self.conf.regr_output_size)
        #test for non overlap of task labels between classification and regression
        if (class_task_map is not None) and (regr_task_map is not None):
            if class_task_map.index.intersection(regr_task_map.index).size > 0:
                raise ValueError("classification and regression task map have task labels in common")
        return class_task_map, regr_task_map
    
    
    @staticmethod    
    def __validate_map__(task_map, category, category_output_size):
        if task_map is not None:
            if category_output_size == 0:
                raise ValueError("A {0} task map has been provided for a model without {0} tasks".format(category))
            if type(task_map) == pd.Series:
                pass
            elif type(task_map) == dict:
                task_map = pd.Series(task_map)
            else:
                raise TypeError("{0} task_map needs be either of type pandas.Series or be a dictionary, but is a {1}".\
                                 format(category, type(task_map)))
            if not task_map.dtype == int:
                raise TypeError("The {0} task_map needs to have values of type int".format(category))
            if task_map.max() >= category_output_size:
                raise ValueError("The maximum value of {0} task_map exceeps the number of {0} outputs ({1})".\
                                 format(category,category_output_size))
            if not task_map.is_unique:
                raise ValueError("the task indexes in {0} task map are not unique".format(category))
            if not task_map.index.is_unique:
                raise ValueError("the task labels in {0} task map are not unique".format(category))
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

    
    def predict_decorated_series_from_tensor(self, X: torch.Tensor, class_task_map=None, regr_task_map=None) -> pd.Series:
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
            raise ValueError("This function expects only single row tensor, but te tensor passed has size of {0}".format(X.size(0)))

        #if task maps are passed as arguments we use them
        if (class_task_map is not None) or (regr_task_map is not None):
            class_task_map, regr_task_map = self.__validate_maps__(class_task_map, regr_task_map)
        #otherwise we fall back to the tasks maps passed upon intialization, if peresent
        elif self.has_task_maps:
            class_task_map = self.class_task_map
            regr_task_map = self.regr_task_map
        #if those don't exist, the function cannpot proceed
        else:
            raise ValueError("Task maps must be passed either at intialization time of the predictor or when calling the prediction functions")
               
        y_class_array, y_regr_array = self.predict_from_tensor(X)
        
        if class_task_map is not None:
            y_class_series = pd.Series(y_class_array[0,class_task_map.values],index = class_task_map.index)
        else:
            y_class_series = None
        if regr_task_map is not None:
            y_regr_series = pd.Series(y_regr_array[0,regr_task_map.values],index = regr_task_map.index)
        else:
            y_regr_series = None
        #pd.concat drops silently None items, unless all items to concat are None
        return pd.concat([y_class_series, y_regr_series])

    
    def predict_decorated_series_from_csr(self, x_csr: csr_matrix, class_task_map=None, regr_task_map=None) -> pd.Series:
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
        return self.predict_decorated_series_from_tensor(X, class_task_map, regr_task_map)
            

    def predict_hidden_from_tensor(self, X: torch.Tensor) -> np.ndarray:
        """
        This function computes the last hidden layer of the model
        
        Args:
            X (torch.sparse_coo_tensor) : fingerprint features as tporch sparse_coo_tensor
        
        Returns:
            numpy.ndarray of hidden layer values
        """
        
        with torch.no_grad():
            return self.net(X.to(self.device), last_hidden=True).cpu().numpy()            

    
    def predict_hidden_from_csr(self, x_csr):
        """
        This function computes the last hidden layer of the model
        
        Args:
            x_csr (scipy.sparse.csr_matrix) : fingerprint features as csr_matrix
        
        Returns:
            numpy.ndarray of hidden layer values
        """
        X = csr_to_torch_coo(x_csr)
        return self.predict_hidden_from_tensor(X)        
                

        
        


def t8df_to_task_map(t8_df: pd.DataFrame, task_type: str, name_column : str = "input_assay_id", concat_task_tye : bool = False, concat_threshold : bool = True) -> pd.Series:
    """
    This function extracts from a t8 type dataframe (or a selected slice thereof) a task_map for the predictor object
    
    Args:
        t8_df (pandas.DataFrame): dataframe to extarct thge task map from
        task_type (str):          either "classification" or "regression"
        name_column (str) :       column in datafarem to use as task labels
        concat_task_type (bool):  Prepend task_label with the task_type, this can be usefull for hybdrid models where there can be indetically names tasks
        concat_threshold (bool):  If set to true for classification tasks the threshold value will be appended to the task name
    
    """
    if not task_type in ["classification","regression"]:
        raise ValueError("Task type must be either \"classification\" or \"regression\", passed type is {0}".format(task_type)) 
    task_id_column = "cont_{0}_task_id".format(task_type)
    if not task_id_column in t8_df:
        raise ValueError("task index column \"{0}\" is not present in the task dataframe".format(task_id_column))
    if t8_df[task_id_column].isnull().any():
        raise ValueError("Null value task indices are present in data frame")    
    task_ids = t8_df[task_id_column].astype(int)
    if not name_column in t8_df:
        raise ValueError("task name column \"{0}\" is not present in the task dataframe".format(name_column))
    task_labels = t8_df[task_id_column].copy().astype(str)
    if concat_task_tye:
        task_labels = task_type + '_' + task_labels
    if task_type == "classification" and concat_threshold:
        if not "threshold" in t8_df:
            raise ValueError("option \"concat_threshold\" was chosen, but \"threshold\" column not present in dataframe")
        task_labels = task_labels +  "_" + t8_df["threshold"].astype(str)
    if not task_labels.is_unique:
        raise ValueError("task labels are not unique, try to use a diffeent name column, and/or make use of concat_threshold option")
    return pd.Series(task_ids.values, index = task_labels)


            
