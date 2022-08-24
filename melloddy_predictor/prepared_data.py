# Copyright 2022 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import warnings

import melloddy_tuner.tunercli  # type: ignore
import melloddy_tuner.utils.helper  # type: ignore
import pandas as pd
from scipy.sparse import csr_matrix  # type: ignore


class PreparedData:
    """
    The data prepared by melloddy_tuner

    Args:
        encryption_key (pathlib.Path): Path of the encryption key `json` used to shuffle the bits of the descriptors
            (fingerprints) in `melloddy_tuner`.
            Ex: `inputs/config/example_key.json`.
        preparation_parameters (pathlib.Path): Path of the parameters `json` to be used to prepare the dataset with
            `melloddy_tuner`.
            Ex: `inputs/config/example_parameters.json`.
            More details in `melloddy_tuner`'s `README.md`, Section `# Parameter definitions`.
        smiles (pd.DataFrame): The test data. A loaded T2 structure.
        num_cpu (int): The number of CPUs to use in the data preparation by tuner

    Raises:
        FileNotFoundError: encryption_key not found
        FileNotFoundError: preparation_parameters not found
        Warning: if some SMILES failed to be prepared.
    """

    _device: str
    _tuner_encryption_key: pathlib.Path
    _tuner_configuration_parameters: pathlib.Path
    _data: csr_matrix
    _df_failed: pd.DataFrame
    _compound_ids: pd.DataFrame
    _num_cpu: int

    def __init__(
        self,
        encryption_key: pathlib.Path,
        preparation_parameters: pathlib.Path,
        smiles: pd.DataFrame,
        num_cpu: int,
    ):
        if not os.path.isfile(encryption_key):
            raise FileNotFoundError(encryption_key)
        if not os.path.isfile(preparation_parameters):
            raise FileNotFoundError(preparation_parameters)

        self._tuner_encryption_key = encryption_key
        self._tuner_configuration_parameters = preparation_parameters
        self._num_cpu = num_cpu

        data, df_failed, compound_mapping = melloddy_tuner.tunercli.do_prepare_prediction_online(
            input_structure=smiles,
            key_path=str(self._tuner_encryption_key),
            config_file=str(self._tuner_configuration_parameters),
            num_cpu=self._num_cpu,
        )

        if not df_failed.empty:
            warnings.warn(
                f"""
{len(df_failed)} SMILES failed to be prepared.
These SMILES will be ignored during the prediction.
You can use the `failed_compounds` attribute to get the failing SMILES dataframe.
Preview of failing SMILES:
{df_failed}
                """,
                Warning,
            )

        compound_ids = compound_mapping["input_compound_id"].reset_index().drop("index", axis=1)
        assert compound_ids["input_compound_id"].is_unique

        self._data = data
        self._df_failed = df_failed
        self._compound_ids = compound_ids

    @property
    def data(self) -> csr_matrix:
        """
        Returns:
            csr_matrix: The prepared data / the x_matrix returned by melloddy_tuner
        """
        return self._data

    @property
    def failed_compounds(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: The failed compounds, the smiles which can't be processed.
                The rows are the compounds ids (`input_compound_id` from the `smiles` file),
                 and the column `error_message` contains the error returned by `melloddy_tuner`
        """
        return self._df_failed

    @property
    def compound_ids(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: a single column "input_compound_id" with the compound_ids in the same order as the data
                and future predictions
        """
        return self._compound_ids
