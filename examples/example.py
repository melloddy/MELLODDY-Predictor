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

from pathlib import Path

import melloddy_tuner.utils.helper  # type: ignore
import pandas as pd

from melloddy_predictor import Model
from melloddy_predictor import PreparedData

smiles_path = Path("inputs/data/T2_100samples.csv")
df: pd.DataFrame = melloddy_tuner.utils.helper.read_input_file(str(smiles_path))

prepared_data = PreparedData(
    encryption_key=Path("inputs/config/example_key.json"),
    preparation_parameters=Path("inputs/config/example_parameters.json"),
    smiles=df,
)

model = Model(Path("inputs/models/example_cls_model"), load_on_demand=False)

cls_pred, reg_pred = model.predict(prepared_data=prepared_data, classification_tasks=[0, 1, 2])

model.unload()

print("\n cls_pred dataframe : \n")
print(cls_pred.head(10))
