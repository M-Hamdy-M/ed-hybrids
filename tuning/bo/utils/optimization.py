# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import os
import pickle as pkl

module_path = os.path.abspath(os.path.dirname(__file__))


def get_hyperparams(dataset_name, model_name, id, cv_idx=None, base_directory=None):
    if base_directory is not None:
        output_loc = os.path.join(base_directory, f"{dataset_name}/{model_name}")
    else:
        output_loc = os.path.join(module_path, f"../hyperparams/{dataset_name}/{model_name}")
               
    if cv_idx is not None:
        output_loc = os.path.join(output_loc, f"cv_idx_{cv_idx}")
    save_path = os.path.join(output_loc, f"{id}.pk")

    
    with open(save_path , "rb") as file :
        best_dict = pkl.load(file)
    return best_dict