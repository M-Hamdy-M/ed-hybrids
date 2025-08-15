# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import pickle as pkl
import time
from pathlib import Path 
import time
import os
import sys

import numpy as np
import ConfigSpace as CS
from smac import HyperparameterOptimizationFacade, Scenario


# Determine the path to the main directory
module_path = os.path.abspath(os.path.dirname(__file__))
main_directory = os.path.abspath(os.path.join(module_path, '../../../'))
if main_directory not in sys.path:
    sys.path.append(main_directory)

from data.DLoader.uci import UCIDataset

def configspace_from_map(conf_map, seed=42):
    conf_space = CS.ConfigurationSpace(seed=seed)
    conf_space.add_hyperparameters(conf_map)
    return conf_space

def optimize(hp_config, output_loc, opt_function, n_trials=100, walltime_limit=np.inf, verbose=0, callbacks=[]):
    scenario = Scenario(hp_config,
                        output_directory=Path(output_loc),
                        walltime_limit=walltime_limit, 
                        n_trials=n_trials,
                        n_workers=1)

    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario)
    smac = HyperparameterOptimizationFacade(scenario,
                                            opt_function,
                                            initial_design=initial_design,
                                            overwrite=True,
                                            callbacks=callbacks)
    incumbent = smac.optimize()
    incumbent_cost: float | list[float] = smac.validate(incumbent)

    best_dict = {"target": 1-incumbent_cost,
                 "params": dict(incumbent)}
    return best_dict



def tune(
        optimizable, 
        dataset, 
        n_trials,
        ## save location params
        save_id,
        save_base_path=None,
        force_tune=False,
        
        seed=42,
        evaluation_reps=5,
        return_models=False,
        verbose=0,
    ):    
    try:
        loader = UCIDataset(dataset, parent=os.path.join(
                module_path, "../../../data/DLoader/UCIdata_Corrected"))
        optimizable.compile(dataset_loader=loader, dataset_name=dataset)
        if save_base_path is None:
            save_base_path = os.path.join(module_path, "../hyperparams")
        output_loc = os.path.join(save_base_path, f"{dataset}/{optimizable.name}")
        save_path = os.path.join(output_loc, f"{save_id}.pk")
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
        
        if os.path.exists(save_path) and not force_tune:
            with open(save_path , "rb") as file :
                best_dict = pkl.load(file)
        else:
            print(f"Optimizing {optimizable.name} on {dataset}")
            log_path = os.path.join(output_loc, "logs")
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            

            
            start_time = time.time()
            config_space = optimizable.get_config_space()
            best_dict = optimize(config_space,
                                log_path, optimizable.evaluate_cost, n_trials=n_trials, verbose=verbose)
            
            ds_time = time.time() - start_time
            best_dict = {"target": best_dict['target'],
                        "params":  best_dict['params'],
                        "time": ds_time,
                        "trials": n_trials,
                        "name": optimizable.name,
                        "base_params": optimizable.base_params}
            with open(save_path, 'wb') as file:
                pkl.dump(best_dict, file)
        params = optimizable.base_params | best_dict["params"]
        metrics = optimizable.evaluate(params, seed, n_times=evaluation_reps, return_models=return_models)
        return metrics
    except Exception as e:
        print(f"Encountered error while tuning {optimizable.name} on {dataset}")
        raise e