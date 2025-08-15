# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

from abc import ABC, abstractmethod
import numpy as np
import random 
import time
from collections import defaultdict
import os
import ConfigSpace as CS

class OptimizableModel(ABC):
    def __init__(self, 
            Model, 
            base_params,
            name,
            verbose=0,
        ):
        self.Model = Model
        self.base_params = base_params
        self.name = name
        
        self.verbose = verbose
        

    @property
    def dataset_loader(self):
        if not hasattr(self, '_dataset_loader'):
            raise Exception("Trying to access dataset_loader before setting it! Try compiling the optimizable first.")
        else: return self._dataset_loader

    @dataset_loader.setter
    def dataset_loader(self, dataset_loader):
        self._dataset_loader = dataset_loader

    def _get_extra_params(self):
        return {}

    def get_bounds(self):
        if hasattr(self, "_get_bounds"):
            return self._get_bounds()
        else:
            raise NotImplementedError("get_bounds is not implemented!")
        
    
    def get_config_space(self):
        if hasattr(self, "_get_configuration_space"):
            return self._get_configuration_space()
        elif hasattr(self, "_get_bounds"):
            return configspace_from_map(self._get_bounds().values())
        else:
            raise NotImplementedError("get_config_space is not implemented! Please implement _get_configuration_space or _get_bounds in the optimizable model class.")
        
    def compile(self, dataset_loader, cv_idx=None, dataset_name="dataset"):
        self.dataset_loader = dataset_loader
        self.dataset_name = dataset_name
        ## if cv_idx is None, we will use all folds for evaluation
        self.cv_idx = cv_idx
        if self.cv_idx is not None and self.cv_idx >= dataset_loader.n_CV:
            raise ValueError(f"cv_idx {self.cv_idx} is out of bounds for dataset {dataset_name} with {dataset_loader.n_CV} folds.")



    def gen_ran_arr(self, size, seed,  max=1000):
        random.seed(seed)
        arr = np.array([int(random.random() * max) for i in range(size)], dtype=int)
        return arr
    
    @staticmethod
    def _evaluate_model(model, X, y):
        pred = model.predict(X)
        return (
            model.calc_accuracy(y=y, pred=pred), 
            model.calc_f1(y=y, pred=pred), 
            model.calc_precision(y=y, pred=pred), 
            model.calc_recall(y=y, pred=pred)
        )
        
    def evaluate(self, config, seed, n_times, return_models=False, save_path=None, force_train=True, device="cuda"):
        if n_times == 1:
            seeds = [seed]
        else:
            seeds = self.gen_ran_arr(n_times, seed)
        params = self.base_params | self._get_extra_params() | dict(config)
        metrics = []
        if return_models: models = []
        
        if self.cv_idx is None:
            cv_indices = np.arange(self.dataset_loader.n_CV)
        else:
            cv_indices = [self.cv_idx]
        
        train_time = []
        other_metrics = defaultdict(list)

        for rep_idx in range(n_times):
            for cv_idx in cv_indices:
                X_train, y_train, X_val, y_val, X_test, y_test, _, _ = self.dataset_loader.getitem(
                    cv_idx) 
                if save_path is not None and not os.path.exists(os.path.join(save_path, f"cv_{cv_idx}_rep_{rep_idx}.pt")):
                    sp = os.path.join(save_path, f"cv_{cv_idx}_rep_{rep_idx}.pt")
                    print(f"WARNING: Couldn't find a pretrained base model at the specified path {sp}!!")
                if not force_train and save_path is not None and os.path.exists(os.path.join(save_path, f"cv_{cv_idx}_rep_{rep_idx}.pt")):
                    model = self.Model.load_from_checkpoint(os.path.join(save_path, f"cv_{cv_idx}_rep_{rep_idx}.pt"), device=device)
                else:
                    train_time.append(time.time())
                    model = self.Model(seed=seeds[rep_idx], verbose=self.verbose, **params)
                    model.fit(X_train, y_train, X_val, y_val)
                    train_time[-1] = time.time() - train_time[-1]
                    if save_path is not None:
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        print("saving model with n_layers: ", model.n_layers, "n_nodes: ", model.n_nodes)
                        model.save(os.path.join(save_path, f"cv_{cv_idx}_rep_{rep_idx}.pt"))
                
                if return_models: models.append(model)
                train_metrics = OptimizableModel._evaluate_model(model, X_train, y_train)
                val_metrics = OptimizableModel._evaluate_model(model, X_val, y_val)
                test_metrics = OptimizableModel._evaluate_model(model, X_test, y_test)
                metrics.append([*train_metrics, *val_metrics,  *test_metrics])
                if hasattr(model, "measure_flops") and callable(getattr(model, "measure_flops")):
                    other_metrics["flops"].append(model.measure_flops(X_train[:5]))
                else:
                    other_metrics["flops"].append(0)
                extra_metrics = model.evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
                for k, v in extra_metrics.items():
                    other_metrics[k].append(v)
                del model
        return_dict = {"metrics": np.array(metrics), "train_time": train_time, "other_metrics": other_metrics}
        if return_models: return_dict["models"] = models
        return return_dict
    

            
    def evaluate_cost(self, config, seed=42):
        params = self.base_params | self._get_extra_params() | dict(config)

        avg_acc = []
        if self.cv_idx is None:
            cv_indices = np.arange(self.dataset_loader.n_CV)
        else:
            cv_indices = [self.cv_idx]

        for i in cv_indices:

            X_train, y_train, X_val, y_val, _, _, _, _ = self.dataset_loader.getitem(i)
            try:  
                model = self.Model(seed=seed, verbose=self.verbose, **params)
                model.fit(X_train, y_train, X_val, y_val)

                avg_acc.append(model.score(X_val, y_val))
            except Exception as e:
                print(f"Encountered error during evaluating {self.name} on {self.dataset_name} with config: {config}")
                print("params: ", params)
                raise e
            del model
        return 1-np.mean(avg_acc)
    
def configspace_from_map(conf_map, seed=42):
    conf_space = CS.ConfigurationSpace(seed=seed)
    conf_space.add_hyperparameters(conf_map)
    return conf_space