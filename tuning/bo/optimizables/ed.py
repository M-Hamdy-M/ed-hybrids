# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

from tuning.bo.core.types.optimizable_model import OptimizableModel
from models.hybrid import Ed

import ConfigSpace as CS
import torch

class OptimizableEdRVFL(OptimizableModel):
    def __init__(self, name="EdRVFL", device=None, verbose=0, **kwargs):
        bp = {
            "device": device if device is not None else "cuda" if torch.cuda.is_available() else "cpu",
            "dynamic_rounds": "n_layers" in kwargs,
            "input_direct_link": False,
            "warm_up_rounds": None,
            "min_rounds": 4,
            "sampling_method": None,
            "force_train_method": "CFS",
            "bn_learnable": True,

            "l1_weight": None,
            "l2_weight": None,
            "epochs": None,
            "learning_rate": None,
            "batch_percentage": None,
            "prune_percentage": 0,
        }
        if "prune_percentage" not in kwargs:
            del bp["prune_percentage"]
        super().__init__(Ed, bp | kwargs, name=name, verbose=verbose)
        


    def _get_bounds(self):
        bounds = {
            "track_running_stats": CS.CategoricalHyperparameter(
                "track_running_stats", (True, False), default_value=True),

            "gamma": CS.UniformFloatHyperparameter(
                "gamma", lower=0.5, upper=2, default_value=0.5),
            "beta": CS.UniformFloatHyperparameter(
                "beta", lower=-2, upper=2, default_value=0.5),
            "cfs_l2_weight": CS.UniformFloatHyperparameter("cfs_l2_weight", lower=1e-7, upper=4e3, default_value=1e-2, log=True),
            "n_layers": CS.UniformIntegerHyperparameter("n_layers", lower=1, upper=15, default_value=5),
            "prune_percentage": CS.UniformFloatHyperparameter("prune_percentage",lower=0,upper=1, default_value=0.2),
        }
        if "n_layers" in self.base_params:
            del bounds["n_layers"]
            
        if "prune_percentage" in self.base_params:
            del bounds["prune_percentage"]
            
        X_train, _, _, _, _, _, _, _ = self.dataset_loader.getitem(0)
        if X_train.shape[0] <= 10_000:
            bounds |= {
                "n_nodes": CS.UniformIntegerHyperparameter(
                    "n_nodes", lower=256, upper=1024, default_value=256),
            }
        else:
            bounds |= {
                "n_nodes": CS.UniformIntegerHyperparameter(
                    "n_nodes", lower=1024, upper=4069, default_value=256),
            }
        return bounds
    
class OptimizableSS(OptimizableModel):
    def __init__(self, name="SS", device=None, verbose=0, **kwargs):
        super().__init__(Ed, {
            "device": device if device is not None else "cuda" if torch.cuda.is_available() else "cpu",
            "n_layers": 20,
            "warm_up_rounds":["bp", "bp", "cfs", "cfs"],
            "epochs": 500,
            "input_direct_link": False,
            "epochs_early_stop": True,
            "dynamic_rounds": True,
            "loss": "mse",
            "sampling_method": "exp_acc",
            "min_rounds": 4,
            } | kwargs, name=name, verbose=verbose)


    def _get_bounds(self):
        bounds = {
            "l2_weight": CS.UniformFloatHyperparameter(
                "l2_weight", lower=1e-7, upper=4e3, default_value=1e-2, log=True),
            "cfs_l2_weight": CS.UniformFloatHyperparameter("cfs_l2_weight", lower=1e-7, upper=4e3, default_value=1e-2, log=True),
            "l1_weight": CS.UniformFloatHyperparameter(
                "l1_weight", lower=1e-7, upper=1e-2, default_value=1e-5, log=True),
            "learning_rate": CS.UniformFloatHyperparameter(
                "learning_rate", lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
            "batch_percentage": CS.UniformFloatHyperparameter("batch_percentage", lower=0.05, upper=0.5, default_value=0.05),

            "gamma": CS.UniformFloatHyperparameter(
                "gamma", lower=0.5, upper=2, default_value=0.5),
            "beta": CS.UniformFloatHyperparameter(
                "beta", lower=-2, upper=2, default_value=0.5),
            "bn_learnable": CS.CategoricalHyperparameter("bn_learnable", (True, False), default_value=True),
            "track_running_stats": CS.CategoricalHyperparameter(
                "track_running_stats", (True, False), default_value=True),
        }
        if "p_drop" not in self.base_params:
            bounds["p_drop"] =  CS.UniformFloatHyperparameter("p_drop",lower=0,upper=0.5, default_value=0.05)
        X_train, _, _, _, _, _, _, _ = self.dataset_loader.getitem(0)
        if X_train.shape[0] <= 10_000:
            bounds |= {
                "n_nodes": CS.UniformIntegerHyperparameter(
                    "n_nodes", lower=256, upper=1024, default_value=256),
            }
        else:
            bounds |= {
                "n_nodes": CS.UniformIntegerHyperparameter(
                    "n_nodes", lower=1024, upper=4069, default_value=256),
            }
        return bounds

class OptimizableSS(OptimizableModel):
    def __init__(self, name="SS", device=None, verbose=0, **kwargs):
        super().__init__(Ed, {
            "device": device if device is not None else "cuda" if torch.cuda.is_available() else "cpu",
            "n_layers": 20,
            "warm_up_rounds":["bp", "bp", "cfs", "cfs"],
            "epochs": 500,
            "input_direct_link": False,
            "epochs_early_stop": True,
            "dynamic_rounds": True,
            "loss": "mse",
            "min_rounds": 4,
            } | kwargs, name=name, verbose=verbose)


    def _get_bounds(self):
        bounds = {
            "l2_weight": CS.UniformFloatHyperparameter(
                "l2_weight", lower=1e-7, upper=4e3, default_value=1e-2, log=True),
            "cfs_l2_weight": CS.UniformFloatHyperparameter("cfs_l2_weight", lower=1e-7, upper=4e3, default_value=1e-2, log=True),
            "l1_weight": CS.UniformFloatHyperparameter(
                "l1_weight", lower=1e-7, upper=1e-2, default_value=1e-5, log=True),
            "learning_rate": CS.UniformFloatHyperparameter(
                "learning_rate", lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
            "batch_percentage": CS.UniformFloatHyperparameter("batch_percentage", lower=0.05, upper=0.5, default_value=0.05),

            "gamma": CS.UniformFloatHyperparameter(
                "gamma", lower=0.5, upper=2, default_value=0.5),
            "beta": CS.UniformFloatHyperparameter(
                "beta", lower=-2, upper=2, default_value=0.5),
            "bn_learnable": CS.CategoricalHyperparameter("bn_learnable", (True, False), default_value=True),
            "track_running_stats": CS.CategoricalHyperparameter(
                "track_running_stats", (True, False), default_value=True),
        }
        if "p_drop" not in self.base_params:
            bounds["p_drop"] =  CS.UniformFloatHyperparameter("p_drop",lower=0,upper=0.5, default_value=0.05)
        if "sampling_method" not in self.base_params:
            bounds["sampling_method"] =  CS.CategoricalHyperparameter("sampling_method", ("exp_loss", "exp_acc", "loss", "acc"), default_value=True),
            
        X_train, _, _, _, _, _, _, _ = self.dataset_loader.getitem(0)
        if X_train.shape[0] <= 10_000:
            bounds |= {
                "n_nodes": CS.UniformIntegerHyperparameter(
                    "n_nodes", lower=256, upper=1024, default_value=256),
            }
        else:
            bounds |= {
                "n_nodes": CS.UniformIntegerHyperparameter(
                    "n_nodes", lower=1024, upper=4069, default_value=256),
            }
        return bounds
