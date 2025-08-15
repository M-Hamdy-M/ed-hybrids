# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

from tuning.bo.core.types.optimizable_model import OptimizableModel
from models.perceptron.mlp import MLPModel

import ConfigSpace as CS
import torch

class OptimizableMLP(OptimizableModel):
    def __init__(self, name="MLP", device=None, verbose=0, **kwargs):
        super().__init__(MLPModel, {
            "device": device if device is not None else "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 500,
            "early_stop": True,
            "gamma": 1,
            "beta": 0,
            "bn_learnable": True,
            "track_running_stats": True,
            "loss": "mse",
            "initialize_linear": False,
            } | kwargs, name=name, verbose=verbose)


    def _get_bounds(self):
        bounds = {
        "n_layers": CS.UniformIntegerHyperparameter("n_layers", lower=1, upper=15, default_value=5),
        "l2_weight": CS.UniformFloatHyperparameter(
            "l2_weight", lower=1e-7, upper=4e3, default_value=1e-2, log=True),
        "learning_rate": CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
        "l1_weight": CS.UniformFloatHyperparameter(
            "l1_weight", lower=1e-7, upper=1e-2, default_value=1e-5, log=True),
        "batch_percentage": CS.UniformFloatHyperparameter("batch_percentage", lower=0.05, upper=0.5, default_value=0.05),
        "output_direct_link": CS.CategoricalHyperparameter("output_direct_link", (True, False), default_value=True), 
        }
        if "p_drop" not in self.base_params:
            bounds["p_drop"] =  CS.UniformFloatHyperparameter("p_drop",lower=0,upper=0.5, default_value=0.05)
        if "use_drop_out" not in self.base_params:
            bounds["use_drop_out"] =  CS.CategoricalHyperparameter(
                "use_drop_out", (True, False), default_value=True)
            
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
