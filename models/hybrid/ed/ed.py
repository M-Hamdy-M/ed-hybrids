# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import numpy as np
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from collections import defaultdict
import time

from models.hybrid.training_methods import TrainingMethods
from models.perceptron import MLP
from models.utils.early_stoppers import EarlyStopping
import copy
from models.types.model import ClassificationModel
from fvcore.nn import FlopCountAnalysis

class Ed(ClassificationModel):
    def __init__(self,
                n_nodes,
                n_layers,

                l1_weight,
                l2_weight,
                cfs_l2_weight,

                warm_up_rounds=["bp", "bp", "cfs", "cfs"],

                epochs: int=500,
                learning_rate=1e-3,
                batch_percentage=0.05,
                loss="mse",

                gamma=1,
                beta=0,
                bn_learnable=True,
                track_running_stats=True,

                input_direct_link=False,
                epochs_early_stop=True,
                dynamic_rounds=True,
                min_rounds=4,

                sampling_method="exp_loss",
                force_train_method=None,

                device: str = 'cpu',
                seed=None,
                verbose=0,
                initialize_linear=False,
                eval_config=True,
                p_drop=0,
                use_selu=False,
                boost=False,
                boost_lr=False,
                bootstrap=False,
                prune_percentage=None,
                n_prune_nodes=None,
                 ):

        super().__init__(seed=seed, verbose=verbose)
        self.n_nodes = int(n_nodes)
        self.n_layers = int(n_layers)

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.cfs_l2_weight = cfs_l2_weight

        self.gamma = gamma
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats

        self.epochs = epochs
        self.lr = learning_rate
        self.batch_percentage = batch_percentage
        self.epochs_early_stop = epochs_early_stop
        self.loss = loss
        self.dynamic_rounds = dynamic_rounds
        self.min_rounds = min_rounds
        self.input_direct_link = input_direct_link 

        self.initialize_linear = initialize_linear
        self.eval_config = eval_config
        self.p_drop = p_drop
        self.use_selu = use_selu
        self.boost = boost
        self.boost_lr = boost_lr
        self.bootstrap = bootstrap

        self.prune_percentage = prune_percentage
        self.n_prune_nodes = n_prune_nodes

        self.device = device
        self.initialized = False

        self.validate_params(force_train_method, sampling_method, warm_up_rounds)

    def validate_params(self, force_train_method, sampling_method, warm_up_rounds):
        if force_train_method and (sampling_method or warm_up_rounds):
            print("WARNING: sampling_method and warm_up_rounds must both be None if force_train_method is specified. Their values will be ignored!")
            self.sampling_method = None
            self.warm_up_rounds = None
            self.force_train_method = force_train_method
        elif sampling_method and not warm_up_rounds:
            raise ValueError("ERROR: if sampling_method is specified, warm_up_rounds should be a non-empty list!")
        elif not force_train_method and not sampling_method: 
            raise ValueError("ERROR: either force_train_method or sampling_method has to be specified!")
        
        self.do_exp = False
        self.sampling_method = sampling_method
        if self.sampling_method is not None:
            if self.sampling_method == "exp_acc":
                self.use_loss = False
                self.do_exp = True
            elif self.sampling_method == "loss":
                self.use_loss = True
            elif self.sampling_method == "acc":
                self.use_loss = False
            else:
                self.use_loss = True
                self.do_exp = True
                if self.sampling_method != "exp_loss":
                    print("WARNING: Invalid value for sampling method using exp_loss")

        self.force_train_method = force_train_method
        self.warm_up_rounds = warm_up_rounds

    def initialize(self, X_train, y_train):
        self.classes = y_train.shape[1]
        self.params = EasyDict()
        self.params['models'] = []
        self.params["sampling_matrix"] = np.empty((0, 3), dtype=object)
        self.params['estimator_weights'] = []
        self.history = defaultdict(list)
        self.initialized = True
    

    def softmax(self, arr):
        return torch.tensor(arr, dtype=torch.float64, device=self.device).softmax(0)

    def _fit(self, X_train, y_train, X_val, y_val):
        try:
            wc_start_time = time.time()
            if not self.initialized:
                self.initialize(X_train=X_train, y_train=y_train)

            X_train = torch.Tensor(X_train).float().to(self.device)
            y_train = torch.Tensor(y_train).float().to(self.device)
            X_val = torch.Tensor(X_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)

            train_sample_weight = torch.ones(X_train.shape[0]).to(self.device) / X_train.shape[0]
            val_sample_weight = torch.ones(X_val.shape[0]).to(self.device) / X_val.shape[0]

            bootstrap_indices = self.generate_bootstrap_indices(X_train.shape[0], train_sample_weight)

            early_stopper = EarlyStopping(
                patience=4, delta=0, checkpoint_name=None, is_minimization=False)
            
            train_encoding = X_train.clone().to(self.device)
            val_encoding = X_val.clone().to(self.device)
            train_sample_weight = torch.ones(X_train.shape[0]).to(self.device)
            ensemble_preds_per_layer = []

            for layer_idx in range(self.n_layers):
                if self.force_train_method:
                    training_method = TrainingMethods[self.force_train_method]
                elif layer_idx < len(self.warm_up_rounds):
                    training_method = TrainingMethods[self.warm_up_rounds[layer_idx].upper()]
                else:
                    tmp = self.params.sampling_matrix
                    values = tmp[:, 1].astype(np.float64)
                    bp_mean = np.mean(
                        values[tmp[:, 0] == TrainingMethods.BP])
                    cfs_mean = np.mean(
                        values[tmp[:, 0] == TrainingMethods.CFS])
                    averages = [bp_mean, cfs_mean]

                    if self.do_exp:
                        normalized_averages = self.softmax(
                            [bp_mean, cfs_mean]).cpu().numpy()
                    else:
                        total_probability = sum(averages)
                        normalized_averages = [
                            (prob / total_probability) for prob in averages]

                    training_method = np.random.choice(
                        [TrainingMethods.BP, TrainingMethods.CFS], p=normalized_averages)

                ###########################################################################################################################################################
                # train using selected method
                loss_fn = nn.MSELoss()
                model = MLP(
                    n_layers=1, 
                    n_nodes=self.n_nodes, 
                    output_direct_link=True,
                    gamma=self.gamma, 
                    beta=self.beta,
                    bn_learnable=self.bn_learnable,
                    track_running_stats=self.track_running_stats,
                    device=self.device, 
                    l1_weight=self.l1_weight,
                    l2_weight=self.l2_weight,
                    cfs_l2_weight=self.cfs_l2_weight,
                    n_raw_features=X_train.shape[1],
                    loss=self.loss if training_method == TrainingMethods.BP else "mse",
                    initialize_linear=self.initialize_linear,
                    eval_config=self.eval_config,
                    p_drop=self.p_drop if training_method == TrainingMethods.BP else 0,
                    use_selu=self.use_selu,
                ).to(self.device)
                model.initialize(X_train=train_encoding, y_train=y_train)

                if training_method == TrainingMethods.BP:
                    opt = Adam(
                        model.parameters(), lr=self.lr)

                    train_time = model.do_bp_training(
                        X_train=train_encoding[bootstrap_indices], 
                        y_train=y_train[bootstrap_indices], 
                        X_raw=X_train[bootstrap_indices], 
                        X_val=val_encoding, 
                        y_val=y_val, 
                        X_val_raw=X_val, 

                        epochs=self.epochs,
                        sample_weights=train_sample_weight[bootstrap_indices], 
                        optimizer=opt, 
                        batch_percentage=self.batch_percentage, 
                        early_stop=self.epochs_early_stop, 
                    )

                elif training_method == TrainingMethods.CFS:
                    train_time = model.do_cfs_training(
                        X_train=train_encoding, 
                        y_train=y_train, 
                        X_raw=X_train, 
                        X_val=val_encoding, 
                        y_val=y_val, 
                        X_val_raw=X_val
                    )
                if self.n_prune_nodes or self.prune_percentage:
                    model.prune_neurons(
                        X_train=train_encoding, 
                        y_train=y_train, 
                        X_raw=X_train, 
                        X_val=val_encoding, 
                        y_val=y_val, 
                        X_val_raw=X_val,
                        n_nodes=self.n_prune_nodes, 
                        percentage=self.prune_percentage, )
                ###########################################################################################################################################################
                # evaluation
                t_pred, train_loss, train_acc = self.__eval__(
                    model, train_encoding, X_train, y_train, loss_fn)
                v_pred, val_loss, val_acc = self.__eval__(
                    model, val_encoding, X_val, y_val, loss_fn)                           

                ensemble_preds_per_layer.append(v_pred.detach().clone())
                # Ensemble prediction using current layers
                

                self.register_history(train_loss=train_loss,
                                        train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, train_time=train_time, layer_idx=layer_idx)
                
                ## update sampling matrix based on performance
                if self.sampling_method:
                    if self.use_loss:
                        self.params.sampling_matrix = np.vstack([self.params.sampling_matrix, [
                                                                [training_method, 1 / val_loss.cpu().numpy(), 1]]])
                    else:
                        self.params.sampling_matrix = np.vstack(
                            [self.params.sampling_matrix, [[training_method, val_acc * 100, 1]]])

                
                train_encoding = model.transform(train_encoding).detach().clone()
                val_encoding = model.transform(val_encoding)

                if self.input_direct_link:
                    train_encoding = torch.cat((train_encoding, X_train), dim=1)
                    val_encoding = torch.cat((val_encoding, X_val), dim=1)

                self.params.models.append(model)

                if self.boost : 
                    train_sample_weight , train_estimator_weight = self.get_weights(X_train, y_train, train_sample_weight, self.boost)
                    if X_val is not None:
                        val_sample_weight , val_estimator_weight = self.get_weights(X_val, y_val, val_sample_weight, self.boost)
                        self.params.estimator_weights.append(val_estimator_weight)
                    else : 
                        self.params.estimator_weights.append(train_estimator_weight)
                else : 
                    self.params.estimator_weights.append(1)



                stacked_preds = torch.stack(ensemble_preds_per_layer, dim=1)  # shape: [batch_size, n_layers_so_far, n_classes]
                weights = torch.tensor(self.params.estimator_weights, device=self.device)[:layer_idx + 1].reshape(1, -1, 1)
                weighted_preds = (stacked_preds * weights).sum(dim=1) / weights.sum()

                ensemble_acc = accuracy_score(y_val.argmax(dim=1).cpu(), weighted_preds.argmax(dim=1).cpu())
                ensemble_loss = loss_fn(weighted_preds, y_val).mean().item()

                # Save to history
                if 'ensemble_acc' not in self.history:
                    self.history['ensemble_acc'] = []
                if 'ensemble_loss' not in self.history:
                    self.history['ensemble_loss'] = []

                self.history['ensemble_acc'].append(ensemble_acc)
                self.history['ensemble_loss'].append(ensemble_loss)

                
                
                
                if self.dynamic_rounds and layer_idx >= self.min_rounds:
                    early_stopper(val_acc)

                    if early_stopper.early_stop:
                        break

                if self.bootstrap :
                    bootstrap_indices = self.generate_bootstrap_indices(X_train.shape[0], train_sample_weight)
                        
            if self.dynamic_rounds and (early_stopper.early_stop or early_stopper.counter != 0):
                # remove last few layers since no signifcant improvement is opserved
                self.params.models = self.params.models[:-
                                                        early_stopper.counter]
                self.params.sampling_matrix[-early_stopper.counter:, 2] = 0
                self.params.estimator_weights = self.params.estimator_weights[:-
                                                        early_stopper.counter]
            self.wallclock_time = time.time() - wc_start_time
            del X_train, y_train, X_val, y_val
            torch.cuda.empty_cache()
            return self.wallclock_time
        except Exception as e:
            print("Error Training SS")
            print("seed: ", self.seed)
            raise e


    def generate_bootstrap_indices(self, size, weights):
        if self.bootstrap :
           return  torch.Tensor(list(WeightedRandomSampler(weights, size, replacement=True))).type(torch.long)
        else :
            return torch.arange(size)
        

    def get_weights(self, X , y_true, sample_weight, eps=1e-6):
        with torch.no_grad():
            scores = self.get_lw_scores(X)
            y_pred = scores[:, -1, :].argmax(1)
            y_true = y_true.argmax(1)
            incorrect_mask = y_pred != y_true
            error_rate = torch.sum(incorrect_mask) / len(incorrect_mask)
            estimator_weight = torch.log((1 - (error_rate + eps)) / (error_rate + eps)).to(self.device) + torch.log(torch.tensor(self.classes -1)).to(self.device) 
            if self.boost : 
                sample_weight *= torch.exp(estimator_weight * incorrect_mask * self.boost_lr)
                sample_weight /= torch.sum(sample_weight)

        return sample_weight, estimator_weight
    

    def register_history(self, train_loss, train_acc, val_loss, val_acc, train_time, layer_idx):
        # Store Stats
        if layer_idx == len(self.history["train_loss"]):
            self.history['train_loss'].append([train_loss.item()])
            self.history['train_acc'].append([train_acc])
            self.history['val_loss'].append([val_loss.item()])
            self.history['val_acc'].append([val_acc])
            self.history['train_time'].append([train_time])
        elif layer_idx < len(self.history["train_loss"]):
            self.history['train_loss'][layer_idx].append(train_loss.item())
            self.history['train_acc'][layer_idx].append(train_acc)
            self.history['val_loss'][layer_idx].append(val_loss.item())
            self.history['val_acc'][layer_idx].append(val_acc)
            self.history['train_time'][layer_idx].append(train_time)
        else:
            raise Exception(f"Invalid value for layer_idx, got {layer_idx}")

    def __eval__(self, model, encoding, X, y, loss_fn):

        model.eval()
        with torch.no_grad():
            pred = model(encoding, X)

            loss = loss_fn(pred, y).mean()
            pred_np = pred.cpu().detach().numpy()
            acc = accuracy_score(y.argmax(1).cpu(), pred_np.argmax(1))

        return pred, loss, acc

    def _predict(self, X):
        with torch.no_grad():
            scores = self.get_lw_scores(X)
            estimator_weights = torch.tensor(self.params.estimator_weights, device=scores.device).reshape(1, -1, 1)
            scores *= estimator_weights
            return scores.mean(dim=1).cpu()

    def get_lw_scores(self, X):
        with torch.no_grad():
            # Ensure X is a PyTorch tensor and move it to the correct device
            X = X.to(self.device) if isinstance(X, torch.Tensor) else torch.tensor(X).float().to(self.device)
            
            # Initialize the scores tensor with the correct shape
            scores = torch.empty(
                (X.shape[0], len(self.params.models), self.classes), device=self.device)
            
            # Clone the input tensor for encoding
            encoding = X.clone().to(self.device)
            
            for i, model in enumerate(self.params.models):
                model.eval()
                pred_score = model(encoding, X)
                scores[:, i, :] = pred_score.cpu()  # Move predictions to CPU if necessary
                
                # Transform encoding using the model's transform method
                encoding = model.transform(encoding)
                
                if self.input_direct_link:
                    # Concatenate the original input X to the encoding
                    encoding = torch.cat((encoding, X), dim=1)
            
        return scores
    
    def measure_flops(self, sample_input):
        # if not self.params.models:
        #     raise ValueError("No trained models found. Run `_fit` before measuring FLOPs.")

        # sample_input = torch.tensor(sample_input).float().to(self.device)
        # # if sample_input.ndim == 1:
        # #     sample_input = sample_input.unsqueeze(0)

        # model_wrapper = MOMLPWithHeads(
        #     model=self,
        #     base_models=self.params.models,
        #     input_dim=sample_input.shape[1],
        #     input_direct_link=self.input_direct_link
        # ).to(self.device)

        # model_wrapper.eval()
        # with torch.no_grad():
        #     flops = FlopCountAnalysis(model_wrapper, sample_input)
        #     return flops.total()
        return 0
        
        
    def count_parameters(self):
        # if not self.params.models:
        #     raise ValueError("No trained models found. Run `_fit` before counting parameters.")
        # total = sum(p.numel() for m in self.params.models for p in m.parameters())
        # trainable = sum(p.numel() for m in self.params.models for p in m.parameters() if p.requires_grad)
        # return total, trainable
        return 0, 0


    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        sm = self.params.sampling_matrix
        used_methods = sm[:, 0][sm[:, 2] == 1]

        if self.force_train_method is None:
            bp_count = np.sum(used_methods == TrainingMethods.BP)
            cfs_count = np.sum(used_methods == TrainingMethods.CFS)
            total_count = len(used_methods)
            bp_total_ratio = bp_count / total_count if total_count > 0 else 0
            bp_cfs_ratio = bp_count / cfs_count if cfs_count > 0 else float('inf') if bp_count > 0 else None
        
        else:
            bp_total_ratio = None
            bp_cfs_ratio = None
            
        warmup_len = len(self.warm_up_rounds) if self.warm_up_rounds else 0
        used_methods_after_wu = used_methods[warmup_len:]
    
        bp_count_after_wu = np.sum(used_methods_after_wu == TrainingMethods.BP)
        cfs_count_after_wu = np.sum(used_methods_after_wu == TrainingMethods.CFS)
        total_after_wu = len(used_methods_after_wu)
        
        if cfs_count_after_wu == 0:
            bf_cfs_ratio_after_wu = float('inf') if bp_count_after_wu > 0 else None
        else:
            bf_cfs_ratio_after_wu = bp_count_after_wu / cfs_count_after_wu
            
        bp_total_ratio_wu = bp_count_after_wu / total_after_wu if total_after_wu > 0 else 0

        n_layers = len(self.params.models)
        n_neurons = self.n_nodes
        total_params, trainable_params = self.count_parameters()
        return {
            "bf_cfs_ratio": bp_cfs_ratio,
            "bf_cfs_ratio_after_wu": bf_cfs_ratio_after_wu,
            "bp_total_ratio": bp_total_ratio,
            "bp_total_ratio_wu": bp_total_ratio_wu,
            "n_layers": n_layers,
            "n_neurons": n_neurons,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


class MOMLPWithHeads(nn.Module):
    def __init__(self, model, base_models, input_dim, input_direct_link=False):
        super().__init__()
        self.model = model
        self.base_models = nn.ModuleList(base_models)
        self.input_direct_link = input_direct_link
        self.input_dim = input_dim

    def forward(self, x):
        with torch.no_grad():
            out = self.model.predict(x)
        return out