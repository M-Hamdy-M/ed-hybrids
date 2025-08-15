# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import torch.nn as nn
import numpy as np
import torch
from models.types.model import ClassificationModel
from models.perceptron.mlp import MLP
from fvcore.nn import FlopCountAnalysis


class MOMLP(ClassificationModel):
    def __init__(self, 
                n_layers,
                n_nodes,
                l1_weight,
                l2_weight,
                epochs,
                learning_rate,
                batch_percentage,
                output_direct_link,
                gamma,
                beta, 
                bn_learnable,
                track_running_stats,
                device,
                seed=None,
                verbose=0,
                loss="ce",
                early_stop=True,

                add_cfs_output=True,
                cfs_l2_weight=0,
                initialize_linear=False,
                use_selu=False,
                p_drop=0,
                use_drop_out=True,
                 ):
        
        super().__init__(seed=seed, verbose=verbose)

        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.output_direct_link = output_direct_link

        self.epochs = epochs
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.batch_percentage = batch_percentage
        if loss.lower() not in ["mse", "ce"]:
            raise Exception("Invalid value for loss. Expected either mse or ce!")
        self.loss = loss
        self.loss_fn = nn.MSELoss() if self.loss == "mse" else nn.CrossEntropyLoss()

        self.l2_weight = l2_weight
        self.l1_weight = l1_weight

        self.gamma = gamma
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats

        self.device = device
        self.initialized = False

        self.add_cfs_output = add_cfs_output
        self.initialize_linear = initialize_linear
        self.use_selu = use_selu
        if use_drop_out:
            self.p_drop = p_drop
        else:
            self.p_drop = 0
        if self.p_drop < 0:
            self.p_drop = 0
        self.cfs_l2_weight = cfs_l2_weight

        
    def initialize(self, X_train, y_train):
        self.n_classes = y_train.shape[1]
        self.initialized = True
        self.output_layers = []
    
    @property
    def cfs_l2_weight(self):
        return self._cfs_l2_weight

    @cfs_l2_weight.setter
    def cfs_l2_weight(self, cfs_l2_weight):
        if type(cfs_l2_weight) is list:
            if len(cfs_l2_weight) != self.n_layers:
                raise ValueError(f"Invalid value for cfs_l2_weight. cfs_l2_weight must be equal to the number of layers. Expected scalar or array of length {self.n_layers} but got array of length {len(cfs_l2_weight)}")
            else:
                self._cfs_l2_weight = cfs_l2_weight
        else:
            self._cfs_l2_weight = [cfs_l2_weight] * self.n_layers

    
    def _fit(self, X_train , y_train, X_val = None , y_val = None):
        if not self.initialized:
            self.initialize(X_train=X_train, y_train=y_train)
        X_train = torch.Tensor(X_train).float().to(self.device)
        y_train = torch.Tensor(y_train).float().to(self.device)
        if not X_val is None : 
            X_val = torch.Tensor(X_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)

        ## Train a normal MLP
        self.model = MLP(
            n_layers=self.n_layers,
            n_nodes=self.n_nodes,
            output_direct_link=self.output_direct_link,
            bn_learnable=self.bn_learnable,
            track_running_stats=self.track_running_stats,
            device=self.device,
            l2_weight=self.l2_weight,
            l1_weight=self.l1_weight,
            cfs_l2_weight=None,
            n_raw_features=None,
            gamma=self.gamma,
            beta=self.beta,
            loss=self.loss,
            initialize_linear=self.initialize_linear,
            use_selu=self.use_selu,
            p_drop=self.p_drop,
        ).to(self.device)
        self.model.initialize(X_train=X_train, y_train=y_train)

        opt = torch.optim.Adam(self.model.parameters() , lr = self.learning_rate)
        self.model.do_bp_training(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.epochs,
            sample_weights=torch.ones(X_train.shape[0]).to(self.device),
            optimizer=opt,
            batch_percentage=self.batch_percentage,
            early_stop=self.early_stop,
        )

        if self.add_cfs_output:
            output_layers, estimator_weights = self.fit_output_layers(X_train, y_train, X_val, y_val)
            
        
    def fit_output_layers(self, X_train, y_train, X_val, y_val):
        output_layers = []
        estimator_weights = np.empty(self.n_layers + 1)
        self.model.eval()
        with torch.no_grad():
            encoding = X_train
            val_encoding = X_val
            
            for idx, layer in enumerate(self.model.hidden):
                encoding = layer(encoding)
                extended_encoding = torch.cat([X_train.clone(), encoding], axis=1)
                n_samples, n_features = extended_encoding.shape
                cfs_layer = nn.Linear(in_features=n_features, out_features=y_train.shape[1]).to(self.device)
                if n_features < n_samples:   # primal space
                    beta = torch.mm(torch.mm(torch.pinverse(torch.eye(extended_encoding.shape[1]).to(
                        self.device) * self.cfs_l2_weight[idx]+torch.mm(extended_encoding.T, extended_encoding)), extended_encoding.T), y_train)
                else:   # dual space
                    beta = torch.mm(extended_encoding.T, torch.mm(torch.pinverse(torch.eye(
                        extended_encoding.shape[0]).to(self.device) * self.cfs_l2_weight[idx]+torch.mm(extended_encoding, extended_encoding.T)), y_train))
                cfs_layer.weight[:] = nn.Parameter(beta.T)
                cfs_layer.bias[:] = nn.Parameter(torch.zeros_like(cfs_layer.bias))

                output_layers.append(cfs_layer)
                
                val_encoding = layer(val_encoding)
                val_extended_encoding = torch.cat([X_val.clone(), val_encoding], axis=1)

                val_pred = cfs_layer(val_extended_encoding)
                estimator_weights[idx] = 1
            
            if self.output_direct_link:
                val_encoding = val_extended_encoding
                
            val_pred = self.model.output(val_encoding)
            estimator_weights[-1] = 1
            self.output_layers = output_layers
            self.estimator_weights = estimator_weights
            return output_layers, estimator_weights
         
    def _predict(self, X):
        n_examples, n_features = X.shape
        self.model.eval()
        X = torch.Tensor(X).float().to(self.device)
        with torch.no_grad():
            if not self.add_cfs_output: return self.model(X).cpu()
            scores = np.empty((self.n_layers + 1, n_examples, self.n_classes))
            encoding = X
            
            for idx, layer in enumerate(self.model.hidden):
                encoding = layer(encoding)
                extended_encoding = torch.cat([X.clone(), encoding], axis=1)
                output = self.output_layers[idx](extended_encoding)
                scores[idx] = output.cpu()
            if self.output_direct_link:
                encoding = extended_encoding
                
            output = self.model.output(encoding)
            scores[-1] = output.cpu()
            scores = scores * self.estimator_weights[:, np.newaxis, np.newaxis]
            return scores.mean(0)



    def measure_flops(self, sample_input):
        wrapper_model = MOMLPWithHeads(self.model, self.output_layers).to(self.device)
        wrapper_model.eval()
        sample_input = torch.Tensor(sample_input).float().to(self.device)
        sample_input = sample_input.unsqueeze(0) if sample_input.ndim == 1 else sample_input
        with torch.no_grad():
            flops = FlopCountAnalysis(wrapper_model, sample_input)
            return flops.total()

    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.add_cfs_output and hasattr(self, "output_layers"):
            total += sum(p.numel() for layer in self.output_layers for p in layer.parameters())
            trainable += sum(p.numel() for layer in self.output_layers for p in layer.parameters() if p.requires_grad)

        return total, trainable


    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        n_layers = self.n_layers
        n_neurons = self.n_nodes
        total_params, trainable_params = self.count_parameters()
        return {
            "n_layers": n_layers,
            "n_neurons": n_neurons,
            "total_params": total_params,
            "trainable_params": trainable_params
        }

        

class MOMLPWithHeads(nn.Module):
    def __init__(self, base_model, output_layers):
        super().__init__()
        self.base_model = base_model
        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, x):
        out = x
        outputs = []

        for i, layer in enumerate(self.base_model.hidden):
            out = layer(out)
            extended = torch.cat([x, out], dim=-1)
            if i < len(self.output_layers):
                pred = self.output_layers[i](extended)
            else:
                pred = x
            outputs.append(pred)

        avg_output = torch.stack(outputs, dim=0).mean(dim=0)
        return avg_output