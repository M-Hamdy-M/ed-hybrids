# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 07
# Description: MLP model
# -----------------------------------
import time
from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from models.utils.early_stoppers import EarlyStopping
from models.types.model import ClassificationModel
from fvcore.nn import FlopCountAnalysis

## SNN implementation is influenced from [https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_MLP_MNIST.ipynb]
class MLP(nn.Module):
    def __init__(self,  
                n_layers,
                n_nodes,
                bn_learnable,
                track_running_stats,
                device,
                output_direct_link=True,
                l2_weight=0,
                l1_weight=0,
                cfs_l2_weight=None,
                n_raw_features=None,
                gamma=1,
                beta=0, 
                loss="mse",
                initialize_linear=False,
                eval_config=True,
                use_selu=False,
                p_drop=0,
                 ):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.output_direct_link = output_direct_link
        self.n_raw_features = n_raw_features

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        if cfs_l2_weight is None:
            cfs_l2_weight = l2_weight
        self.cfs_l2_weight = cfs_l2_weight

        if loss.lower() not in ["mse", "ce"]:
            raise Exception("Invalid value for loss. Expected either mse or ce!")
        self.loss = loss
        self.loss_fn = nn.MSELoss() if self.loss == "mse" else nn.CrossEntropyLoss()


        self.gamma = gamma
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats
        

        self.device = device
        self.initialized = False
        self.initialize_linear = initialize_linear
        self.eval_config = eval_config
        self.use_selu = use_selu
        if p_drop < 0:
            p_drop = 0
        self.p_drop = p_drop

    def __init_weights__(self, m):
        if isinstance(m, nn.Linear) and self.use_selu:
            nn.init.constant_(m.bias, 0)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        elif isinstance(m, nn.Linear) and self.initialize_linear:
            m.weight.data.uniform_(-1, 1)
            m.bias.data.uniform_(0, 1)
        # if bn is not learnable, set the values for gama and beta manually and freez their leanring
        if isinstance(m, nn.BatchNorm1d) and (m.weight is not None) and (m.bias is not None):
            if self.bn_learnable:
                m.weight.data.fill_(self.gamma)
                m.bias.data.fill_(self.beta)

            else:
                m.weight.data.fill_(self.gamma)
                m.bias.data.fill_(self.beta)
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def initialize(self, X_train, y_train):
        self.n_classes = y_train.shape[1]
        n_features = X_train.shape[1]

        if self.n_raw_features is None:
            self.n_raw_features = n_features

        self.initialize_hidden_layers(n_features, self.n_nodes)
        # append the output layer
        self.initialize_output_layer(self.n_nodes, self.n_classes, self.output_direct_link)

        self.history = defaultdict(list)
        self.initialized = True

    def initialize_hidden_layers(self, n_features, n_nodes):
        self.hidden = []
        for layer_idx in range(self.n_layers):   
            if self.use_selu:
                layer = [
                    nn.Linear(n_nodes if layer_idx != 0 else n_features, n_nodes),
                    nn.SELU(),
                    nn.AlphaDropout(p=self.p_drop),
                ]
            else:
                layer = [
                    nn.Linear(n_nodes if layer_idx != 0 else n_features, n_nodes),
                    nn.ReLU(),
                ]
                if self.p_drop:
                    layer.append(nn.Dropout(p=self.p_drop))
                layer.append(nn.BatchNorm1d(n_nodes, affine=self.bn_learnable, momentum=0.1, track_running_stats=self.track_running_stats))
            

            self.hidden.append(nn.Sequential(*layer))
        self.hidden = nn.Sequential(*self.hidden).to(self.device)
        self.hidden.apply(self.__init_weights__)

    def initialize_output_layer(self, n_nodes, n_classes, output_direct_link):
        if output_direct_link:
            self.output = nn.Linear(n_nodes + self.n_raw_features, n_classes).to(self.device)
        else:
            self.output = nn.Linear(n_nodes, n_classes).to(self.device)
            
        if self.loss == "ce":
            self.output = nn.Sequential(self.output, nn.Softmax(1)).to(self.device)
        elif self.loss != "mse":
            raise ValueError("Invalid value for loss. expected either mse or ce but got", self.loss)
        

    def forward(self, X, X_raw=None):
        if X_raw is None:
            X_raw = X.clone()

        encoding = self.hidden(X)
        if self.output_direct_link:
            merged = torch.cat([X_raw, encoding], axis=1)
            probability = self.output(merged)
        else:
            probability = self.output(encoding)
        return probability

    
    def transform(self, X):
        encoding = self.hidden(X)
        return encoding
    
    def set_bn_momentum(self, momentum):
        
        for layer_idx in range(len(self.hidden)):
            # print("setting: ", self.hidden[bn_layer_idx])
            self.hidden[layer_idx][2].momentum = momentum

    def do_bp_training(self, X_train, y_train, X_val, y_val, epochs, sample_weights, optimizer, batch_percentage=1, early_stop=True, X_raw=None, X_val_raw=None):
        t = time.time()
        if not self.initialized:
            self.initialize(X_train=X_train, y_train=y_train)

        if early_stop:
            early_stopper = EarlyStopping(patience=10, delta=1e-5, model=self, save_location="ram")
        for epoch in range(epochs):
            self.train()
            # set momentum in the first epoch
            if epoch == 0:
                self.set_bn_momentum(momentum=0.2)

            n_samples = X_train.shape[0]
            batch_size = int(np.ceil(batch_percentage * n_samples))

            indexs = torch.randperm(n_samples)
            for batch_id in range(0, n_samples, batch_size):
                batch_idx = indexs[batch_id: batch_id + batch_size]
                if len(batch_idx) <= 1:
                    break

                train_pred = self(X_train[batch_idx], X_raw[batch_idx] if X_raw is not None else None)
                loss = self.loss_fn(train_pred, y_train[batch_idx])
                loss = (loss * sample_weights[batch_idx]).mean()
                # compute L1 loss
                l1_parameters = []
                for parameter in self.parameters():
                    l1_parameters.append(parameter.abs().reshape(-1))
                l1 = self.l1_weight * torch.cat(l1_parameters).sum()
                loss += l1
                
                # compute L2 loss
                l2_parameters = []
                for parameter in self.parameters():
                    l2_parameters.append(parameter.reshape(-1))
                l2 = self.l2_weight * torch.square(torch.cat(l2_parameters)).sum()
                loss += l2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        
            history_entity = self.__eval__(X=X_train, y=y_train, X_val=X_val, y_val=y_val, train_method="bp", X_raw=X_raw, X_val_raw=X_val_raw)
            for k in history_entity:
                self.history[k].append(history_entity[k])
           
            if early_stop:
                try:
                    with torch.no_grad():
                        self.eval()
                        val_pred = self(X_val, X_val_raw)
                        loss = self.loss_fn(val_pred, y_val).mean()
                        early_stopper(loss, self)
                        if early_stopper.early_stop:
                            early_stopper.load_checkpoint(self)
                            break
                except Exception as e:
                    print(e)
                    raise e    
                
        te = time.time() - t
        self.history["bp_time"].append(te)  
        return te
    
    def do_cfs_training(self, X_train, y_train, X_raw=None, X_val=None, y_val=None, X_val_raw=None, return_details=False):
        t = time.time()
        if not self.initialized:
            self.initialize(X_train=X_train, y_train=y_train)

        with torch.no_grad():
            self.set_bn_momentum(momentum=None)
            if self.eval_config:
                self.eval()
            try:
                n_samples, n_features = X_train.shape
                encoding = self.transform(X_train)
                if self.output_direct_link:
                    encoding = torch.cat([X_raw, encoding], axis=1)


                if n_features < n_samples:   # primal space
                    beta = torch.mm(torch.mm(torch.pinverse(torch.eye(encoding.shape[1]).to(
                        self.device) * self.cfs_l2_weight+torch.mm(encoding.T, encoding)), encoding.T), y_train)
                    # beta = torch.mm(torch.mm(torch.inverse(torch.eye(encoding.shape[1]).to(
                    #     self.device) * self.cfs_l2_weight+torch.mm(encoding.T, encoding)), encoding.T), y)
                else:   # dual space
                    beta = torch.mm(encoding.T, torch.mm(torch.pinverse(torch.eye(
                        encoding.shape[0]).to(self.device) * self.cfs_l2_weight+torch.mm(encoding, encoding.T)), y_train))
                    # beta = torch.mm(encoding.T, torch.mm(torch.inverse(torch.eye(
                    #     encoding.shape[0]).to(self.device) * self.cfs_l2_weight+torch.mm(encoding, encoding.T)), y))

                self.output.weight[:] = nn.Parameter(beta.T)
                self.output.bias[:] = nn.Parameter(torch.zeros_like(self.output.bias))

                # if X_val is not None and y_val is not None and X_val_raw is not None:
                if X_val is not None and y_val is not None:
                    history_entity = self.__eval__(X=X_train, y=y_train, X_val=X_val, y_val=y_val, train_method="cfs", X_raw=X_raw, X_val_raw=X_val_raw)
                    for k in history_entity:
                        self.history[k].append(history_entity[k])
            except Exception as e:
                print("Error training MLP using CFS")
                raise e
        te = time.time() - t
        self.history["cfs_time"].append(te) 
        if return_details:
            return te, encoding, beta
        return te  
    
    def prune_neurons(self, X_train, y_train, X_raw=None, X_val=None, y_val=None, X_val_raw=None, n_nodes=None, percentage=None):
        if n_nodes is None and percentage is None:
            raise ValueError("Either n_nodes or percentage has to be specified!")
        if n_nodes == 0 or percentage == 0:
            return
        if self.n_layers != 1:
            raise Exception("Pruning is only implemented for single-hidden layer networks")
        contributions = torch.sum(torch.abs(self.output.weight), dim=0) # shape is (n_nodes, )
        n_nodes_to_prune = n_nodes if n_nodes else int(self.n_nodes * percentage)
        n_nodes_to_keep = self.n_nodes - n_nodes_to_prune
        offset = len(contributions) - self.n_nodes     ## this will be n input features if direct links is enabled or 0 otherwise
        self.n_nodes = n_nodes_to_keep
        indices = contributions[offset:].argsort()[n_nodes_to_prune:]
        ## update hidden layer
        old_weights = self.hidden[0][0].weight
        in_features = old_weights.shape[1]
        self.hidden[0][0] = nn.Linear(in_features, n_nodes_to_keep).to(device=self.device)
        self.hidden[0][0].weight = nn.Parameter(old_weights[indices, :])
        if not self.use_selu:
            self.hidden[0][-1] = nn.BatchNorm1d(n_nodes_to_keep, affine=self.bn_learnable, momentum=0.1, track_running_stats=self.track_running_stats).to(device=self.device)
        ## update output layer
        self.initialize_output_layer(n_nodes=n_nodes_to_keep, n_classes=self.n_classes, output_direct_link=self.output_direct_link)
        self.do_cfs_training(X_train=X_train, y_train=y_train, X_raw=X_raw, X_val=X_val, y_val=y_val, X_val_raw=X_val_raw)


    def __eval__(self, X, y, X_val, y_val, train_method, X_raw=None, X_val_raw=None):
        with torch.no_grad():
            self.eval()
            train_pred = self(X, X_raw)
            train_loss = self.loss_fn(train_pred, y).mean()
         
            train_acc = accuracy_score(y.cpu().argmax(1), train_pred.cpu().argmax(1))

            val_pred = self(X_val, X_val_raw)
            val_loss = self.loss_fn(val_pred, y_val).mean()
         
            val_acc = accuracy_score(y_val.cpu().argmax(1), val_pred.cpu().argmax(1))
         
            return {"train_loss": train_loss.cpu(), "train_acc": train_acc, "val_loss": val_loss.cpu(), "val_acc": val_acc, "train_method": train_method}
        
    def get_embeddings(self, idx, X, X_raw=None):
        assert 0 <= idx <= self.n_layers, f"Invalid idx {idx}. Must be in [0, {self.n_layers}]"

        if X_raw is None:
            X_raw = X.clone()

        out = X
        for i, layer in enumerate(self.hidden):
            out = layer(out)
            if i == idx:
                return out

        if self.output_direct_link:
            out = torch.cat([X_raw.to(self.device), out], dim=1)
        out = self.output(out)
        return out 


class MLPModel(ClassificationModel):
    def __init__(self, 
                 n_layers,
                n_nodes,
                l2_weight,
                l1_weight,
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
                initialize_linear=True,
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
        
        self.initialize_linear = initialize_linear

        self.device = device
        self.use_selu = use_selu
        if use_drop_out:
            self.p_drop = p_drop
        else:
            self.p_drop = 0      

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

    def _fit(self, X_train , y_train, X_val = None , y_val = None):
        X_train = torch.Tensor(X_train).float().to(self.device)
        y_train = torch.Tensor(y_train).float().to(self.device)
        if not X_val is None : 
            X_val = torch.Tensor(X_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)

        
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
     
    def _predict(self, X):
        self.model.eval()
        X = torch.Tensor(X).float().to(self.device)
        pred_score = self.model(X)
        return pred_score.detach().cpu().numpy()
    
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
        # return 0, 0

    def measure_flops(self, sample_input):
        try:
            sample_input = torch.Tensor(sample_input).float().to(self.device)
            self.model.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(self.model, sample_input)
                total_flops = flops.total()
                return total_flops
        except Exception as e:
            raise Exception(f"FLOPs measurement failed: {e}")
        
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
    
    def save(self, path):
        """
        Save the model to the specified path.
        """        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_layers': self.n_layers,
            'n_nodes': self.n_nodes,
            'output_direct_link': self.output_direct_link,
            'l2_weight': self.l2_weight,
            'l1_weight': self.l1_weight,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_percentage': self.batch_percentage,
            'gamma': self.gamma,
            'beta': self.beta,
            'bn_learnable': self.bn_learnable,
            'track_running_stats': self.track_running_stats,
            'device': str(self.device),
            'loss': self.loss,
            'initialize_linear': self.initialize_linear,
            'use_selu': self.use_selu,
            'p_drop': self.p_drop
        }, path)
        print(f"MLP saved to {path}")
        

    def load(self, path):
        """
        Load the model from the specified path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"MLP loaded from {path}")

    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = MLPModel(
            n_layers=checkpoint['n_layers'],
            n_nodes=checkpoint['n_nodes'],
            output_direct_link=checkpoint['output_direct_link'],
            l2_weight=checkpoint['l2_weight'],
            l1_weight=checkpoint['l1_weight'],
            epochs=checkpoint['epochs'],
            learning_rate=checkpoint['learning_rate'],
            batch_percentage=checkpoint['batch_percentage'],
            gamma=checkpoint['gamma'],
            beta=checkpoint['beta'],
            bn_learnable=checkpoint['bn_learnable'],
            track_running_stats=checkpoint['track_running_stats'],
            device=torch.device(checkpoint['device']),
            loss=checkpoint['loss'],
            initialize_linear=checkpoint['initialize_linear'],
            use_selu=checkpoint.get('use_selu', False),
            p_drop=checkpoint.get('p_drop', 0),
        )
        n_features = checkpoint["model_state_dict"]["hidden.0.0.weight"].shape[1]
        n_classes = checkpoint["model_state_dict"]["output.weight"].shape[0]
        model.model.initialize(X_train=torch.zeros(1, n_features), y_train=torch.zeros(1, n_classes))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.to(model.device)
        return model