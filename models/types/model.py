# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import random
import numpy as np
import torch

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class Model(ABC):
    
    def __init__(self, seed=None, verbose=0):
        """
        Initialize the Model with an optional seed parameter for reproducibility.
        """
        if seed is not None:
            self.set_seed(seed)
        self.seed = seed
        self.verbose=verbose

    def set_seed(self, seed):
        """
        Set the seed for reproducibility in PyTorch, NumPy, and random.
        """
        random.seed(int(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def _fit(self, X_train, y_train, X_val, y_val, **fitting_args):
        pass
    
    @abstractmethod
    def _score(self, X, y):
        pass

    def score(self, X, y):
        return self._score(X, y)
    
    def predict(self, X):
        return self._predict(X)

    def fit(self, X_train, y_train, X_val=None, y_val=None, **fitting_args):
        self._fit(X_train, y_train, X_val, y_val, **fitting_args)
    
    def log(self, *message, level, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if level <= verbose:
            print(*message)

class ClassificationModel(Model):
    
    def calc_accuracy(self, X=None, y=None, pred=None):
        """
        Calculate accuracy score. If `pred` is provided, use it directly;
        otherwise, use `self.predict(X)` to get predictions.
        """
        if pred is None:
            if X is None:
                raise ValueError("Either X or pred must be provided.")
            y_pred = self.predict(X)
        else:
            y_pred = pred
            
        return accuracy_score(y.argmax(axis=1), y_pred.argmax(axis=1))

    def calc_f1(self, X=None, y=None, pred=None):
        """
        Calculate accuracy score. If `pred` is provided, use it directly;
        otherwise, use `self.predict(X)` to get predictions.
        """
        if pred is None:
            if X is None:
                raise ValueError("Either X or pred must be provided.")
            y_pred = self.predict(X)
        else:
            y_pred = pred
            
        return f1_score(y.argmax(1), y_pred.argmax(1), average="macro", zero_division=0)

    def calc_precision(self, X=None, y=None, pred=None):
        """
        Calculate accuracy score. If `pred` is provided, use it directly;
        otherwise, use `self.predict(X)` to get predictions.
        """
        if pred is None:
            if X is None:
                raise ValueError("Either X or pred must be provided.")
            y_pred = self.predict(X)
        else:
            y_pred = pred
            
        return precision_score(y.argmax(1), y_pred.argmax(1), average="macro", zero_division=0)

    def calc_recall(self, X=None, y=None, pred=None):
        """
        Calculate accuracy score. If `pred` is provided, use it directly;
        otherwise, use `self.predict(X)` to get predictions.
        """
        if pred is None:
            if X is None:
                raise ValueError("Either X or pred must be provided.")
            y_pred = self.predict(X)
        else:
            y_pred = pred
            
        return recall_score(y.argmax(1), y_pred.argmax(1), average="macro", zero_division=0)

    def calc_conf_mat(self, X=None, y=None, pred=None):
        """
        Calculate accuracy score. If `pred` is provided, use it directly;
        otherwise, use `self.predict(X)` to get predictions.
        """
        if pred is None:
            if X is None:
                raise ValueError("Either X or pred must be provided.")
            y_pred = self.predict(X)
        else:
            y_pred = pred
            
        return confusion_matrix(y.argmax(1), y_pred.argmax(1))
    
    def _score(self, X, y):
        return self.calc_accuracy(X, y)
    
    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        return {}
        # return {"TEST": 99}

    def save(self, path):
        """
        Save the model to the specified path.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError(f"Save method not implemented for {self.__class__.__name__}. ")