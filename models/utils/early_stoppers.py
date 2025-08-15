# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import torch
import uuid
import numpy as np
import os
import copy


class EarlyStopping:
    """
    A class to implement early stopping during model training. This class monitors a given performance measure 
    (e.g., validation loss or accuracy) and stops training if the measure does not improve after a certain number 
    of epochs (patience). It also supports saving and loading model checkpoints.

    Attributes:
    -----------
    patience : int
        Number of epochs to wait for an improvement before stopping training.
    delta : float
        Minimum change in the monitored measure to qualify as an improvement.
    save_location : str
        Where to save the model checkpoints. Options are "disk" or "ram".
    checkpoint_path : str or None
        The path where the model checkpoint is saved (only relevant if save_location is "disk").
    best_measure : float
        The best value of the monitored measure observed during training.
    early_stop : bool
        Flag to indicate if training should be stopped early.
    measures : list
        List to store the history of the monitored measure.
    counter : int
        Counts the number of epochs without improvement.
    is_minimization : bool
        Indicates if the monitored measure is being minimized (e.g., for loss) or maximized (e.g., for accuracy).
    track_moving_avg : bool
        Flag to indicate if a moving average of the measure should be tracked.
    smoothing : int
        The window size for the moving average.
    ema : float or None
        Exponentially moving average of the measure (if tracking moving average).
    checkpoint : dict or None
        The state dictionary of the model (if save_location is "ram").
    """
    def __init__(self, patience=10, delta=0, checkpoint_name='checkpoints', is_minimization=True, model=None, save_location="disk", track_moving_avg=False, smoothing=10):
        """
        Initializes the EarlyStopping object with the given parameters.

        Parameters:
        -----------
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 10).
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement (default is 0).
        checkpoint_name : str, optional
            Name of the directory where checkpoints will be saved (default is 'checkpoints').
        is_minimization : bool, optional
            Whether the monitored measure should be minimized or maximized (default is True).
        model : torch.nn.Module, optional
            The model to be saved during checkpointing (default is None).
        save_location : str, optional
            Location to save the checkpoint, either "disk" or "ram" (default is "disk").
        track_moving_avg : bool, optional
            Whether to track the moving average of the monitored measure (default is False).
        smoothing : int, optional
            Smoothing factor for the moving average (default is 10).
        """
        self.patience = patience
        self.delta = delta
        self.save_location = save_location
        if save_location == "disk" and checkpoint_name is not None:
            self.checkpoint_path = f"{checkpoint_name}/{str(uuid.uuid4())}.pt"
            if not os.path.exists(checkpoint_name):
                os.makedirs(checkpoint_name)
        else:
            self.checkpoint_path = None
        if is_minimization:
            self.best_measure = np.inf
        else:
            self.best_measure = -np.inf
        self.early_stop = False
        self.measures = []
        self.counter = 0
        self.is_minimization = is_minimization
        if model is not None:
            self.save_checkpoint(np.inf if is_minimization else -np.inf, model)
        self.track_moving_avg = track_moving_avg
        if self.track_moving_avg:
            self.smoothing = smoothing
            self.ema = None

    def __call__(self, measure, model=None, save_dict={}):
        """
        Call method to evaluate the monitored measure and determine if early stopping should be triggered.
        
        Parameters:
        -----------
        measure : float
            The current value of the monitored measure.
        model : torch.nn.Module, optional
            The current model instance to be saved as a checkpoint (default is None).
        """
        if not self.track_moving_avg:
            tracked_measure = measure
        else:
            tracked_measure = measure if len(self.measures) < self.smoothing else self.moving_average()
            
        
        if (self.is_minimization and tracked_measure < self.best_measure - self.delta) or (not self.is_minimization and tracked_measure > self.best_measure + self.delta):
            self.best_measure = tracked_measure
            self.counter = 0
            if self.save_location == "ram":
                if model is None:
                    raise Exception(
                        "No model were passed, unable to save checkpoint")
                self.save_checkpoint(tracked_measure, model, save_dict)
            elif self.checkpoint_path is not None:
                if model is None:
                    raise Exception(
                        "No model were passed, unable to save checkpoint")
                self.save_checkpoint(tracked_measure, model, save_dict)
                
                
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        self.measures.append(measure)

    @property
    def best_iter(self):
        """
        Returns the iteration (epoch) at which the best measure was observed.
        
        Returns:
        --------
        int
            The iteration number of the best measure.
        """
        return len(self.measures) - self.counter

    def save_checkpoint(self, measure, model, save_dict={}):
        """
        Saves the model checkpoint either to disk or in RAM.

        Parameters:
        -----------
        measure : float
            The current value of the monitored measure.
        model : torch.nn.Module
            The model instance to be saved.
        """
        if self.save_location == "disk":
            torch.save(save_dict | {'model_state_dict': model.state_dict(),
                    'measure': measure}, self.checkpoint_path)
        elif self.save_location == "ram":
            self.checkpoint = copy.deepcopy(model.state_dict())
        else:
            raise Exception("Should never happen!")
        
    def load_checkpoint(self, model):
        """
        Loads the model checkpoint either from disk or from RAM.

        Parameters:
        -----------
        model : torch.nn.Module
            The model instance where the checkpoint will be loaded.
        """
        if self.save_location == "disk":
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.best_measure = checkpoint['measure']
        elif self.save_location == "ram":
            model.load_state_dict(self.checkpoint)
        else:
            raise Exception("Should never happen!")
        
        
    def moving_average(self):
        """
        Computes the moving average of the monitored measure.
        
        Returns:
        --------
        float
            The current moving average of the monitored measure.
        """
        return np.convolve(self.measures[-self.smoothing:], np.ones(self.smoothing), 'valid') / self.smoothing