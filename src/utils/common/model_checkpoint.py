import os
import torch

class ModelCheckpoint:
    def __init__(self, path='checkpoint.pt', verbose=False):
        self.best_val_loss = float('inf')
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.verbose = verbose
        self.model_config = {}

    def set_config(self, key: str, value: any): 
        self.model_config[key] = value

    def save_if_better(self, val_loss, model):
        if val_loss < self.best_val_loss:
            if self.verbose:
                print(f"Validation loss improved ({self.best_val_loss:.6f} → {val_loss:.6f}). Saving model.")
            torch.save({ "model_state": model.state_dict(), "model_config": self.model_config }, self.path)
            self.best_val_loss = val_loss

    def load_best_model_config(self):
        """
        Returns: (state, config)
        """
        checkpoint = torch.load(self.path, weights_only=False)
        return (checkpoint["model_state"], checkpoint["model_config"])
    

    @staticmethod
    def load(path: str):
        """
        Returns: (state, config)
        """
        checkpoint = torch.load(path, weights_only=False)
        return (checkpoint["model_state"], checkpoint["model_config"])