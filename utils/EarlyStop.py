import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, delta=0, monitor='acc'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.val_loss_min = np.Inf

    def __call__(self, val, model, path):
        if self.monitor == 'acc':
            score = val
        else:
            score = -val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model, path)
            self.counter = 0

    def save_checkpoint(self, val, model, path):
        """Saves model when validation loss decrease."""
        if self.monitor == 'acc':
            if val > self.val_acc_max:
                torch.save(model.state_dict(), path)
                self.val_acc_max = val
                self.counter = 0
        if self.monitor == 'loss':
            if val < self.val_loss_min:
                torch.save(model.state_dict(), path)
                self.val_loss_min = val
                self.counter = 0
