"""Small helpers shared by the training scripts."""
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningMean:
    """Per-key running means of scalar values, weighted by batch size."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def update(self, values, n=1):
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    def means(self):
        return {key: self.sums[key] / self.counts[key] for key in self.sums}


class EarlyStopping:
    """Tracks the best validation score (higher is better) and epochs without improvement."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, score):
        """Record an epoch's score; return True if it is a new best."""
        is_best = score > self.best
        if score > self.best + self.min_delta:
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if is_best:
            self.best = score
        return is_best

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience

    def state_dict(self):
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]
