"""Early stopping with best-checkpoint restore.

Used by every training stage: stage 1/2/3 watch a validation loss, stage 4
watches either the validation diffusion loss or validation CRPS (configurable).
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience: int = 8, delta: float = 0.0, mode: str = "min",
                 path: str | None = None, name: str = "checkpoint", verbose: bool = True,
                 keep_in_memory: bool = True):
        assert mode in ("min", "max")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.path = path
        self.name = name
        self.verbose = verbose
        self.keep_in_memory = keep_in_memory

        self.counter = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        self.best_epoch = -1
        self.early_stop = False
        self._state = None

    def _is_better(self, score) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.delta
        return score > self.best_score + self.delta

    def __call__(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        """Returns True when this epoch produced a new best model."""
        improved = self._is_better(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self._save(model)
            if self.verbose:
                print(f"    [early-stop] new best {self.name}: {score:.6f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    [early-stop] no improvement ({self.counter}/{self.patience}), "
                      f"best={self.best_score:.6f} @ epoch {self.best_epoch}")
            if self.counter >= self.patience:
                self.early_stop = True
        return improved

    def _save(self, model):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self.keep_in_memory:
            self._state = state
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)
            torch.save(state, os.path.join(self.path, f"{self.name}.pth"))

    def load_best(self, model) -> bool:
        state = self._state
        if state is None and self.path is not None:
            fp = os.path.join(self.path, f"{self.name}.pth")
            if os.path.exists(fp):
                state = torch.load(fp, map_location="cpu")
        if state is None:
            print(f"    [early-stop] no checkpoint for '{self.name}', keeping current weights")
            return False
        model.load_state_dict(state)
        if self.verbose:
            print(f"    [early-stop] restored best '{self.name}' "
                  f"({self.best_score:.6f} @ epoch {self.best_epoch})")
        return True
