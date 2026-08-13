"""Early stopping on a monitored validation metric (lower is better)."""


class EarlyStopper:
    def __init__(self, patience: int = 4, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_step = -1
        self.n_bad = 0

    def update(self, value: float, step: int) -> bool:
        """Record a new metric value. Returns True if training should stop."""
        if value < self.best - self.min_delta:
            self.best = value
            self.best_step = step
            self.n_bad = 0
        else:
            self.n_bad += 1
        return self.n_bad >= self.patience

    @property
    def improved(self) -> bool:
        return self.n_bad == 0
