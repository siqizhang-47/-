"""Point metrics: MAPE (electricity), MAPE+ (positive-only), MAPE_eps, MAE, RMSE.

All operate on 1-D numpy arrays of predictions/truth for a single variable;
streaming accumulators are provided for shard-wise evaluation.
"""
import numpy as np


class StreamingPointMetrics:
    def __init__(self, positive_scale: float = 1.0):
        self.positive_scale = positive_scale
        self.abs_err = 0.0
        self.sq_err = 0.0
        self.n = 0
        self.ape_sum = 0.0          # over y > 0 only (MAPE+ / MAPE for electricity)
        self.n_pos = 0
        self.ape_eps_sum = 0.0      # stabilised full-sample MAPE_eps
        self.n_eps = 0

    def update(self, pred: np.ndarray, truth: np.ndarray):
        pred = np.asarray(pred, dtype=np.float64).ravel()
        truth = np.asarray(truth, dtype=np.float64).ravel()
        err = np.abs(pred - truth)
        self.abs_err += err.sum()
        self.sq_err += np.square(pred - truth).sum()
        self.n += len(truth)
        pos = truth > 0
        if pos.any():
            self.ape_sum += (err[pos] / truth[pos]).sum()
            self.n_pos += int(pos.sum())
        denom = np.maximum(np.abs(truth), 0.01 * self.positive_scale)
        self.ape_eps_sum += (err / denom).sum()
        self.n_eps += len(truth)

    def compute(self):
        return {
            "mae": self.abs_err / max(self.n, 1),
            "rmse": float(np.sqrt(self.sq_err / max(self.n, 1))),
            "mape_pos": 100.0 * self.ape_sum / max(self.n_pos, 1),
            "mape_eps": 100.0 * self.ape_eps_sum / max(self.n_eps, 1),
        }
