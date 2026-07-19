"""Zero-state metrics for cooling / heating / pv (spec 15.8)."""
import numpy as np


class StreamingZeroMetrics:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.tp = self.fp = self.tn = self.fn = 0
        self.brier_sum = 0.0
        self.n = 0
        self.true_zero = 0
        self.pred_zero = 0
        self.probs = []
        self.labels = []

    def update(self, active_prob: np.ndarray, active_truth: np.ndarray):
        p = np.asarray(active_prob, dtype=np.float64).ravel()
        y = np.asarray(active_truth, dtype=bool).ravel()
        pred_active = p > 0.5
        self.tp += int((pred_active & y).sum())
        self.fp += int((pred_active & ~y).sum())
        self.tn += int((~pred_active & ~y).sum())
        self.fn += int((~pred_active & y).sum())
        self.brier_sum += float(np.square(p - y.astype(np.float64)).sum())
        self.n += len(y)
        self.true_zero += int((~y).sum())
        self.pred_zero += int((~pred_active).sum())
        # keep a subsample for ECE if the stream is huge
        self.probs.append(p)
        self.labels.append(y)

    def _ece(self):
        p = np.concatenate(self.probs)
        y = np.concatenate(self.labels).astype(np.float64)
        order = np.argsort(p)
        p, y = p[order], y[order]
        bins = np.array_split(np.arange(len(p)), self.n_bins)  # equal-frequency bins
        ece = 0.0
        for b in bins:
            if len(b) == 0:
                continue
            conf = p[b].mean()
            acc = y[b].mean()
            ece += (len(b) / len(p)) * abs(acc - conf)
        return ece

    def reliability_bins(self):
        p = np.concatenate(self.probs)
        y = np.concatenate(self.labels).astype(np.float64)
        order = np.argsort(p)
        p, y = p[order], y[order]
        bins = np.array_split(np.arange(len(p)), self.n_bins)
        rows = []
        for b in bins:
            if len(b) == 0:
                continue
            rows.append({"confidence": float(p[b].mean()),
                         "accuracy": float(y[b].mean()),
                         "count": int(len(b))})
        return rows

    def compute(self):
        n = max(self.n, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        true_zero_rate = self.true_zero / n
        pred_zero_rate = self.pred_zero / n
        return {
            "active_accuracy": (self.tp + self.tn) / n,
            "active_precision": precision,
            "active_recall": recall,
            "active_f1": f1,
            "brier": self.brier_sum / n,
            "ece": self._ece(),
            "true_zero_rate": true_zero_rate,
            "pred_zero_rate": pred_zero_rate,
            "zero_rate_abs_error": abs(true_zero_rate - pred_zero_rate),
        }


def active_prob_from_samples(samples: np.ndarray) -> np.ndarray:
    """Baselines without a gate: P(active) = mean(samples > 0) (spec 15.8)."""
    return (np.asarray(samples) > 0).mean(axis=-1)
