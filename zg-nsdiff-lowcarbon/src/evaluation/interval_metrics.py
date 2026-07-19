"""Interval metrics with the corrected PICP counting (spec 2.1/2.2, 15.4/15.5).

PICP accumulates covered_count / total_count over coordinates, and the main
interval is the central 95% one: quantiles 0.025 / 0.975.
"""
import numpy as np

MAIN_INTERVAL = (0.025, 0.975)
NOMINAL_LEVELS = [0.50, 0.80, 0.90, 0.95]


def interval_bounds(samples: np.ndarray, lower_q: float, upper_q: float):
    """samples [..., S] -> (lower, upper) along the sample axis."""
    lower = np.quantile(samples, lower_q, axis=-1)
    upper = np.quantile(samples, upper_q, axis=-1)
    return lower, upper


class StreamingIntervalMetrics:
    def __init__(self, levels=None):
        self.levels = list(levels or NOMINAL_LEVELS)
        self.covered = {lv: 0 for lv in self.levels}
        self.total = {lv: 0 for lv in self.levels}
        self.width_sum = {lv: 0.0 for lv in self.levels}
        self.truth_min = np.inf
        self.truth_max = -np.inf

    def update(self, samples: np.ndarray, truth: np.ndarray):
        """samples [..., S], truth [...] for one variable."""
        truth = np.asarray(truth, dtype=np.float64)
        self.truth_min = min(self.truth_min, float(truth.min()))
        self.truth_max = max(self.truth_max, float(truth.max()))
        for lv in self.levels:
            lq, uq = (1 - lv) / 2, 1 - (1 - lv) / 2
            lower, upper = interval_bounds(samples, lq, uq)
            in_range = (truth >= lower) & (truth <= upper)
            self.covered[lv] += int(in_range.sum())
            self.total[lv] += int(in_range.size)
            self.width_sum[lv] += float((upper - lower).sum())

    def compute(self, truth_range: float = None):
        out = {}
        rng = truth_range if truth_range is not None else (self.truth_max - self.truth_min)
        rng = max(rng, 1e-12)
        for lv in self.levels:
            n = max(self.total[lv], 1)
            picp = self.covered[lv] / n
            aw = self.width_sum[lv] / n
            key = f"{int(round(lv * 100))}"
            out[f"picp_{key}"] = picp
            out[f"aw_{key}"] = aw
            out[f"pinaw_{key}"] = aw / rng
            out[f"coverage_error_{key}"] = abs(picp - lv)
        out["picp"] = out["picp_95"]
        out["aw"] = out["aw_95"]
        out["pinaw"] = out["pinaw_95"]
        out["coverage_error"] = out["coverage_error_95"]
        return out
