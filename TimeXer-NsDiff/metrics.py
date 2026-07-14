"""
Metrics for the TimeXer-NsDiff probabilistic forecasts (plan §22).

Deterministic (on the sample mean, raw scale): MAE, RMSE, sMAPE.
Probabilistic: CRPS, QICE, interval coverage @50/80/90/95 (PICP),
mean interval width @90 (MIW), Energy Score, Variogram Score.

CRPS / QICE / coverage / width / MAE / RMSE / sMAPE are accumulated in a
streaming fashion (RunningMetrics) so the full [M,S,H,4] sample tensor never has
to be held in memory. Energy Score and Variogram Score are multivariate (over the
4 targets) and computed on a capped subsample of windows.

All metrics are reported in REAL (inverse-transformed, physical) units. Note that
ES/VS on raw values are dominated by the largest-magnitude target (Electricity);
switch to standardised inputs in main.evaluate if a scale-balanced score is wanted.
"""
import numpy as np

COVERAGE_LEVELS = [50, 80, 90, 95]
QICE_BINS = 10


class RunningMetrics:
    def __init__(self, n_targets=4, n_bins=QICE_BINS):
        self.K, self.n_bins = n_targets, n_bins
        self.n = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.smape_sum = 0.0
        self.crps_sum = 0.0
        self.cov_hits = {q: 0.0 for q in COVERAGE_LEVELS}
        self.width_sum = {q: 0.0 for q in COVERAGE_LEVELS}
        self.bin_counts = np.zeros(n_bins, dtype=np.float64)

    def update(self, samples_raw, truth_raw):
        # samples_raw: [b,S,H,K]   truth_raw: [b,H,K]
        S = samples_raw.shape[1]
        mean = samples_raw.mean(1)                              # [b,H,K]
        diff = mean - truth_raw
        self.abs_sum += np.abs(diff).sum()
        self.sq_sum += (diff ** 2).sum()
        self.smape_sum += (2 * np.abs(diff) /
                           (np.abs(mean) + np.abs(truth_raw) + 1e-6)).sum()
        npts = truth_raw.size
        self.n += npts

        # CRPS (per point, sample based): E|X-y| - 0.5 E|X-X'|
        y = truth_raw[:, None]                                  # [b,1,H,K]
        term1 = np.abs(samples_raw - y).mean(1)                 # [b,H,K]
        xs = np.sort(samples_raw, axis=1)
        w = (2 * np.arange(1, S + 1) - S - 1).reshape(1, S, 1, 1)
        term2 = 2.0 * (w * xs).sum(1) / (S * S)
        self.crps_sum += (term1 - 0.5 * term2).sum()

        # coverage + interval width
        for q in COVERAGE_LEVELS:
            lo = np.percentile(samples_raw, (100 - q) / 2, axis=1)
            hi = np.percentile(samples_raw, (100 + q) / 2, axis=1)
            self.cov_hits[q] += ((truth_raw >= lo) & (truth_raw <= hi)).sum()
            self.width_sum[q] += (hi - lo).sum()

        # QICE bin counts
        ql = np.arange(self.n_bins + 1) * (100.0 / self.n_bins)
        pq = np.percentile(samples_raw, ql, axis=1)             # [nb+1,b,H,K]
        pq = pq.reshape(self.n_bins + 1, -1)
        tgt = truth_raw.reshape(-1)
        member = ((tgt[None] - pq) > 0).astype(int).sum(0)      # [npts]
        c = np.array([(member == v).sum() for v in range(self.n_bins + 2)], np.float64)
        c[1] += c[0]; c[-2] += c[-1]
        self.bin_counts += c[1:-1]

    def finalize(self):
        n = max(self.n, 1)
        out = {
            "MAE": self.abs_sum / n,
            "RMSE": np.sqrt(self.sq_sum / n),
            "sMAPE": 100.0 * self.smape_sum / n,
            "CRPS": self.crps_sum / n,
            "QICE": float(np.mean(np.abs(1.0 / self.n_bins -
                                        self.bin_counts / self.bin_counts.sum()))),
        }
        for q in COVERAGE_LEVELS:
            out[f"PICP{q}"] = self.cov_hits[q] / n
        out["MIW90"] = self.width_sum[90] / n
        return out


def energy_score(samples, truth):
    """Multivariate Energy Score over the K targets. samples [W,S,H,K], truth [W,H,K].
    Per (window,h) 4-dim vector; ES = E||X-y|| - 0.5 E||X-X'||, averaged."""
    W, S, H, K = samples.shape
    total, cnt = 0.0, 0
    for w in range(W):
        for h in range(H):
            X = samples[w, :, h, :]                 # [S,K]
            y = truth[w, h, :]                      # [K]
            t1 = np.linalg.norm(X - y[None], axis=1).mean()
            d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            t2 = d.sum() / (S * S)
            total += t1 - 0.5 * t2
            cnt += 1
    return total / max(cnt, 1)


def variogram_score(samples, truth, p=0.5):
    """Variogram score of order p over the K targets (captures dependence)."""
    W, S, H, K = samples.shape
    total, cnt = 0.0, 0
    for w in range(W):
        for h in range(H):
            X = samples[w, :, h, :]                 # [S,K]
            y = truth[w, h, :]
            # |y_i - y_j|^p  vs  E|X_i - X_j|^p over all target pairs
            yd = np.abs(y[:, None] - y[None, :]) ** p
            xd = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(0)
            total += ((yd - xd) ** 2).sum()
            cnt += 1
    return total / max(cnt, 1)
