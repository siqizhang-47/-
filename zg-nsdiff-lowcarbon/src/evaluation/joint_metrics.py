"""Joint multivariate metrics (spec 15.7): standardized Energy Score,
Variogram Score (p=0.5), correlation matrix error.

All computed on trajectories standardized by the TRAIN raw scales so
electricity's magnitude does not dominate.
"""
import numpy as np

VARIOGRAM_ORDER = 0.5


class StreamingJointMetrics:
    def __init__(self, train_scale: np.ndarray, variogram_order: float = VARIOGRAM_ORDER,
                 max_pairwise_samples: int = 100, corr_max_windows: int = 20000,
                 rng_seed: int = 0):
        self.scale = np.asarray(train_scale, dtype=np.float64).reshape(1, 1, -1)
        self.p = variogram_order
        self.max_pairwise = max_pairwise_samples
        self.corr_max_windows = corr_max_windows
        self.rng = np.random.default_rng(rng_seed)
        self.es_sum = 0.0
        self.vs_sum = 0.0
        self.n = 0
        self._true_flat = []
        self._gen_flat = []

    def update(self, samples: np.ndarray, truth: np.ndarray):
        """samples [n,H,D,S], truth [n,H,D] in raw units."""
        z_truth = truth / self.scale                          # [n,H,D]
        z_samples = samples / self.scale[..., None]           # [n,H,D,S]
        n, H, D, S = z_samples.shape
        flat_t = z_truth.reshape(n, H * D)
        flat_s = np.moveaxis(z_samples, -1, 1).reshape(n, S, H * D)

        # Energy Score: mean_s ||X_s - y|| - 1/(2 S'^2) sum ||X_s - X_r||
        term1 = np.linalg.norm(flat_s - flat_t[:, None, :], axis=-1).mean(axis=1)
        if S > self.max_pairwise:
            sel = self.rng.choice(S, self.max_pairwise, replace=False)
            sub = flat_s[:, sel, :]
        else:
            sub = flat_s
        Ssub = sub.shape[1]
        diff = sub[:, :, None, :] - sub[:, None, :, :]
        pair = np.linalg.norm(diff, axis=-1)                  # [n,Ssub,Ssub]
        term2 = pair.sum(axis=(1, 2)) / (2.0 * Ssub * Ssub)
        self.es_sum += float((term1 - term2).sum())

        # Variogram Score over the flattened H*D coordinates
        gamma_true = np.abs(flat_t[:, :, None] - flat_t[:, None, :]) ** self.p
        gamma_gen = (
            np.abs(sub[:, :, :, None] - sub[:, :, None, :]) ** self.p
        ).mean(axis=1)
        self.vs_sum += float(np.square(gamma_true - gamma_gen).sum(axis=(1, 2)).sum())

        self.n += n

        # correlation matrices: keep truth + one random generated trajectory
        if sum(len(x) for x in self._true_flat) < self.corr_max_windows:
            pick = self.rng.integers(0, S)
            self._true_flat.append(flat_t)
            self._gen_flat.append(flat_s[:, pick, :])

    def compute(self):
        out = {
            "energy_score": self.es_sum / max(self.n, 1),
            "variogram_score": self.vs_sum / max(self.n, 1),
        }
        t = np.concatenate(self._true_flat, axis=0)
        g = np.concatenate(self._gen_flat, axis=0)
        R_true = np.corrcoef(t, rowvar=False)
        R_gen = np.corrcoef(g, rowvar=False)
        mask = np.isfinite(R_true) & np.isfinite(R_gen)
        out["corr_error"] = float(np.linalg.norm(np.where(mask, R_gen - R_true, 0.0), "fro"))
        return out
