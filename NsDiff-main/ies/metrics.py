import numpy as np

# ---------- 概率保真 ----------
def crps(y, s, max_pair=50):                 # y:(N,O,D) s:(N,S,O,D) → 标量
    t1 = np.mean(np.abs(s - y[:, None]))
    S = s.shape[1]
    idx = np.linspace(0, S - 1, min(S, max_pair)).round().astype(int)
    xs = s[:, idx]
    t2 = 0.5 * np.mean(np.abs(xs[:, :, None] - xs[:, None]))
    return float(t1 - t2)


def crps_per_target(y, s):
    return [crps(y[..., i:i + 1], s[..., i:i + 1]) for i in range(y.shape[-1])]


def qice(y, s, bins=10):                      # PIT 校准误差(百分点)
    pit = np.mean(s <= y[:, None], axis=1).reshape(-1)
    c, _ = np.histogram(np.clip(pit, 0, 1), bins=np.linspace(0, 1, bins + 1))
    p = c / max(c.sum(), 1)
    return float(np.mean(np.abs(p - 1.0 / bins)) * 100)


def qice_per_target(y, s, bins=10):
    return [qice(y[..., i:i + 1], s[..., i:i + 1], bins) for i in range(y.shape[-1])]


# ---------- 联合(标准化空间算) ----------
def energy_score(y, s, max_pair=50):          # 跨变量联合:每步在 D 维上算能量分
    N, S, O, D = s.shape
    idx = np.linspace(0, S - 1, min(S, max_pair)).round().astype(int); xs = s[:, idx]
    d1 = np.linalg.norm(s - y[:, None], axis=-1).mean()                      # E||X-y||_D
    d2 = np.linalg.norm(xs[:, :, None] - xs[:, None], axis=-1).mean()        # E||X-X'||_D
    return float(d1 - 0.5 * d2)


def variogram_score(y, s, p=0.5):
    N, S, O, D = s.shape; vs = 0.0
    for a in range(D):
        for b in range(D):
            emp = (np.abs(s[..., a] - s[..., b]) ** p).mean(1)               # (N,O)
            real = np.abs(y[..., a] - y[..., b]) ** p
            vs += ((real - emp) ** 2).mean()
    return float(vs)


def corr_error(y, s):                          # 变量相关矩阵误差 + 返回两矩阵(画图2)
    r_real = np.corrcoef(y.reshape(-1, y.shape[-1]).T)
    r_gen = np.corrcoef(s.mean(1).reshape(-1, s.shape[-1]).T)
    return float(np.nanmean(np.abs(r_real - r_gen))), r_real, r_gen


def psd_distance(y, s):
    P = lambda z: np.abs(np.fft.rfft(z, axis=1)) ** 2
    return float(np.linalg.norm(P(y) - P(s.mean(1)), axis=1).mean())


def acf_distance(y, s, L=24):
    def acf(z):                                # z:(N,O,D)
        z = z - z.mean(1, keepdims=True); out = []
        for l in range(1, L + 1):
            num = (z[:, l:] * z[:, :-l]).sum(1)
            den = (z ** 2).sum(1) + 1e-8
            out.append((num / den))
        return np.stack(out, 1)                # (N,L,D)
    return float(np.abs(acf(y) - acf(s.mean(1))).mean())


# ---------- 多样性 ----------
def apd(s, max_pair=50):                        # 条件内平均成对距离
    S = s.shape[1]; idx = np.linspace(0, S - 1, min(S, max_pair)).round().astype(int)
    xs = s[:, idx]
    return float(np.linalg.norm(xs[:, :, None] - xs[:, None], axis=-1).mean())


def avg_std(s):                                 # 条件平均标准差
    return float(s.std(1).mean())


# ---------- 点误差(kW 分变量) ----------
def mae_rmse_per_target(y, s):
    m = s.mean(1)
    mae = np.abs(m - y).mean((0, 1)); rmse = np.sqrt(((m - y) ** 2).mean((0, 1)))
    return mae.tolist(), rmse.tolist()


def dtw_distance(a, b):                          # 单序列 O(O^2) DP
    n, m = len(a), len(b); D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def dtw_mean(y, s, sample_windows=200):          # 抽样窗口算 DTW,均值
    m = s.mean(1); N, O, D = y.shape
    idx = np.linspace(0, N - 1, min(N, sample_windows)).round().astype(int)
    vals = [dtw_distance(m[i, :, d], y[i, :, d]) for i in idx for d in range(D)]
    return float(np.mean(vals))
