"""
A3. Train conditional marginal distributions for 8 generation variables.
v2: improved training schedule + zero-head calibration reporting.

Changes vs v1:
  - Batch 4096 -> 2048 so each epoch produces ~2x more gradient updates
  - Epochs 60 -> 200, patience 8 -> 25; CosineAnnealingLR (lr 1e-3 -> 1e-5)
    -> E and g_GE no longer hit the epoch cap; later epochs polish the fit
  - Per-variable OVERRIDES dict for fine-tuning if a variable misbehaves
  - Best-epoch val_pin / val_bce tracked separately (was only val_total before)
  - Zero-head calibration reporting on every zero-inflated variable:
        pi_on_zero  : mean pi where y==0   (should approach 1)
        pi_on_pos   : mean pi where y>0    (should approach 0)
        pi_brier    : Brier score          (lower = better calibration)
        pi_auc      : ROC-AUC of pi as zero classifier (higher = better)
        pi_max      : max pi               (was the only thing checked before)

The "pi_max ~= 1" pattern observed for C_tilde / H_tilde / g_GE / g_B is
correct behaviour: those variables are deterministically zero in certain
conditional regimes (e.g. cooling load at 3am in summer).  pi_max==1
means the network is rightly confident on those samples; the new pi_on_pos
metric verifies the model is not over-confident on non-zero hours.

Quantile head: monotone spline with K=20 knots (cumulative softplus
increments).  Hidden=128, two layers, dropout 0.1.

Outputs:
    ckpt/marginals/{E,W,C_tilde,H_tilde,g_GE,g_FC,G_AC,g_B}.pt
    ckpt/marginals/marginal_summary.csv
    ckpt/marginals/marginal_report.json
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
ROOT = Path("/workspace/code")
PROC = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "ckpt" / "marginals"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Energy / efficiency constants for deriving Q_rec, Q_need
H_G = 9.89          # city gas LHV  (kWh / m^3)
KAPPA_GE = 0.5
KAPPA_FC = 0.5
RHO_H = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Data loading + auxiliary computation
# ----------------------------------------------------------------------------
def load_split(split: str) -> dict:
    d = np.load(PROC / f"{split}.npz", allow_pickle=False)
    m = np.load(PROC / f"masks_{split}.npz", allow_pickle=False)
    out = {k: d[k] for k in d.files}
    for k in m.files:
        out[k] = m[k]
    out["Y_names"] = list(d["Y_names"])
    out["C_names"] = list(d["C_names"])
    return out


def derive_aux(Y: np.ndarray, Y_names: list) -> dict:
    """Compute Q_rec, Q_dem, Q_need per hour (shape (N,24))."""
    g_GE = Y[..., Y_names.index("g_GE")]
    P_GE = Y[..., Y_names.index("P_GE")]
    g_FC = Y[..., Y_names.index("g_FC")]
    P_FC = Y[..., Y_names.index("P_FC")]
    C = Y[..., Y_names.index("C")]
    H = Y[..., Y_names.index("H")]
    W = Y[..., Y_names.index("W")]
    eta_GE = np.zeros_like(g_GE, dtype=np.float32)
    eta_FC = np.zeros_like(g_FC, dtype=np.float32)
    sel = g_GE > 1e-3
    eta_GE[sel] = (P_GE[sel] / (H_G * g_GE[sel])).astype(np.float32)
    sel = g_FC > 1e-3
    eta_FC[sel] = (P_FC[sel] / (H_G * g_FC[sel])).astype(np.float32)
    eta_GE = np.clip(eta_GE, 0.0, 0.6)
    eta_FC = np.clip(eta_FC, 0.0, 0.6)
    Q_rec = KAPPA_GE * (1 - eta_GE) * H_G * g_GE + KAPPA_FC * (1 - eta_FC) * H_G * g_FC
    Q_dem = C + H
    Q_need = np.maximum(W + RHO_H * H - Q_rec, 0.0)
    return dict(Q_rec=Q_rec.astype(np.float32),
                Q_dem=Q_dem.astype(np.float32),
                Q_need=Q_need.astype(np.float32))


# ----------------------------------------------------------------------------
# Feature builder
# ----------------------------------------------------------------------------
def cyclic(x: np.ndarray, period: float):
    return np.stack([np.sin(2 * np.pi * x / period),
                     np.cos(2 * np.pi * x / period)], axis=-1)


def regime_onehot(regime_hr: np.ndarray) -> np.ndarray:
    out = np.zeros(regime_hr.shape + (3,), dtype=np.float32)
    out[..., 0] = (regime_hr == 1)
    out[..., 1] = (regime_hr == 2)
    out[..., 2] = (regime_hr == 3)
    return out


def build_features(spec_keys: list, data: dict, aux: dict) -> np.ndarray:
    Y = data["Y"]
    C = data["C"]
    Y_names = data["Y_names"]
    C_names = data["C_names"]
    regime = data["regime"]
    regime_hr = np.broadcast_to(regime[:, None], (Y.shape[0], 24)).astype(np.int8)
    feats = []
    for key in spec_keys:
        if key in ("T_out", "I", "RH", "v"):
            feats.append(C[..., C_names.index(key)][..., None])
        elif key == "h":
            feats.append(cyclic(C[..., C_names.index("h")], 24.0))
        elif key == "d":
            feats.append(cyclic(C[..., C_names.index("d")], 7.0))
        elif key in ("w", "s"):
            feats.append(C[..., C_names.index(key)][..., None])
        elif key == "regime":
            feats.append(regime_onehot(regime_hr))
        elif key in ("E", "P_PV", "P_GE", "P_FC", "g_GE", "g_FC",
                     "C_load", "H_load", "W_load"):
            name_map = {"E": "E", "P_PV": "P_PV", "P_GE": "P_GE", "P_FC": "P_FC",
                        "g_GE": "g_GE", "g_FC": "g_FC",
                        "C_load": "C", "H_load": "H", "W_load": "W"}
            feats.append(Y[..., Y_names.index(name_map[key])][..., None])
        elif key in ("Q_dem", "Q_rec", "Q_need"):
            feats.append(aux[key][..., None])
        else:
            raise KeyError(f"Unknown feature key: {key}")
    return np.concatenate(feats, axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------
# Per-variable spec
# ----------------------------------------------------------------------------
VAR_SPECS = {
    "E":       dict(target="E",    filt=None,        zi=False,
                    cond=["T_out", "RH", "v", "h", "d", "w", "s", "regime"]),
    "W":       dict(target="W",    filt=None,        zi=False,
                    cond=["h", "d", "w", "s", "regime"]),
    "C_tilde": dict(target="C",    filt="m_C",       zi=True,
                    cond=["T_out", "RH", "h", "d", "w", "s", "regime"]),
    "H_tilde": dict(target="H",    filt="m_H",       zi=True,
                    cond=["T_out", "h", "d", "w", "s", "regime"]),
    "g_GE":    dict(target="g_GE", filt="a_GE",      zi=True,
                    cond=["E", "P_PV", "h", "T_out", "w"]),
    "g_FC":    dict(target="g_FC", filt="a_FC",      zi=True,
                    cond=["E", "P_PV", "h", "T_out", "w"]),
    "G_AC":    dict(target="G_AC", filt="Q_dem_pos", zi=True,
                    cond=["Q_dem", "Q_rec", "T_out", "s", "regime"]),
    "g_B":     dict(target="g_B",  filt=None,        zi=True,
                    cond=["Q_need", "T_out", "h", "w", "s", "regime"]),
}

# Per-variable hyperparameter overrides (everything else uses defaults below).
# Empty for now; defaults are large enough that all variables converge naturally.
OVERRIDES: dict = {}


def extract_dataset(name: str, data: dict, aux: dict):
    spec = VAR_SPECS[name]
    Y = data["Y"]
    Y_names = data["Y_names"]
    if spec["target"] == "G_AC":
        target = (Y[..., Y_names.index("g_AC1")]
                  + Y[..., Y_names.index("g_AC2")]
                  + Y[..., Y_names.index("g_AC3")])
    else:
        target = Y[..., Y_names.index(spec["target"])]
    if spec["filt"] is None:
        keep = np.ones_like(target, dtype=bool)
    elif spec["filt"] == "Q_dem_pos":
        keep = aux["Q_dem"] > 0
    else:
        keep = data[spec["filt"]].astype(bool)
    X = build_features(spec["cond"], data, aux)
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = target.reshape(-1).astype(np.float32)
    keep_flat = keep.reshape(-1)
    return X_flat[keep_flat], y_flat[keep_flat]


# ----------------------------------------------------------------------------
# Quantile model (monotone spline with K knots) + optional zero head
# ----------------------------------------------------------------------------
class QuantileSplineNet(nn.Module):
    def __init__(self, in_dim: int, K: int = 20, hidden: int = 128,
                 dropout: float = 0.1, zero_inflated: bool = False):
        super().__init__()
        self.K = K
        self.zi = zero_inflated
        out_dim = 1 + K + (1 if zero_inflated else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        base = z[:, 0]
        deltas = F.softplus(z[:, 1:1 + self.K])
        Q_knots = base.unsqueeze(1) + torch.cumsum(deltas, dim=1)
        zero_logit = z[:, -1] if self.zi else None
        return Q_knots, zero_logit


def Q_at_tau(Q_knots: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    K = Q_knots.shape[1]
    pos = tau.clamp(0.0, 1.0) * (K - 1)
    i_lo = pos.floor().long().clamp(0, K - 2)
    frac = (pos - i_lo.float()).unsqueeze(0)
    i_hi = i_lo + 1
    q_lo = Q_knots[:, i_lo]
    q_hi = Q_knots[:, i_hi]
    return q_lo + (q_hi - q_lo) * frac


def cdf_at_y(Q_knots: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    K = Q_knots.shape[1]
    diff = Q_knots - y.unsqueeze(1)
    le = (diff <= 0).long()
    i_lo = (le.sum(dim=1) - 1).clamp(0, K - 2)
    i_hi = i_lo + 1
    q_lo = Q_knots.gather(1, i_lo.unsqueeze(1)).squeeze(1)
    q_hi = Q_knots.gather(1, i_hi.unsqueeze(1)).squeeze(1)
    frac = ((y - q_lo) / (q_hi - q_lo + 1e-9)).clamp(0.0, 1.0)
    below = y <= Q_knots[:, 0]
    above = y >= Q_knots[:, -1]
    tau = (i_lo.float() + frac) / (K - 1)
    tau = torch.where(below, torch.zeros_like(tau), tau)
    tau = torch.where(above, torch.ones_like(tau), tau)
    return tau


def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor, tau: torch.Tensor):
    e = y.unsqueeze(1) - q_pred
    return torch.maximum(tau * e, (tau - 1) * e).mean()


def fast_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC with the rank-sum estimator (no sklearn dependency)."""
    s = scores.detach().cpu().numpy().astype(np.float64)
    y = labels.detach().cpu().numpy().astype(np.int64)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------------
# Per-variable training
# ----------------------------------------------------------------------------
TAU_GRID = torch.linspace(0.025, 0.975, 19)

DEFAULT = dict(
    epochs=200,
    patience=25,
    batch=2048,
    lr=1e-3,
    lr_min=1e-5,
    weight_decay=1e-4,
    K=20,
    hidden=128,
    dropout=0.1,
)


def train_one(name: str, Xtr_np, ytr_np, Xva_np, yva_np, zi: bool) -> dict:
    cfg = {**DEFAULT, **OVERRIDES.get(name, {})}
    t0 = time.time()
    n_tr = len(ytr_np)
    n_va = len(yva_np)

    # ---- feature standardization ----
    feat_mean = Xtr_np.mean(axis=0)
    feat_std = Xtr_np.std(axis=0) + 1e-6
    Xtr = (Xtr_np - feat_mean) / feat_std
    Xva = (Xva_np - feat_mean) / feat_std

    # ---- target scaling (using positive values only) ----
    pos_tr = ytr_np > 0
    if pos_tr.any():
        y_pos_mean = float(ytr_np[pos_tr].mean())
        y_pos_std = max(float(ytr_np[pos_tr].std()), 1e-3)
    else:
        y_pos_mean, y_pos_std = 0.0, 1.0

    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(DEVICE)
    ytr_t = torch.from_numpy(ytr_np.astype(np.float32)).to(DEVICE)
    Xva_t = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    yva_t = torch.from_numpy(yva_np.astype(np.float32)).to(DEVICE)
    yva_zero_lbl = (yva_t == 0).float()
    ytr_zero_lbl = (ytr_t == 0).float()

    in_dim = Xtr_t.shape[1]
    model = QuantileSplineNet(
        in_dim, K=cfg["K"], hidden=cfg["hidden"],
        dropout=cfg["dropout"], zero_inflated=zi,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"], eta_min=cfg["lr_min"])
    tau_grid = TAU_GRID.to(DEVICE)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    best_pin = 0.0
    best_bce = 0.0
    bad = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        # ---- training ----
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        train_loss_sum = train_pin_sum = train_bce_sum = 0.0
        for i in range(0, n_tr, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            xb = Xtr_t.index_select(0, idx)
            yb = ytr_t.index_select(0, idx)
            zb = ytr_zero_lbl.index_select(0, idx)
            Q_knots, zlogit = model(xb)

            pos = yb > 0
            if pos.any():
                Qp = Q_at_tau(Q_knots[pos], tau_grid)
                yp = (yb[pos] - y_pos_mean) / y_pos_std
                Qp_std = (Qp - y_pos_mean) / y_pos_std
                pin = pinball_loss(Qp_std, yp, tau_grid)
            else:
                pin = torch.tensor(0.0, device=DEVICE)

            loss = pin
            if zi:
                bce = F.binary_cross_entropy_with_logits(zlogit, zb)
                loss = loss + bce
            else:
                bce = torch.tensor(0.0, device=DEVICE)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = len(idx)
            train_loss_sum += float(loss.detach()) * bs
            train_pin_sum += float(pin.detach()) * bs
            train_bce_sum += float(bce.detach()) * bs

        train_loss = train_loss_sum / n_tr
        train_pin = train_pin_sum / n_tr
        train_bce = train_bce_sum / n_tr
        sched.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            Q_knots_va, zlogit_va = model(Xva_t)
            pos_va = yva_t > 0
            val_pin = 0.0
            if pos_va.any():
                Qp = Q_at_tau(Q_knots_va[pos_va], tau_grid)
                yp = (yva_t[pos_va] - y_pos_mean) / y_pos_std
                Qp_std = (Qp - y_pos_mean) / y_pos_std
                val_pin = float(pinball_loss(Qp_std, yp, tau_grid))
            val_bce = 0.0
            if zi:
                val_bce = float(F.binary_cross_entropy_with_logits(zlogit_va, yva_zero_lbl))
            val_loss = val_pin + val_bce
        history.append({
            "epoch": epoch,
            "train": train_loss, "train_pin": train_pin, "train_bce": train_bce,
            "val": val_loss, "val_pin": val_pin, "val_bce": val_bce,
            "lr": opt.param_groups[0]["lr"],
        })

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_pin = val_pin
            best_bce = val_bce
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    elapsed = time.time() - t0
    converged_naturally = bad >= cfg["patience"]
    model.load_state_dict(best_state)

    # ---------- A3 gate checks + zero-head calibration ----------
    model.eval()
    with torch.no_grad():
        Q_knots, zlogit = model(Xva_t)

        if zi:
            pi = torch.sigmoid(zlogit)
            mask_zero = yva_zero_lbl > 0.5
            mask_pos = ~mask_zero
            pi_on_zero = float(pi[mask_zero].mean()) if mask_zero.any() else float("nan")
            pi_on_pos = float(pi[mask_pos].mean()) if mask_pos.any() else float("nan")
            pi_max = float(pi.max())
            pi_brier = float(((pi - yva_zero_lbl) ** 2).mean())
            pi_auc = fast_auroc(pi, yva_zero_lbl.long())
        else:
            pi_on_zero = pi_on_pos = pi_max = pi_brier = pi_auc = 0.0

        # quantile monotonicity
        taus = torch.linspace(0.0, 1.0, 21, device=DEVICE)
        Q_grid = Q_at_tau(Q_knots, taus)
        mono_rate = float((Q_grid[:, 1:] >= Q_grid[:, :-1] - 1e-6).all(dim=1).float().mean())

        # CDF-ICDF roundtrip
        rng = np.random.default_rng(0)
        sub_np = rng.integers(0, n_va, size=1000)
        sub_idx = torch.from_numpy(sub_np).long().to(DEVICE)
        u_rand = torch.from_numpy(rng.uniform(0.02, 0.98, size=1000).astype(np.float32)).to(DEVICE)
        Q_sub = Q_knots.index_select(0, sub_idx)
        y_sample = Q_at_tau(Q_sub, u_rand).diagonal()
        u_back = cdf_at_y(Q_sub, y_sample)
        roundtrip_err = float((u_rand - u_back).abs().mean())

    return {
        "name": name,
        "cfg": cfg,
        "n_train": n_tr,
        "n_val": n_va,
        "n_params": n_params,
        "best_val": best_val,
        "best_pin": best_pin,
        "best_bce": best_bce,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "converged_naturally": converged_naturally,
        "elapsed_sec": round(elapsed, 1),
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "y_pos_mean": y_pos_mean,
        "y_pos_std": y_pos_std,
        "pi_max": pi_max,
        "pi_on_zero": pi_on_zero,
        "pi_on_pos": pi_on_pos,
        "pi_brier": pi_brier,
        "pi_auc": pi_auc,
        "monotone_rate": mono_rate,
        "roundtrip_err": roundtrip_err,
        "in_dim": in_dim,
        "K": cfg["K"],
        "hidden": cfg["hidden"],
        "zi": zi,
        "history": history,
        "state": best_state,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print(f"[device] {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print("[load splits]")
    tr = load_split("train")
    va = load_split("val")
    aux_tr = derive_aux(tr["Y"], tr["Y_names"])
    aux_va = derive_aux(va["Y"], va["Y_names"])

    summary_rows = []
    for name in VAR_SPECS.keys():
        print(f"\n=== {name} ===")
        Xtr, ytr = extract_dataset(name, tr, aux_tr)
        Xva, yva = extract_dataset(name, va, aux_va)
        print(f"  train rows: {len(ytr):>7}   val rows: {len(yva):>6}   "
              f"feat dim: {Xtr.shape[1]}   y>0 rate: {(ytr > 0).mean():.3f}")
        res = train_one(name, Xtr, ytr, Xva, yva, zi=VAR_SPECS[name]["zi"])

        out = {
            "state_dict": res.pop("state"),
            "spec": {
                "name": name,
                "in_dim": res["in_dim"],
                "K": res["K"],
                "hidden": res["hidden"],
                "zi": res["zi"],
                "cond_keys": VAR_SPECS[name]["cond"],
                "target": VAR_SPECS[name]["target"],
                "filter": VAR_SPECS[name]["filt"],
            },
            "feat_mean": np.array(res["feat_mean"], dtype=np.float32),
            "feat_std": np.array(res["feat_std"], dtype=np.float32),
            "y_pos_mean": res["y_pos_mean"],
            "y_pos_std": res["y_pos_std"],
            "history": res["history"],
            "cfg": res["cfg"],
        }
        ckpt_path = CKPT_DIR / f"{name}.pt"
        torch.save(out, ckpt_path)

        tag_conv = "✓" if res["converged_naturally"] else "× (epoch cap hit)"
        print(f"  saved {ckpt_path}")
        print(f"  params={res['n_params']}  best_val={res['best_val']:.4f}"
              f"  (pin={res['best_pin']:.4f}  bce={res['best_bce']:.4f})"
              f"  best_epoch={res['best_epoch']}/{res['epochs_run']}"
              f"  time={res['elapsed_sec']}s  converged={tag_conv}")
        print(f"  quantile: monotone_rate={res['monotone_rate']:.4f}  "
              f"roundtrip_err={res['roundtrip_err']:.2e}")
        if res["zi"]:
            print(f"  zero-head: pi_max={res['pi_max']:.4f}  "
                  f"pi_on_zero={res['pi_on_zero']:.4f}  pi_on_pos={res['pi_on_pos']:.4f}  "
                  f"brier={res['pi_brier']:.4f}  auc={res['pi_auc']:.4f}")

        summary_rows.append({
            "name": name,
            "n_train": res["n_train"], "n_val": res["n_val"],
            "n_params": res["n_params"],
            "best_val": res["best_val"],
            "best_pin": res["best_pin"], "best_bce": res["best_bce"],
            "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "converged": res["converged_naturally"],
            "time_s": res["elapsed_sec"],
            "monotone_rate": res["monotone_rate"],
            "roundtrip_err": res["roundtrip_err"],
            "pi_max": res["pi_max"],
            "pi_on_zero": res["pi_on_zero"],
            "pi_on_pos": res["pi_on_pos"],
            "pi_brier": res["pi_brier"],
            "pi_auc": res["pi_auc"],
            "zi": res["zi"],
        })

    df = pd.DataFrame(summary_rows)
    out_csv = CKPT_DIR / "marginal_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[summary] -> {out_csv}")
    print(df.to_string(index=False))

    report = {
        "device": str(DEVICE),
        "defaults": DEFAULT,
        "overrides": OVERRIDES,
        "thresholds": {
            "E_W_pi_max_max": 0.01,
            "monotone_rate_min": 0.99,
            "roundtrip_err_max": 0.01,
            "pi_brier_max": 0.10,
            "pi_auc_min": 0.85,
        },
        "results": summary_rows,
    }
    with open(CKPT_DIR / "marginal_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] -> {CKPT_DIR / 'marginal_report.json'}")


if __name__ == "__main__":
    main()
