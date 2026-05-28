"""
A3. Train conditional marginal distributions for 8 generation variables.

Per the PS-BCD v4.1 plan (Step 3 + A3 table):

variable     | type           | filter                | conditions
-------------+----------------+-----------------------+----------------------------------
E            | pure quantile  | none                  | T_out,RH,v,h,d,w,s,regime
W            | pure quantile  | none                  | h,d,w,s,regime
C_tilde      | zero-inflated  | m_C==1                | T_out,RH,h,d,w,s,regime
H_tilde      | zero-inflated  | m_H==1                | T_out,h,d,w,s,regime
g_GE         | zero-inflated  | a_GE==1               | E,P_PV,h,T_out,w
g_FC         | zero-inflated  | a_FC==1               | E,P_PV,h,T_out,w
G_AC         | zero-inflated  | Q_dem>0               | Q_dem,Q_rec,T_out,s,regime
g_B          | zero-inflated  | none                  | Q_need,T_out,h,w,s,regime

Quantile parameterization: monotone spline with K knots at tau_k = k/(K-1).
The network outputs (base, delta_1, ..., delta_K) where delta_k >= 0 (softplus),
giving Q(tau_k | x) = base + sum_{j<=k} delta_j.  Quantile values between knots
are obtained by linear interpolation, guaranteeing monotone CDF.

CPU-friendly: hidden=128, K=20 knots, batch=1024, AdamW, early stopping.

Outputs:
    ckpt/marginals/{E,W,C_tilde,H_tilde,g_GE,g_FC,G_AC,g_B}.pt
    ckpt/marginals/marginal_summary.csv
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
KAPPA_GE = 0.5      # GE waste-heat recovery fraction
KAPPA_FC = 0.5      # FC waste-heat recovery fraction
RHO_H = 1.0         # heat-load fraction borne by boiler (initial guess)

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
    """regime_hr: (N,24) int8 with values 1,2,3 -> (N,24,3) float32"""
    out = np.zeros(regime_hr.shape + (3,), dtype=np.float32)
    out[..., 0] = (regime_hr == 1)
    out[..., 1] = (regime_hr == 2)
    out[..., 2] = (regime_hr == 3)
    return out


def build_features(spec_keys: list, data: dict, aux: dict) -> np.ndarray:
    """Build feature matrix (N,24,F) given a list of condition keys."""
    Y = data["Y"]
    C = data["C"]
    Y_names = data["Y_names"]
    C_names = data["C_names"]
    regime = data["regime"]  # (N,)

    # Broadcast day-level regime to (N,24)
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
            # raw Y columns (mapped names below)
            name_map = {"E": "E", "P_PV": "P_PV", "P_GE": "P_GE", "P_FC": "P_FC",
                        "g_GE": "g_GE", "g_FC": "g_FC",
                        "C_load": "C", "H_load": "H", "W_load": "W"}
            feats.append(Y[..., Y_names.index(name_map[key])][..., None])
        elif key in ("Q_dem", "Q_rec", "Q_need"):
            feats.append(aux[key][..., None])
        else:
            raise KeyError(f"Unknown feature key: {key}")
    return np.concatenate(feats, axis=-1).astype(np.float32)  # (N,24,F)


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


def extract_dataset(name: str, data: dict, aux: dict):
    spec = VAR_SPECS[name]
    Y = data["Y"]
    Y_names = data["Y_names"]
    # ----- target -----
    if spec["target"] == "G_AC":
        target = (Y[..., Y_names.index("g_AC1")]
                  + Y[..., Y_names.index("g_AC2")]
                  + Y[..., Y_names.index("g_AC3")])
    else:
        target = Y[..., Y_names.index(spec["target"])]
    # ----- filter mask -----
    if spec["filt"] is None:
        keep = np.ones_like(target, dtype=bool)
    elif spec["filt"] == "Q_dem_pos":
        keep = aux["Q_dem"] > 0
    else:
        keep = data[spec["filt"]].astype(bool)
    # ----- features -----
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
        deltas = F.softplus(z[:, 1:1 + self.K])           # (B, K) >= 0
        Q_knots = base.unsqueeze(1) + torch.cumsum(deltas, dim=1)
        zero_logit = z[:, -1] if self.zi else None
        return Q_knots, zero_logit


def Q_at_tau(Q_knots: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Q_knots: (B,K), tau: (T,) in [0,1].  Returns (B,T) via linear interp."""
    K = Q_knots.shape[1]
    pos = tau.clamp(0.0, 1.0) * (K - 1)
    i_lo = pos.floor().long().clamp(0, K - 2)
    frac = (pos - i_lo.float()).unsqueeze(0)            # (1,T)
    i_hi = i_lo + 1
    q_lo = Q_knots[:, i_lo]                              # (B,T)
    q_hi = Q_knots[:, i_hi]
    return q_lo + (q_hi - q_lo) * frac


def cdf_at_y(Q_knots: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Invert: given y (B,), return u = F(y|x) in [0,1]."""
    K = Q_knots.shape[1]
    diff = Q_knots - y.unsqueeze(1)                      # (B,K)
    # last index where Q_knot[i] <= y
    le = (diff <= 0).long()
    i_lo = (le.sum(dim=1) - 1).clamp(0, K - 2)
    i_hi = i_lo + 1
    q_lo = Q_knots.gather(1, i_lo.unsqueeze(1)).squeeze(1)
    q_hi = Q_knots.gather(1, i_hi.unsqueeze(1)).squeeze(1)
    frac = ((y - q_lo) / (q_hi - q_lo + 1e-9)).clamp(0.0, 1.0)
    # below/above support
    below = y <= Q_knots[:, 0]
    above = y >= Q_knots[:, -1]
    tau = (i_lo.float() + frac) / (K - 1)
    tau = torch.where(below, torch.zeros_like(tau), tau)
    tau = torch.where(above, torch.ones_like(tau), tau)
    return tau


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def pinball_loss(q_pred: torch.Tensor, y: torch.Tensor, tau: torch.Tensor):
    """q_pred: (B,T), y: (B,), tau: (T,).  Returns scalar mean pinball."""
    e = y.unsqueeze(1) - q_pred
    return torch.maximum(tau * e, (tau - 1) * e).mean()


# ----------------------------------------------------------------------------
# Per-variable training
# ----------------------------------------------------------------------------
TAU_GRID = torch.linspace(0.025, 0.975, 19)  # 19 quantile levels for training


def train_one(name: str, Xtr_np, ytr_np, Xva_np, yva_np, zi: bool,
              epochs: int = 60, patience: int = 8, batch: int = 4096,
              lr: float = 1e-3, weight_decay: float = 1e-4,
              K: int = 20, hidden: int = 128) -> dict:
    t0 = time.time()
    n_tr = len(ytr_np)
    n_va = len(yva_np)

    # Feature standardization (computed from train)
    feat_mean = Xtr_np.mean(axis=0)
    feat_std = Xtr_np.std(axis=0) + 1e-6
    Xtr = (Xtr_np - feat_mean) / feat_std
    Xva = (Xva_np - feat_mean) / feat_std

    # Target scaling: standardize y (in original units).  We center to make the
    # network output near zero; we don't apply log so we can ICDF back trivially.
    y_pos_mean = float(ytr_np[ytr_np > 0].mean()) if (ytr_np > 0).any() else 1.0
    y_pos_std = float(ytr_np[ytr_np > 0].std()) if (ytr_np > 0).any() else 1.0
    y_pos_std = max(y_pos_std, 1e-3)

    # Convert to tensors (move full dataset to GPU; <100MB per variable)
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(DEVICE)
    ytr_t = torch.from_numpy(ytr_np.astype(np.float32)).to(DEVICE)
    Xva_t = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
    yva_t = torch.from_numpy(yva_np.astype(np.float32)).to(DEVICE)

    in_dim = Xtr_t.shape[1]
    model = QuantileSplineNet(in_dim, K=K, hidden=hidden, zero_inflated=zi).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tau_grid = TAU_GRID.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        train_loss = 0.0
        n_batches = 0
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            xb = Xtr_t.index_select(0, idx)
            yb = ytr_t.index_select(0, idx)
            Q_knots, zlogit = model(xb)

            # pinball on positive subset
            pos = yb > 0
            if pos.any():
                Qp = Q_at_tau(Q_knots[pos], tau_grid)            # (B+, T)
                yp = (yb[pos] - y_pos_mean) / y_pos_std
                # Quantiles also work in standardized units
                Qp_std = (Qp - y_pos_mean) / y_pos_std
                pin = pinball_loss(Qp_std, yp, tau_grid)
            else:
                pin = torch.tensor(0.0, device=DEVICE)

            loss = pin
            if zi:
                z_lbl = (yb == 0).float()
                bce = F.binary_cross_entropy_with_logits(zlogit, z_lbl)
                loss = loss + bce

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.detach()) * len(idx)
            n_batches += 1
        train_loss /= n_tr

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
                z_lbl = (yva_t == 0).float()
                val_bce = float(F.binary_cross_entropy_with_logits(zlogit_va, z_lbl))
            val_loss = val_pin + val_bce
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "val_pin": val_pin, "val_bce": val_bce})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    # ---- validation checks per A3 ----
    model.eval()
    with torch.no_grad():
        Q_knots, zlogit = model(Xva_t)
        # (a) zero probability for E should be everywhere < 0.01 (info)
        if zi:
            pi_zero = torch.sigmoid(zlogit).detach().cpu().numpy()
        else:
            pi_zero = np.zeros(n_va, dtype=np.float32)
        # (b) quantile monotonicity check: per-sample monotone in tau
        taus = torch.linspace(0.0, 1.0, 21, device=DEVICE)
        Q_grid = Q_at_tau(Q_knots, taus)                        # (B, 21)
        mono_rate = float((Q_grid[:, 1:] >= Q_grid[:, :-1] - 1e-6).all(dim=1).float().mean())
        # (c) CDF-ICDF roundtrip on 1000 random points
        rng = np.random.default_rng(0)
        sub_np = rng.integers(0, n_va, size=1000)
        sub_idx = torch.from_numpy(sub_np).long().to(DEVICE)
        u_rand = torch.from_numpy(rng.uniform(0.02, 0.98, size=1000).astype(np.float32)).to(DEVICE)
        Q_sub = Q_knots.index_select(0, sub_idx)
        y_sample = Q_at_tau(Q_sub, u_rand).diagonal()  # (1000,)
        u_back = cdf_at_y(Q_sub, y_sample)             # (1000,)
        roundtrip_err = float((u_rand - u_back).abs().mean())

    return {
        "name": name,
        "n_train": n_tr,
        "n_val": n_va,
        "n_params": n_params,
        "best_val": best_val,
        "epochs_run": len(history),
        "elapsed_sec": round(elapsed, 1),
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "y_pos_mean": y_pos_mean,
        "y_pos_std": y_pos_std,
        "pi_zero_max": float(pi_zero.max() if len(pi_zero) else 0.0),
        "monotone_rate": mono_rate,
        "roundtrip_err": roundtrip_err,
        "in_dim": in_dim,
        "K": K,
        "hidden": hidden,
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
        res = train_one(name, Xtr, ytr, Xva, yva,
                        zi=VAR_SPECS[name]["zi"])
        # Save checkpoint
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
        }
        ckpt_path = CKPT_DIR / f"{name}.pt"
        torch.save(out, ckpt_path)
        print(f"  saved {ckpt_path}  "
              f"params={res['n_params']}  best_val={res['best_val']:.4f}  "
              f"epochs={res['epochs_run']}  time={res['elapsed_sec']}s")
        print(f"  checks: pi_zero_max={res['pi_zero_max']:.4f}  "
              f"monotone_rate={res['monotone_rate']:.4f}  "
              f"roundtrip_err={res['roundtrip_err']:.6f}")
        summary_rows.append({
            "name": name,
            "n_train": res["n_train"], "n_val": res["n_val"],
            "n_params": res["n_params"], "epochs": res["epochs_run"],
            "time_s": res["elapsed_sec"], "best_val": res["best_val"],
            "pi_zero_max": res["pi_zero_max"],
            "monotone_rate": res["monotone_rate"],
            "roundtrip_err": res["roundtrip_err"],
            "zi": res["zi"],
        })

    df = pd.DataFrame(summary_rows)
    out_csv = CKPT_DIR / "marginal_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[summary] -> {out_csv}")
    print(df.to_string(index=False))

    # A3 gate report (interpret per the plan thresholds)
    report = {
        "thresholds": {
            "pi_zero_max_for_E_W": 0.01,
            "monotone_rate_min": 0.99,
            "roundtrip_err_max": 0.01,
        },
        "results": summary_rows,
    }
    with open(CKPT_DIR / "marginal_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] -> {CKPT_DIR / 'marginal_report.json'}")


if __name__ == "__main__":
    main()
