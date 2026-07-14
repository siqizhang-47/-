"""
TimeXer-NsDiff: train / evaluate entry point.

    export PYTHONPATH=./
    # train on GPU 4, then evaluate (one-line metrics + 2 figures)
    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 main.py train_eval --device=cuda:4

    # evaluate only from a saved checkpoint
    python3 main.py evaluate --device=cuda:4 --ckpt=results/best.pt

Outputs (results/):
    best.pt                 trained checkpoint (weights + config + scalers)
    metrics.txt / .json     MAE RMSE sMAPE CRPS QICE PICP50/80/90/95 MIW90 ES VS
    fig1_forecast.png       probabilistic fan chart (sample test day)
    fig2_attention.png      target <- exogenous cross-attention heat-map
"""
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from data_ies import build_dataloaders
from model import TimeXerNsDiff
from metrics import RunningMetrics, energy_score, variogram_score, COVERAGE_LEVELS
from figures import fig_pdf_kde, fig_timeseries, fig_correlation
from timexer_nsdiff_adapter import TARGET_NAMES


def _default_cfg(device):
    return dict(horizon=24, seq_len=168, d_model=128, patch_len=24, n_heads=8,
                e_layers=2, d_ff=512, dropout=0.1, d_x=128, diffusion_steps=20,
                rolling_length=96, beta_schedule="linear", beta_start=1e-4,
                beta_end=1e-2, device=device)


def _build_model(cfg):
    return TimeXerNsDiff(**cfg)


def train(device="cuda:4", data_root="./data/IES/aligned_energy_weather_summary.xlsx",
          out_dir="./results",
          batch_size=64, epochs=40, lr=1e-3, weight_decay=1e-4, patience=8,
          grad_clip=1.0, num_workers=4, max_train_batches=None, max_val_batches=None):
    os.makedirs(out_dir, exist_ok=True)
    loaders, meta = build_dataloaders(data_root, batch_size=batch_size, num_workers=num_workers)
    cfg = _default_cfg(device)
    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] parameters: {n_params/1e6:.2f}M  device={device}")

    def run_epoch(loader, train_mode, cap):
        model.train(train_mode)
        tot, nb = 0.0, 0
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        desc = "train" if train_mode else "val"
        with ctx, tqdm(total=cap or len(loader), desc=desc, leave=False) as bar:
            for i, b in enumerate(loader):
                if cap and i >= cap:
                    break
                he = b["history_energy"].to(device); fe = b["future_energy"].to(device)
                fc = b["future_calendar"].to(device); fw = b["future_weather"].to(device)
                loss, parts = model.train_loss(he, fc, fw, fe)
                if train_mode:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                tot += float(loss.detach()); nb += 1
                bar.update(1); bar.set_postfix(loss=f"{float(loss.detach()):.3f}", **parts)
        return tot / max(nb, 1)

    best_val, bad = float("inf"), 0
    for ep in range(epochs):
        tr = run_epoch(loaders["train"], True, max_train_batches)
        va = run_epoch(loaders["val"], False, max_val_batches)
        print(f"epoch {ep+1}/{epochs}  train {tr:.4f}  val {va:.4f}")
        if va < best_val - 1e-5:
            best_val, bad = va, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                        "target_scaler": {"min": meta["target_scaler"].min,
                                          "range": meta["target_scaler"].range},
                        "target_names": meta["target_names"]},
                       os.path.join(out_dir, "best.pt"))
            print(f"  saved best (val={best_val:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {ep+1}"); break
    return os.path.join(out_dir, "best.pt")


@torch.no_grad()
def evaluate(device="cuda:4", data_root="./data/IES/aligned_energy_weather_summary.xlsx",
             out_dir="./results", ckpt="./results/best.pt", n_samples=100, batch_size=64,
             num_workers=4, es_vs_windows=512, ts_days=7, kde_per_batch=1500,
             max_test_batches=None):
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]; cfg["device"] = device
    model = _build_model(cfg).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    tnames = ck.get("target_names", TARGET_NAMES)
    K = len(tnames)
    H = cfg["horizon"]
    rng = np.random.default_rng(0)

    loaders, _ = build_dataloaders(data_root, batch_size=batch_size, num_workers=num_workers)
    rm = RunningMetrics(n_targets=K)

    # everything below is in the NORMALISED [0,1] space (targets are min-max scaled)
    es_n, es_truth_n = [], []                  # normalised samples for ES/VS
    gmean_all, truth_all = [], []              # generated mean & truth (correlation)
    pred_pool = {k: [] for k in range(K)}      # predicted sample values (KDE)
    act_pool = {k: [] for k in range(K)}       # actual values (KDE)
    ts_s, ts_t = [], []                        # consecutive-day samples/truth (time series)
    g = 0                                      # global window counter

    loader = loaders["test"]
    with tqdm(total=max_test_batches or len(loader), desc="eval-sampling") as bar:
        for i, b in enumerate(loader):
            if max_test_batches and i >= max_test_batches:
                break
            he = b["history_energy"].to(device)
            fc = b["future_calendar"].to(device); fw = b["future_weather"].to(device)
            y0, mu = model.sample(he, fc, fw, n_samples=n_samples)
            s_n = y0.cpu().numpy()                                    # [b,S,H,K] normalised
            t_n = b["future_energy"].numpy()                         # [b,H,K]   normalised
            rm.update(s_n, t_n)

            gmean_all.append(s_n.mean(1).reshape(-1, K))             # [b*H,K]
            truth_all.append(t_n.reshape(-1, K))
            for k in range(K):
                pv = s_n[:, :, :, k].reshape(-1)
                av = t_n[:, :, k].reshape(-1)
                pred_pool[k].append(rng.choice(pv, size=min(kde_per_batch, pv.size), replace=False))
                act_pool[k].append(rng.choice(av, size=min(kde_per_batch, av.size), replace=False))

            if len(es_n) * s_n.shape[0] < es_vs_windows:
                es_n.append(s_n); es_truth_n.append(t_n)

            bsz = s_n.shape[0]
            for j in range(bsz):
                if (g + j) % H == 0 and len(ts_s) < ts_days:
                    ts_s.append(s_n[j]); ts_t.append(t_n[j])
            g += bsz
            bar.update(1)

    res = rm.finalize()
    es_n = np.concatenate(es_n, 0)[:es_vs_windows]
    es_truth_n = np.concatenate(es_truth_n, 0)[:es_vs_windows]
    res["ES"] = energy_score(es_n, es_truth_n)              # normalised [0,1]
    res["VS"] = variogram_score(es_n, es_truth_n)           # normalised [0,1]
    res = {k: float(v) for k, v in res.items()}

    # figures
    pred_pool = {k: np.concatenate(v) for k, v in pred_pool.items()}
    act_pool = {k: np.concatenate(v) for k, v in act_pool.items()}
    fig_pdf_kde(pred_pool, act_pool, tnames, os.path.join(out_dir, "fig1_pdf_kde.png"))
    if len(ts_s) >= 1:
        fig_timeseries(np.stack(ts_s), np.stack(ts_t), tnames,
                       os.path.join(out_dir, "fig2_timeseries.png"))
    gm = np.concatenate(gmean_all, 0); tr = np.concatenate(truth_all, 0)
    fig_correlation(gm, tr, tnames, os.path.join(out_dir, "fig3_correlation.png"))

    # one-line metrics
    order = ["MAE", "RMSE", "sMAPE", "CRPS", "QICE",
             "PICP50", "PICP80", "PICP90", "PICP95", "MIW90", "ES", "VS"]
    line = "  ".join(f"{k}={res[k]:.4f}" for k in order)
    tag = "[Oracle Weather | normalised 0-1] "
    print("\n" + tag + line)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(tag + line + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"figures + metrics saved under {out_dir}")
    return res


def train_eval(device="cuda:4", **kw):
    train_kw = {k: v for k, v in kw.items() if k in
                {"data_root", "out_dir", "batch_size", "epochs", "lr", "weight_decay",
                 "patience", "grad_clip", "num_workers", "max_train_batches", "max_val_batches"}}
    ckpt = train(device=device, **train_kw)
    eval_kw = {k: v for k, v in kw.items() if k in
               {"data_root", "out_dir", "n_samples", "batch_size", "num_workers",
                "es_vs_windows", "ts_days", "kde_per_batch", "max_test_batches"}}
    return evaluate(device=device, ckpt=ckpt, **eval_kw)


if __name__ == "__main__":
    import fire
    fire.Fire({"train": train, "evaluate": evaluate, "train_eval": train_eval})
