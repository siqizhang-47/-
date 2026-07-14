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
from figures import fig_forecast, fig_attention
from timexer_nsdiff_adapter import EXO_TOKENS, TARGET_NAMES


def _default_cfg(device):
    return dict(horizon=24, seq_len=168, d_model=128, patch_len=24, n_heads=8,
                e_layers=2, d_ff=512, dropout=0.1, d_x=128, diffusion_steps=20,
                rolling_length=96, beta_schedule="linear", beta_start=1e-4,
                beta_end=1e-2, device=device)


def _build_model(cfg):
    return TimeXerNsDiff(**cfg)


def train(device="cuda:4", data_root="./data/IES", out_dir="./results",
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
                        "target_scaler": {"mean": meta["target_scaler"].mean,
                                          "std": meta["target_scaler"].std},
                        "target_names": meta["target_names"]},
                       os.path.join(out_dir, "best.pt"))
            print(f"  saved best (val={best_val:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {ep+1}"); break
    return os.path.join(out_dir, "best.pt")


@torch.no_grad()
def evaluate(device="cuda:4", data_root="./data/IES", out_dir="./results",
             ckpt="./results/best.pt", n_samples=100, batch_size=64, num_workers=4,
             es_vs_windows=512, fig_windows=6, max_test_batches=None):
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]; cfg["device"] = device
    model = _build_model(cfg).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    tmean = np.asarray(ck["target_scaler"]["mean"]); tstd = np.asarray(ck["target_scaler"]["std"])
    tnames = ck.get("target_names", TARGET_NAMES)

    loaders, _ = build_dataloaders(data_root, batch_size=batch_size, num_workers=num_workers)
    rm = RunningMetrics(n_targets=len(tnames))

    es_std, es_truth_std = [], []            # standardised samples for ES/VS
    fig_s, fig_t = [], []                    # raw samples/truth for the fan chart
    attn_sum, attn_cnt = None, 0

    def inv(x):                              # standardised -> raw
        return x * tstd + tmean

    loader = loaders["test"]
    with tqdm(total=max_test_batches or len(loader), desc="eval-sampling") as bar:
        for i, b in enumerate(loader):
            if max_test_batches and i >= max_test_batches:
                break
            he = b["history_energy"].to(device)
            fc = b["future_calendar"].to(device); fw = b["future_weather"].to(device)
            y0, mu, attn = model.sample(he, fc, fw, n_samples=n_samples, return_attention=True)
            s_std = y0.cpu().numpy()                                  # [b,S,H,K]
            t_std = b["future_energy"].numpy()                       # [b,H,K]
            s_raw = inv(s_std); t_raw = b["future_energy_raw"].numpy()
            rm.update(s_raw, t_raw)

            a = attn.mean(1).cpu().numpy()                           # [b,4,16] mean over heads
            attn_sum = a.sum(0) if attn_sum is None else attn_sum + a.sum(0)
            attn_cnt += a.shape[0]

            if len(es_std) * s_std.shape[0] < es_vs_windows:
                es_std.append(s_std); es_truth_std.append(t_std)
            if len(fig_s) * s_raw.shape[0] < fig_windows:
                fig_s.append(s_raw); fig_t.append(t_raw)
            bar.update(1)

    res = rm.finalize()

    es_std = np.concatenate(es_std, 0)[:es_vs_windows]
    es_truth_std = np.concatenate(es_truth_std, 0)[:es_vs_windows]
    res["ES"] = energy_score(es_std, es_truth_std)
    res["VS"] = variogram_score(es_std, es_truth_std)
    res = {k: float(v) for k, v in res.items()}

    # figures
    fs = np.concatenate(fig_s, 0); ft = np.concatenate(fig_t, 0)
    fig_forecast(fs, ft, tnames, os.path.join(out_dir, "fig1_forecast.png"), window_index=0)
    attn_mean = attn_sum / max(attn_cnt, 1)                          # [4,16]
    fig_attention(attn_mean, EXO_TOKENS, tnames, os.path.join(out_dir, "fig2_attention.png"))

    # one-line metrics
    order = ["MAE", "RMSE", "sMAPE", "CRPS", "QICE",
             "PICP50", "PICP80", "PICP90", "PICP95", "MIW90", "ES", "VS"]
    line = "  ".join(f"{k}={res[k]:.4f}" for k in order)
    print("\n[Oracle Weather] " + line)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("[Oracle Weather] " + line + "\n")
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
                "es_vs_windows", "fig_windows", "max_test_batches"}}
    return evaluate(device=device, ckpt=ckpt, **eval_kw)


if __name__ == "__main__":
    import fire
    fire.Fire({"train": train, "evaluate": evaluate, "train_eval": train_eval})
