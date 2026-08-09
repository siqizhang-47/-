"""Test-set scenario generation, metric computation and attention export.

IMPORTANT: every number produced here is an **Oracle Weather** result -- the
model reads the *true* future weather (design document, section 3 and 24.6).
The setting is stamped into every JSON file so it can never be mistaken for an
operational forecast score.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm

from ..metrics.prob_metrics import TARGET_NAMES, compute_all_metrics
from .trainer import make_loader, to_device


@torch.no_grad()
def generate_scenarios(model, dataset, cfg, device, stride: int = 1, desc: str = "test"):
    """Returns pred [W, S, H, K] and true [W, H, K] on the ORIGINAL scale, plus group info."""
    if stride > 1:
        dataset = Subset(dataset, list(range(0, len(dataset), stride)))
    loader = make_loader(dataset, cfg.eval_batch_size, False, cfg.num_workers)

    model.eval()
    preds, trues, groups = [], [], []
    for batch in tqdm(loader, desc=f"sampling[{desc}]", unit="batch"):
        batch = to_device(batch, device)
        samples = model.sample(batch, n_samples=cfg.num_samples, chunk=cfg.sample_chunk)
        preds.append(samples.cpu())
        trues.append(batch["future_energy"].cpu())
        groups.append(batch["group_info"].cpu())
    return torch.cat(preds), torch.cat(trues), torch.cat(groups)


def inverse_transform(pred, true, target_scaler):
    """[W, S, H, K] and [W, H, K] back to physical units."""
    mean = torch.as_tensor(target_scaler.mean_, dtype=pred.dtype)
    std = torch.as_tensor(target_scaler.std_, dtype=pred.dtype)
    return pred * std + mean, true * std + mean


@torch.no_grad()
def export_attention(model, dataset, cfg, device, out_path, max_batches: int = 20):
    """Average the cross-attention map over batch and heads -> [4, 16] heat map."""
    if not model.cfg.use_exog:
        return None
    loader = make_loader(dataset, cfg.eval_batch_size, False, cfg.num_workers)
    model.eval()
    acc, n = None, 0
    for i, batch in enumerate(tqdm(loader, desc="attention", unit="batch", total=max_batches)):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        cond = model.encode(batch)
        attn = cond["attention"]                       # [B, heads, n_query, n_exo]
        if attn is None:
            return None
        a = attn.mean(dim=(0, 1)).cpu().numpy()        # [n_query, n_exo]
        acc = a if acc is None else acc + a
        n += 1
    if acc is None:
        return None
    attn_mean = acc / n
    names = model.condition_encoder.token_names
    queries = (["Shared"] if model.cfg.shared_global_token else TARGET_NAMES)
    payload = {"query_names": queries, "exo_token_names": names,
               "attention": attn_mean.tolist()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    _plot_attention(attn_mean, queries, names, out_path.replace(".json", ".png"))
    return payload


def _plot_attention(attn, rows, cols, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[attention] matplotlib unavailable ({exc}); PNG skipped")
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(cols)), 1.1 * len(rows) + 1.6))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title("TimeXer target -> exogenous cross-attention (batch/head mean)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def evaluate(model, dataset, cfg, device, target_scaler, out_dir, tag="test"):
    pred, true, groups = generate_scenarios(model, dataset, cfg, device,
                                            stride=cfg.test_stride, desc=tag)
    pred, true = inverse_transform(pred, true, target_scaler)

    dev = device if device.type == "cuda" else torch.device("cpu")
    metrics = compute_all_metrics(pred.to(dev), true.to(dev), group_info=groups.to(dev),
                                  variogram_mode=cfg.variogram_mode,
                                  variogram_p=cfg.variogram_p)
    metrics["weather_setting"] = "OracleWeather"
    metrics["ablation"] = cfg.ablation
    metrics["seed"] = cfg.seed
    metrics["split"] = tag

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if cfg.save_samples:
        np.savez_compressed(os.path.join(out_dir, f"samples_{tag}.npz"),
                            pred=pred.numpy().astype(np.float32),
                            true=true.numpy().astype(np.float32))
    return metrics


def print_metrics(metrics, logger=print):
    logger("\n" + "=" * 78)
    logger(f"  RESULTS [{metrics['ablation']} | seed {metrics['seed']} | "
           f"{metrics['weather_setting']}]  windows={metrics['n_windows']} "
           f"samples={metrics['n_samples']}")
    logger("=" * 78)
    header = f"  {'target':<12}{'MAE':>11}{'RMSE':>11}{'sMAPE%':>10}{'CRPS':>11}{'QICE':>9}{'PICP@90':>9}"
    logger(header)
    for name, m in metrics["per_target"].items():
        logger(f"  {name:<12}{m['MAE']:>11.3f}{m['RMSE']:>11.3f}{m['sMAPE']:>10.2f}"
               f"{m['CRPS']:>11.3f}{m['QICE']:>9.4f}{m['PICP@90']:>9.3f}")
    o = metrics["overall"]
    logger(f"  {'MEAN':<12}{o['MAE']:>11.3f}{o['RMSE']:>11.3f}{o['sMAPE']:>10.2f}"
           f"{o['CRPS']:>11.3f}{o['QICE']:>9.4f}{o['PICP@90']:>9.3f}")
    logger(f"  EnergyScore={o['EnergyScore']:.4f}  "
           f"VariogramScore({o['variogram_mode']})={o['VariogramScore']:.4f}")
    if "conditional" in metrics:
        logger("  -- conditional slices (CRPS / PICP@90) --")
        for k, v in metrics["conditional"].items():
            logger(f"     {k:<24} n={v['n']:>7}  CRPS={v['CRPS']:>10.3f}  PICP@90={v['PICP@90']:.3f}")
    logger("=" * 78 + "\n")
