"""Two figures for the TimeXer-NsDiff experiment.

Fig.1  probabilistic forecast fan chart (median + 50%/90% bands vs actual) for a
       sample test day, one subplot per target.
Fig.2  cross-attention heat-map [4 targets x 16 exogenous variables] (plan §23).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_forecast(samples_raw, truth_raw, target_names, out_path, window_index=0):
    """samples_raw [W,S,H,K], truth_raw [W,H,K]."""
    w = window_index
    S, H, K = samples_raw.shape[1], samples_raw.shape[2], samples_raw.shape[3]
    x = np.arange(H)
    med = np.percentile(samples_raw[w], 50, axis=0)
    lo50 = np.percentile(samples_raw[w], 25, axis=0);  hi50 = np.percentile(samples_raw[w], 75, axis=0)
    lo90 = np.percentile(samples_raw[w], 5, axis=0);   hi90 = np.percentile(samples_raw[w], 95, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for k, ax in enumerate(axes.ravel()):
        ax.fill_between(x, lo90[:, k], hi90[:, k], color="#4a7fb0", alpha=0.20, label="90% interval")
        ax.fill_between(x, lo50[:, k], hi50[:, k], color="#4a7fb0", alpha=0.38, label="50% interval")
        ax.plot(x, med[:, k], color="#1f4e79", lw=1.8, label="Median forecast")
        ax.plot(x, truth_raw[w, :, k], color="#d1622b", lw=1.8, label="Actual")
        ax.set_title(target_names[k]); ax.set_xlabel("Forecast hour"); ax.set_ylabel("Value")
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("TimeXer-NsDiff: probabilistic forecast (Oracle Weather)")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def fig_attention(attn_4x16, token_names, target_names, out_path):
    """attn_4x16: [4,16] mean attention weights."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(attn_4x16, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(token_names))); ax.set_xticklabels(token_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(target_names))); ax.set_yticklabels(target_names)
    for i in range(attn_4x16.shape[0]):
        for j in range(attn_4x16.shape[1]):
            ax.text(j, i, f"{attn_4x16[i, j]:.2f}", ha="center", va="center",
                    color="white" if attn_4x16[i, j] < attn_4x16.max() * 0.6 else "black", fontsize=6)
    ax.set_title("Target ← exogenous cross-attention (mean over test & heads)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
