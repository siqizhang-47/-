import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['Electrical', 'Cooling', 'Heating', 'PV']


def fig1_real_vs_gen(y, s, path, w=0):           # 一条窗口:真值 vs 生成均值
    O, D = y.shape[1], y.shape[2]; fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for d in range(D):
        a = ax[d // 2, d % 2]; a.plot(y[w, :, d], 'r-', label='Real')
        a.plot(s[w].mean(0)[:, d], 'b-', label='Generated'); a.set_title(NAMES[d]); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig2_corr(r_real, r_gen, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, (r, t) in enumerate([(r_gen, 'Generated'), (r_real, 'Real')]):
        im = ax[k].imshow(r, vmin=-1, vmax=1, cmap='RdBu_r')
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(NAMES)
        ax[k].set_yticks(range(4)); ax[k].set_yticklabels(NAMES); ax[k].set_title(t)
        for i in range(4):
            for j in range(4):
                ax[k].text(j, i, f'{r[i, j]:.2f}', ha='center', va='center')
    fig.colorbar(im, ax=ax.ravel().tolist()); fig.savefig(path, dpi=150); plt.close(fig)


def fig3_pdf(y, s, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for d in range(4):
        a = ax[d // 2, d % 2]
        a.hist(y[..., d].ravel(), bins=60, density=True, alpha=.5, color='r', label='Real')
        a.hist(s[..., d].ravel(), bins=60, density=True, alpha=.5, color='gray', label='Generated')
        a.set_title(NAMES[d]); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig4_scenarios(s, path, w=0, n=100):          # 源荷场景:多条采样 + 均值±界
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for d in range(4):
        a = ax[d // 2, d % 2]; ss = s[w, :n, :, d]
        a.plot(ss.T, lw=.3, alpha=.3)
        a.plot(ss.mean(0), 'k-', lw=2, label='Mean')
        a.plot(np.quantile(ss, .95, 0), 'k--'); a.plot(np.quantile(ss, .05, 0), 'k--')
        a.set_title(NAMES[d] + ' Scenarios')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
