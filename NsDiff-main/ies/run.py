from __future__ import annotations
import argparse, os, yaml, json, time, numpy as np, torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ies.data import build_dataloaders, inverse_y, PV_IDX
from ies.build_model import build
from ies.g_weather import y_sigma_target
from ies.hurdle import zero_bce_loss, apply_constraints, hurdle_mix
from ies import metrics as M
from ies import plots as P
from src.layer.nsdiff_utils import (q_sample, p_sample_loop, p_sample_loop_pe,
                                     cal_forward_noise, cal_sigma_tilde)

try:
    from tqdm import tqdm
except ImportError:                          # tqdm 缺失时退化为普通可迭代对象
    def tqdm(it=None, *a, **k):
        return it if it is not None else None

EPS = 1e-8
N_WEATHER = 4          # 与 ies.data.WEATHER 一致


def set_seed(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to(b, dev):
    return {k: v.to(dev).float() if v.dtype != torch.int64 else v.to(dev) for k, v in b.items()}


def apply_ablation_inputs(b, abl):
    """§13 w/o 天气:抹掉天气(x_w/y_w 置零 + mark 后 N_WEATHER 列置零),只留日历。"""
    if abl.get('no_weather', False):
        b = dict(b)
        b['x_w'] = torch.zeros_like(b['x_w'])
        b['y_w'] = torch.zeros_like(b['y_w'])
        b['x_mark'] = b['x_mark'].clone(); b['x_mark'][..., -N_WEATHER:] = 0
        b['y_mark'] = b['y_mark'].clone(); b['y_mark'][..., -N_WEATHER:] = 0
    return b


# ---------- 损失(训练一个 batch) ----------
def train_loss(model, f, g, zero, b, rolling, abl):
    x_y, y_y, x_w, y_w = b['x_y'], b['y_y'], b['x_w'], b['y_w']
    x_mark, y_mark = b['x_mark'], b['y_mark']
    y_sigma = y_sigma_target(x_y, y_y, rolling)

    y0 = f(x_y, x_mark, y_mark)                     # f 含 Oracle 天气
    gx = g(x_y, x_w, y_w) + EPS                     # g 含 Oracle 天气
    loss1 = (y0 - y_y).square().mean()
    loss2 = (gx.sqrt() - y_sigma.sqrt()).square().mean()

    if abl.get('no_lsnm', False):                   # §13 w/o LSNM:端点用 N(f, I)
        gx = torch.ones_like(gx)
        y_sigma = torch.ones_like(y_sigma)
        loss2 = torch.zeros((), device=x_y.device)

    n = x_y.size(0); dev = x_y.device
    t = torch.randint(0, model.num_timesteps, (n // 2 + 1,), device=dev)
    t = torch.cat([t, model.num_timesteps - 1 - t])[:n]
    e = torch.randn_like(y_y)
    fwd = cal_forward_noise(model.betas_tilde, model.betas_bar, gx, y_sigma, t)
    y_t = q_sample(y_y, y0, model.alphas_bar_sqrt, model.one_minus_alphas_bar_sqrt, t, noise=e * fwd.sqrt())
    sig_tilde = cal_sigma_tilde(model.alphas, model.alphas_cumprod, model.alphas_cumprod_sum,
                                model.alphas_cumprod_prev, model.alphas_cumprod_sum_prev,
                                model.betas_tilde_m_1, model.betas_bar_m_1, gx, y_sigma, t)
    out, sig_theta = model(x_y, x_mark, y_t, y0, gx, t)   # x_mark 8 维;去噪器不用 enc_out
    sig_theta = sig_theta + EPS
    kl = (e - out).square().mean() + (sig_tilde / sig_theta).mean() - torch.log(sig_tilde / sig_theta).mean()
    loss = kl + loss1 + loss2
    if zero is not None:
        loss = loss + zero_bce_loss(zero(x_y, x_w, y_w), b['y_y_raw'])
    return loss


# ---------- 采样(一个 batch → (B,S,O,4) 标准化) ----------
@torch.no_grad()
def sample_batch(model, f, g, b, num_samples, abl, sample_chunk=10):
    x_y, x_w, y_w, x_mark, y_mark = b['x_y'], b['x_w'], b['y_w'], b['x_mark'], b['y_mark']
    B, dev = x_y.size(0), x_y.device
    y0 = f(x_y, x_mark, y_mark); gx = g(x_y, x_w, y_w)
    if abl.get('no_lsnm', False):
        gx = torch.ones_like(gx)
    # §13 w/o UANS:完美估计器 σ_Y0 = g(x),用 p_sample_loop_pe
    loop = p_sample_loop_pe if abl.get('no_uans', False) else p_sample_loop
    outs = []
    done = 0
    while done < num_samples:
        S = min(sample_chunk, num_samples - done); done += S
        rep = lambda z: z.repeat_interleave(S, dim=0)
        seq = loop(model, rep(x_y), rep(x_mark), rep(y0), rep(gx), rep(y0),
                   model.num_timesteps, model.alphas, model.one_minus_alphas_bar_sqrt,
                   model.alphas_cumprod, model.alphas_cumprod_sum,
                   model.alphas_cumprod_prev, model.alphas_cumprod_sum_prev,
                   model.betas_tilde, model.betas_bar,
                   model.betas_tilde_m_1, model.betas_bar_m_1)
        y0_gen = seq[-1].reshape(B, S, y0.size(1), y0.size(2)).cpu()   # (B,S,O,4)
        outs.append(y0_gen)
    return torch.cat(outs, dim=1)                                     # (B,num_samples,O,4)


def crps_avg(y, s):  # 标准化后跨变量平均;kW 下按需分变量看
    return float(np.mean(M.crps_per_target(y, s)))


@torch.no_grad()
def evaluate(model, f, g, zero, loader, sy, num_samples, dev, hurdle_cfg, abl,
             want_arrays=False, max_batches=None, desc='eval'):
    model.eval(); f.eval(); g.eval()
    Y, S = [], []
    n_total = max_batches if max_batches is not None else len(loader)
    pbar = tqdm(loader, total=n_total, desc=f'{desc} (sampling x{num_samples})', ncols=100, leave=False)
    for bi, b in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break
        b = apply_ablation_inputs(to(b, dev), abl)
        s_std = sample_batch(model, f, g, b, num_samples, abl).numpy()     # (B,S,O,4) std
        s_kw = inverse_y(s_std, sy)                                        # → kW
        s_kw = torch.tensor(s_kw)
        if zero is not None and hurdle_cfg['enable']:
            s_kw = hurdle_mix(s_kw, zero(b['x_y'], b['x_w'], b['y_w']).cpu(), hurdle_cfg.get('thresh', .5))
        if hurdle_cfg.get('nonneg', True):
            s_kw = apply_constraints(s_kw, b['y_hour'].cpu(), PV_IDX,
                                     night=tuple(hurdle_cfg.get('night', (6, 19))))
        Y.append(b['y_y_raw'].cpu().numpy()); S.append(s_kw.numpy())
    y = np.concatenate(Y); s = np.concatenate(S)                          # kW
    # 标准化空间(联合/校准指标)
    yz = (y - sy.mean_) / sy.scale_; sz = (s - sy.mean_) / sy.scale_
    res = {
        'CRPS_kw': crps_avg(y, s), 'CRPS_std': crps_avg(yz, sz),
        'QICE': M.qice(yz, sz), 'EnergyScore': M.energy_score(yz, sz),
        'Variogram': M.variogram_score(yz, sz), 'PSD': M.psd_distance(yz, sz),
        'ACF': M.acf_distance(yz, sz), 'APD': M.apd(sz), 'AvgStd': M.avg_std(sz),
        'DTW': M.dtw_mean(y, s),
    }
    res['CorrErr'], r_real, r_gen = M.corr_error(yz, sz)
    res['CRPS_per_target_kw'] = M.crps_per_target(y, s)
    res['QICE_per_target'] = M.qice_per_target(yz, sz)
    res['MAE_kw'], res['RMSE_kw'] = M.mae_rmse_per_target(y, s)
    return (res, (y, s, r_real, r_gen)) if want_arrays else res


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {dev}')
    set_seed(cfg.get('seed', 2026))
    out = Path(cfg.get('output_dir', 'outputs_ies')); (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True); (out / 'ckpt').mkdir(exist_ok=True)

    dl = build_dataloaders(cfg); sy = dl['scaler_y']
    args, model, f, g, zero = build(cfg, dl['n_weather'], dev)
    params = list(model.parameters()) + list(f.parameters()) + list(g.parameters()) + \
        (list(zero.parameters()) if zero is not None else [])
    opt = AdamW(params, lr=cfg['train']['lr'], weight_decay=cfg['train'].get('weight_decay', 1e-6))
    sch = ReduceLROnPlateau(opt, mode='min', factor=.5, patience=3)
    rolling = cfg['data'].get('rolling_length', 96)
    abl = cfg.get('ablation', {}) or {}
    if abl:
        print(f'ablation: {abl}')
    best, bad, patience = 1e9, 0, cfg['train'].get('patience', 8)
    max_train_batches = cfg['train'].get('max_batches', None)
    val_max_batches = cfg['infer'].get('val_max_batches', None)

    n_train = max_train_batches if max_train_batches is not None else len(dl['train'])
    for ep in range(1, cfg['train']['epochs'] + 1):
        model.train(); f.train(); g.train(); tl = []
        t0 = time.time()
        pbar = tqdm(dl['train'], total=n_train, desc=f'epoch {ep}/{cfg["train"]["epochs"]} [train]', ncols=100)
        for bi, b in enumerate(pbar):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            b = apply_ablation_inputs(to(b, dev), abl); opt.zero_grad()
            loss = train_loss(model, f, g, zero, b, rolling, abl)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, cfg['train'].get('grad_clip', 1.0))
            opt.step(); tl.append(loss.item())
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f'{loss.item():.4f}', avg=f'{np.mean(tl):.4f}')
        val = evaluate(model, f, g, zero, dl['val'], sy, cfg['infer'].get('val_samples', 50), dev,
                       cfg['hurdle'], abl, max_batches=val_max_batches, desc=f'epoch {ep} [val]')
        sch.step(val['CRPS_std'])
        print(f"epoch {ep}  train {np.mean(tl):.4f}  val CRPS(std) {val['CRPS_std']:.4f}  "
              f"QICE {val['QICE']:.3f}  ({time.time() - t0:.1f}s)")
        if val['CRPS_std'] < best - 1e-5:
            best, bad = val['CRPS_std'], 0
            torch.save({'model': model.state_dict(), 'f': f.state_dict(), 'g': g.state_dict(),
                        'zero': (zero.state_dict() if zero else None)}, out / 'ckpt' / 'best.pt')
        else:
            bad += 1
            if bad >= patience:
                print('early stop'); break

    ck = torch.load(out / 'ckpt' / 'best.pt', map_location=dev)
    model.load_state_dict(ck['model']); f.load_state_dict(ck['f']); g.load_state_dict(ck['g'])
    if zero is not None:
        zero.load_state_dict(ck['zero'])

    print('running final test evaluation ...')
    res, (y, s, r_real, r_gen) = evaluate(model, f, g, zero, dl['test'], sy,
                                          cfg['infer']['num_samples'], dev, cfg['hurdle'], abl,
                                          want_arrays=True, max_batches=cfg['infer'].get('test_max_batches', None),
                                          desc='test')
    json.dump(res, open(out / 'tables' / 'metrics.json', 'w'), indent=2, ensure_ascii=False)
    print('\n==== TEST ===='); print(json.dumps(res, indent=2, ensure_ascii=False))
    P.fig1_real_vs_gen(y, s, str(out / 'figures' / 'fig1_real_vs_gen.png'))
    P.fig2_corr(r_real, r_gen, str(out / 'figures' / 'fig2_corr.png'))
    P.fig3_pdf(y, s, str(out / 'figures' / 'fig3_pdf.png'))
    P.fig4_scenarios(s, str(out / 'figures' / 'fig4_scenarios.png'))
    np.savez_compressed(out / 'test_samples.npz', y_true=y, samples=s)


if __name__ == '__main__':
    main()
