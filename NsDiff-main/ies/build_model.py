import torch
from types import SimpleNamespace
from src.models.NsDiff import NsDiff
from ies.mu_weather import WeatherMeanF
from ies.g_weather import WeatherSigmaG
from ies.hurdle import ZeroHead
from ies.weather_embedding import WeatherDataEmbedding


def make_args(cfg, n_weather, device):
    m, d = cfg['model'], cfg['data']
    L, O = d['input_len'], d['pred_len']
    return SimpleNamespace(
        seq_len=L, label_len=L // 2, pred_len=O, device=device, features='M',
        enc_in=4, dec_in=4, c_out=4,                       # 目标维:去噪器/端点/反归一化
        d_model=m['d_model'], n_heads=m['n_heads'], e_layers=m['e_layers'],
        d_layers=m['d_layers'], d_ff=m['d_ff'], moving_avg=m.get('moving_avg', 25),
        factor=m.get('factor', 3), distil=m.get('distil', True),
        timesteps=m['diffusion_steps'], beta_schedule=m.get('beta_schedule', 'linear'),
        beta_start=m['beta_start'], beta_end=m['beta_end'],
        embed='timeF', freq='h', dropout=m['dropout'], activation=m.get('activation', 'gelu'),
        output_attention=False, do_predict=True, k_z=1e-2, k_cond=1,
        p_hidden_dims=[64, 64], p_hidden_layers=2,
        CART_input_x_embed_dim=m.get('CART_input_x_embed_dim', 32), d_z=8,
        diffusion_config_dir='./configs/nsdiff.yml',
    )


def build(cfg, n_weather, device):
    args = make_args(cfg, n_weather, device)
    mark_dim = 4 + n_weather

    model = NsDiff(args, device).to(device)
    # 铁律3:把 NsDiff 自带(但去噪器不用的)enc_embedding 换成天气版,统一 8 维 mark
    model.enc_embedding = WeatherDataEmbedding(4, args.CART_input_x_embed_dim, mark_dim,
                                               args.dropout, cfg['model'].get('weather_mlp', True)).to(device)

    f = WeatherMeanF(args, n_target=4, mark_dim=mark_dim,
                     weather_mlp=cfg['model'].get('weather_mlp', True)).to(device)
    g = WeatherSigmaG(args.seq_len, args.pred_len, 4, n_weather,
                      cfg['model'].get('g_hidden', 512), cfg['data'].get('rolling_length', 96)).to(device)
    zero = (ZeroHead(args.seq_len, args.pred_len, 4, n_weather).to(device)
            if cfg['hurdle']['enable'] else None)
    return args, model, f, g, zero
