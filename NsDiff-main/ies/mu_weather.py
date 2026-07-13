import copy, torch, torch.nn as nn
import src.layer.mu_backbone as ns
from ies.weather_embedding import WeatherDataEmbedding


class WeatherMeanF(nn.Module):
    """封装原 NS-Transformer:enc_in=c_out=4(反归一化一致),天气经 mark 注入。"""
    def __init__(self, args, n_target=4, mark_dim=8, weather_mlp=True):
        super().__init__()
        a = copy.copy(args)
        a.enc_in = a.dec_in = a.c_out = n_target
        self.net = ns.Model(a)
        d, dp = a.d_model, a.dropout
        self.net.enc_embedding = WeatherDataEmbedding(n_target, d, mark_dim, dp, weather_mlp)
        self.net.dec_embedding = WeatherDataEmbedding(n_target, d, mark_dim, dp, weather_mlp)
        self.label_len, self.pred_len, self.n_t = a.label_len, a.pred_len, n_target

    def forward(self, x_y, x_mark, y_mark):
        # 解码器 mark = [历史后 label_len ‖ 未来 O(含 Oracle 天气)]
        dec_mark = torch.cat([x_mark[:, -self.label_len:], y_mark], dim=1)
        dec_val = torch.cat([x_y[:, -self.label_len:],
                             torch.zeros(x_y.size(0), self.pred_len, self.n_t, device=x_y.device)], 1)
        y0_hat, _ = self.net(x_y, x_mark, dec_val, dec_mark)   # (x_enc,x_mark_enc,x_dec,x_mark_dec)
        return y0_hat                                          # (B,O,4)
