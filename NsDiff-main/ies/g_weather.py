import torch, torch.nn as nn, torch.nn.functional as F
from src.utils.sigma import wv_sigma_trailing


class WeatherSigmaG(nn.Module):
    """g(x):目标滑窗方差 + 天气历史统计 + Oracle 未来天气逐步值 → (B,O,4),softplus 保正,带残差。"""
    def __init__(self, seq_len, pred_len, n_target=4, n_weather=4, hidden=512, kernel=24):
        super().__init__()
        self.O, self.k, self.n_t, self.n_w = pred_len, kernel, n_target, n_weather
        base = seq_len - kernel
        w_feat = n_weather * 2 + n_weather * pred_len          # 历史均值/方差 + 未来逐步(Oracle)
        self.mlp = nn.Sequential(nn.Linear(base + w_feat, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, pred_len))
        self.res = nn.Linear(base + w_feat, pred_len)

    def forward(self, x_y, x_w, y_w):                          # y_w:(B,O,W) Oracle
        B = x_y.size(0)
        sig = wv_sigma_trailing(x_y, self.k, discard_rep=True)
        sig = sig[:, -(x_y.size(1) - self.k):, :] + 1e-8       # (B,base,4)
        sig = sig.permute(0, 2, 1)                             # (B,4,base)
        wh = torch.cat([x_w.mean(1), x_w.var(1)], -1)          # (B,2W)
        wf = y_w.reshape(B, -1)                                # (B,O*W)
        wcat = torch.cat([wh, wf], -1).unsqueeze(1).expand(-1, self.n_t, -1)
        feat = torch.cat([sig, wcat], -1)                      # (B,4,base+w_feat)
        out = self.mlp(feat) + self.res(feat)                 # (B,4,O)
        return F.softplus(out).permute(0, 2, 1)[:, -self.O:, :] + 1e-8   # (B,O,4)


def y_sigma_target(x_y, y_y, rolling):
    """g 的监督目标(loss2,目标通道滑窗方差)。"""
    return wv_sigma_trailing(torch.cat([x_y, y_y], 1), rolling)[:, -y_y.size(1):, :] + 1e-8
