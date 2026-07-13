import torch, torch.nn as nn, torch.nn.functional as F


class ZeroHead(nn.Module):
    """预测每步每变量为 0 的概率 pi(x) ∈ (0,1),(B,O,4)。输入用历史目标+天气编码。"""
    def __init__(self, seq_len, pred_len, n_target=4, n_weather=4, hidden=256):
        super().__init__()
        self.O = pred_len
        self.enc = nn.GRU(n_target + n_weather, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_weather * pred_len, hidden), nn.ReLU(),
                                  nn.Linear(hidden, pred_len * n_target))
        self.n_t = n_target

    def forward(self, x_y, x_w, y_w):
        _, h = self.enc(torch.cat([x_y, x_w], -1)); h = h[-1]      # (B,hidden)
        z = torch.cat([h, y_w.reshape(x_y.size(0), -1)], -1)
        logit = self.head(z).view(x_y.size(0), self.O, self.n_t)
        return logit                                              # 未过 sigmoid(配 BCEWithLogits)


def zero_bce_loss(logit, y_y_raw):
    target = (y_y_raw.abs() < 1e-6).float()                       # 原始 kW==0 处为 1
    return F.binary_cross_entropy_with_logits(logit, target)


def apply_constraints(samples, y_hour, pv_idx=3, night=(6, 19)):
    """samples:(B,S,O,4)。非负 + 夜间 PV=0。y_hour:(B,O)。"""
    samples = torch.clamp(samples, min=0.0)
    is_night = (y_hour < night[0]) | (y_hour > night[1])          # (B,O)
    mask = is_night.unsqueeze(1)                                  # (B,1,O)
    samples[..., pv_idx] = samples[..., pv_idx].masked_fill(mask, 0.0)
    return samples


def hurdle_mix(samples, logit, thresh=0.5):
    """按 pi>thresh 把对应位置置 0(两段式)。samples:(B,S,O,4) logit:(B,O,4)。"""
    pi = torch.sigmoid(logit).unsqueeze(1)                        # (B,1,O,4)
    return samples.masked_fill(pi > thresh, 0.0)
