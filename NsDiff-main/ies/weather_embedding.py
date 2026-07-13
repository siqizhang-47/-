import math, torch, torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):                       # (B,T,c_in)
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model); pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class WeatherDataEmbedding(nn.Module):
    """value = 目标 4 通道卷积;temporal = MLP(日历+天气 → d_model)。
       天气(含 Oracle 未来)从 mark 注入;辐照→PV 是强非线性,故用 2 层 MLP。"""
    def __init__(self, c_in, d_model, mark_dim, dropout=0.05, weather_mlp=True):
        super().__init__()
        self.value = TokenEmbedding(c_in, d_model)
        self.pos = PositionalEmbedding(d_model)
        self.temporal = (nn.Sequential(nn.Linear(mark_dim, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
                         if weather_mlp else nn.Linear(mark_dim, d_model, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_mark):               # x:(B,T,c_in)  x_mark:(B,T,mark_dim)
        return self.drop(self.value(x) + self.temporal(x_mark) + self.pos(x))
