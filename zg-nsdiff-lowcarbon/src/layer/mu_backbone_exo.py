"""Mean network f_phi with oracle future exogenous conditioning (spec 7.3).

Structure kept from the original NsDiff mu backbone (Non-stationary
Transformer): instance normalisation of the target history, tau/delta
learners, DSAttention encoder/decoder, final de-normalisation.  The only
change is DataEmbedding -> ExogenousDataEmbedding so real future weather and
calendar features enter both encoder and decoder.
"""
import torch
import torch.nn as nn

from src.layer.attention import AttentionLayer, DSAttention
from src.layer.exogenous_embedding import ExogenousDataEmbedding
from src.layer.transformer_encdec import Decoder, DecoderLayer, Encoder, EncoderLayer


class Projector(nn.Module):
    """MLP learning the de-stationary factors tau/delta."""

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len, out_channels=1, kernel_size=kernel_size,
            padding=padding, padding_mode="circular", bias=False,
        )
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)
        return self.backbone(x)


class Model(nn.Module):
    """forward(history_target, history_condition, decoder_target, decoder_condition)
    -> (mu [B,H,D], future_hidden [B,H,d_model])"""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        self.enc_embedding = ExogenousDataEmbedding(
            configs.enc_in, configs.condition_dim, configs.d_model, configs.dropout
        )
        self.dec_embedding = ExogenousDataEmbedding(
            configs.dec_in, configs.condition_dim, configs.d_model, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads,
                    ),
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )
        self.tau_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
            output_dim=configs.seq_len,
        )

    def forward(self, history_target, history_condition, decoder_target, decoder_condition):
        x_raw = history_target.clone().detach()

        mean_enc = history_target.mean(1, keepdim=True).detach()
        x_enc = history_target - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc
        # decoder value input: normalised history tail + zero future placeholder
        # (real future target NEVER enters; future info flows via decoder_condition)
        x_dec_new = torch.cat(
            [x_enc[:, -self.label_len:, :],
             torch.zeros_like(decoder_target[:, -self.pred_len:, :])],
            dim=1,
        )

        tau = self.tau_learner(x_raw, std_enc).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, history_condition)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_in = self.dec_embedding(x_dec_new, decoder_condition)
        dec_out, hidden = self.decoder(dec_in, enc_out, tau=tau, delta=delta)

        dec_out = dec_out * std_enc + mean_enc
        mu = dec_out[:, -self.pred_len:, :]
        future_hidden = hidden[:, -self.pred_len:, :]
        return mu, future_hidden
