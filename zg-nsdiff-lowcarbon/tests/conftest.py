import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


TINY_CFG = {
    "data": {"context_length": 48, "prediction_length": 8, "label_length": 16},
    "model": {
        "d_model": 32, "n_heads": 2, "e_layers": 1, "d_layers": 1, "d_ff": 32,
        "factor": 3, "dropout": 0.0, "activation": "gelu", "condition_dim": 11,
        "enc_in": 4, "p_hidden_dims": [16, 16], "p_hidden_layers": 2,
        "g_hidden_size": 32, "history_embed_dim": 8, "denoise_context_dim": 16,
        "denoise_hidden_dim": 32, "gate_hidden_dim": 16, "gate_dropout": 0.0,
    },
    "diffusion": {"steps": 6, "beta_schedule": "linear", "beta_start": 1e-4,
                  "beta_end": 1e-2, "rolling_length": 8},
    "loss": {"lambda_mean": 1.0, "lambda_variance": 1.0, "lambda_diffusion": 1.0,
             "lambda_reverse_var": 1.0, "lambda_gate": 0.5},
}


@pytest.fixture
def tiny_cfg():
    import copy
    return copy.deepcopy(TINY_CFG)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(0)
    B, L, H = 3, 48, 8
    batch = {
        "history_target": torch.randn(B, L, 4),
        "future_target": torch.randn(B, H, 4),
        "history_target_raw": torch.rand(B, L, 4) * 100,
        "future_target_raw": torch.rand(B, H, 4) * 100,
        "history_condition": torch.randn(B, L, 11),
        "future_condition": torch.randn(B, H, 11),
        "history_observed": torch.ones(B, L, 4, dtype=torch.bool),
        "future_observed": torch.ones(B, H, 4, dtype=torch.bool),
        "future_active": torch.rand(B, H, 3) > 0.5,
        "forecast_start_index": torch.arange(B, dtype=torch.long),
        "timestamps": torch.arange(B * H, dtype=torch.long).reshape(B, H),
    }
    return batch


@pytest.fixture
def tiny_stats():
    return {
        "electricity_mean": 500.0,
        "electricity_std": 100.0,
        "positive_scale": [1.0, 40.0, 30.0, 20.0],
        "gate_pos_weight": [2.0, 2.5, 1.5],
        "train_scale_raw": [100.0, 50.0, 40.0, 30.0],
    }
