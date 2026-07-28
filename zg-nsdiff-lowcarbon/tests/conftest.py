import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


TINY_CFG = {
    "data": {"context_length": 48, "prediction_length": 8, "label_length": 16},
    "model": {
        "d_model": 32, "n_heads": 2, "e_layers": 1, "d_layers": 1, "d_ff": 32,
        "factor": 3, "dropout": 0.0, "activation": "gelu", "condition_dim": 10,
        "enc_in": 3, "p_hidden_dims": [16, 16], "p_hidden_layers": 2,
        "g_hidden_size": 32, "history_embed_dim": 8, "denoise_context_dim": 16,
        "denoise_hidden_dim": 32, "hidden_dim": 32, "num_layers": 1,
    },
    "diffusion": {"steps": 6, "beta_schedule": "linear", "beta_start": 1e-4,
                  "beta_end": 1e-2, "rolling_length": 8},
    "loss": {"lambda_mean": 1.0, "lambda_variance": 1.0, "lambda_diffusion": 1.0,
             "lambda_reverse_var": 1.0},
}


@pytest.fixture
def tiny_cfg():
    import copy
    return copy.deepcopy(TINY_CFG)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(0)
    B, L, H = 3, 48, 8
    return {
        "history_target": torch.randn(B, L, 3),
        "future_target": torch.randn(B, H, 3),
        "history_target_raw": torch.rand(B, L, 3) * 100,
        "future_target_raw": torch.rand(B, H, 3) * 100,
        "history_condition": torch.randn(B, L, 10),
        "future_condition": torch.randn(B, H, 10),
        "history_observed": torch.ones(B, L, 3, dtype=torch.bool),
        "future_observed": torch.ones(B, H, 3, dtype=torch.bool),
        "forecast_start_index": torch.arange(B, dtype=torch.long),
        "timestamps": torch.arange(B * H, dtype=torch.long).reshape(B, H),
    }


@pytest.fixture
def tiny_stats():
    return {
        "target_mean": [500.0, 200.0, 5.0],
        "target_std": [100.0, 50.0, 2.0],
        "train_scale_raw": [100.0, 50.0, 2.0],
        "target_names": ["electricity", "cooling", "heat"],
    }
