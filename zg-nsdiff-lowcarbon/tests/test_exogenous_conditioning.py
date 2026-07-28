"""With identical target history, changing ONLY the future weather must
change mu, gx and eps_pred — guards against 'interface accepts weather but
the network ignores it'."""
import torch

from src.models.NsDiffExo import NsDiffExo


def test_future_condition_affects_all_three_paths(tiny_cfg, tiny_batch):
    torch.manual_seed(0)
    model = NsDiffExo(tiny_cfg, torch.device("cpu"))
    model.eval()

    batch_a = {k: v.clone() if torch.is_tensor(v) else v for k, v in tiny_batch.items()}
    batch_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in tiny_batch.items()}
    batch_b["future_condition"] = batch_b["future_condition"] + 1.5

    with torch.no_grad():
        mu_a, _ = model.forward_mean(batch_a)
        mu_b, _ = model.forward_mean(batch_b)
        gx_a = model.forward_variance(batch_a)
        gx_b = model.forward_variance(batch_b)
        t = torch.zeros(mu_a.size(0), dtype=torch.long)
        y_t = torch.randn_like(mu_a)
        eps_a, _ = model.denoiser(batch_a["history_target"], batch_a["history_condition"],
                                  batch_a["future_condition"], y_t, mu_a, gx_a, t)
        eps_b, _ = model.denoiser(batch_b["history_target"], batch_b["history_condition"],
                                  batch_b["future_condition"], y_t, mu_a, gx_a, t)

    assert not torch.allclose(mu_a, mu_b), "mean path ignores future weather"
    assert not torch.allclose(gx_a, gx_b), "variance path ignores future weather"
    assert not torch.allclose(eps_a, eps_b), "denoiser ignores future weather"


def test_deepvar_uses_future_condition(tiny_cfg, tiny_batch):
    from src.baselines.deepvar_adapter import DeepVAR
    torch.manual_seed(0)
    model = DeepVAR(tiny_cfg, torch.device("cpu"))
    model.eval()
    torch.manual_seed(1)
    s_a = model.sample(tiny_batch, num_samples=4, chunk_size=4)
    batch_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in tiny_batch.items()}
    batch_b["future_condition"] = batch_b["future_condition"] + 2.0
    torch.manual_seed(1)
    s_b = model.sample(batch_b, num_samples=4, chunk_size=4)
    assert not torch.allclose(s_a, s_b), "DeepVAR ignores future conditions"
