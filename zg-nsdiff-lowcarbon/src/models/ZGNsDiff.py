"""ZG-NsDiff: NsDiff-Exo backbone + occurrence head + masked continuous
losses + Bernoulli gate sampling (spec section 8).

The ONLY differences vs the NsDiff baseline are the occurrence head, the
active magnitude mask and the Bernoulli gating; backbone widths and
exogenous conditioning are identical (spec 7.1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.low_carbon_schema import TARGET_TO_GATE, ZERO_TARGET_INDICES
from src.layer.masked_reduction import masked_mean
from src.layer.occurrence_head import OccurrenceHead
from src.models.NsDiffExo import NsDiffExo


def build_magnitude_mask(future_observed: torch.Tensor, future_active: torch.Tensor) -> torch.Tensor:
    """spec 8.4: electricity -> observed; zero-inflated -> observed & active."""
    magnitude_mask = future_observed.clone()
    for target_index, gate_index in TARGET_TO_GATE.items():
        magnitude_mask[..., target_index] &= future_active[..., gate_index]
    return magnitude_mask


class ZGNsDiff(NsDiffExo):
    def __init__(self, cfg: dict, device):
        super().__init__(cfg, device)
        m = cfg["model"]
        self.occurrence_head = OccurrenceHead(
            d_model=int(m["d_model"]),
            hidden_dim=int(m.get("gate_hidden_dim", 128)),
            out_dim=len(ZERO_TARGET_INDICES),
            dropout=float(m.get("gate_dropout", 0.1)),
        )
        pos_weight = cfg.get("_gate_pos_weight", [1.0, 1.0, 1.0])
        self.register_buffer(
            "gate_pos_weight", torch.tensor(pos_weight, dtype=torch.float32)
        )

    def gate_logits_from_hidden(self, future_hidden):
        return self.occurrence_head(future_hidden)

    def gate_loss(self, gate_logits, future_active, gate_observed):
        gate_elem = F.binary_cross_entropy_with_logits(
            gate_logits,
            future_active.float(),
            pos_weight=self.gate_pos_weight.view(1, 1, -1),
            reduction="none",
        )
        return masked_mean(gate_elem, gate_observed)

    def loss(self, batch, mask, loss_cfg):
        """Masked continuous losses + weighted gate BCE (spec 8.6)."""
        total, logs, elems = super().loss(batch, mask, loss_cfg)
        gate_logits = self.gate_logits_from_hidden(elems["future_hidden"])
        gate_observed = batch["future_observed"][..., ZERO_TARGET_INDICES]
        loss_gate = self.gate_loss(gate_logits, batch["future_active"], gate_observed)
        total = total + float(loss_cfg.get("lambda_gate", 0.5)) * loss_gate
        logs["loss_gate"] = loss_gate.item()
        elems["gate_logits"] = gate_logits
        return total, logs, elems

    @torch.no_grad()
    def predict_gate_prob(self, batch):
        _, future_hidden = self.forward_mean(batch)
        return torch.sigmoid(self.gate_logits_from_hidden(future_hidden))  # [B,H,3]

    @staticmethod
    def sample_gates(gate_prob: torch.Tensor, num_samples: int,
                     generator: torch.Generator = None) -> torch.Tensor:
        """Bernoulli gate scenarios, independent per sample: [B,H,3,S]."""
        expanded = gate_prob.unsqueeze(-1).expand(*gate_prob.shape, num_samples)
        if generator is None:
            return torch.bernoulli(expanded)
        return torch.bernoulli(expanded, generator=generator)


def apply_gate_and_inverse(magnitude_samples: torch.Tensor,
                           gate_samples: torch.Tensor,
                           target_transform) -> torch.Tensor:
    """magnitude_samples [B,H,4,S] (model space) + gates [B,H,3,S]
    -> raw kW samples [B,H,4,S], exact zeros where gate == 0 (spec 10.3)."""
    raw = torch.empty_like(magnitude_samples)
    elec = magnitude_samples[..., 0, :]
    raw[..., 0, :] = (
        elec * target_transform.elec_std + target_transform.elec_mean
    ).clamp_min(0.0)
    for target_index, gate_index in TARGET_TO_GATE.items():
        magnitude_raw = target_transform.positive_inverse(
            magnitude_samples[..., target_index, :], target_index
        )
        raw[..., target_index, :] = gate_samples[..., gate_index, :] * magnitude_raw
    return raw
