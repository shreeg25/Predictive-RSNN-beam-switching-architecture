"""
Feedforward Spiking Neural Network (SNN) for Predictive Beam Switching
Using SNNTorch with Leaky Integrate-and-Fire (LIF) neurons unrolled over time.
Architecture: Input Projection → LIF Layer 1 → LIF Layer 2 → Manual Leaky Integrator
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
from typing import Tuple, Optional

class RecurrentBeamSNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_beams: int,
        hidden_1: int = 256,
        hidden_2: int = 128,
        beta: float = 0.9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_beams = n_beams
        
        # Wide surrogate gradient to keep learning alive
        spike_grad = surrogate.atan(alpha=2.0)
        self.current_gain = 5.0 

        # Spiking Layer 1
        self.fc1 = nn.Linear(n_features, hidden_1)
        self.ln1 = nn.LayerNorm(hidden_1)
        self.lif1 = snn.Leaky(
            beta=beta, threshold=1.0, spike_grad=spike_grad, 
            learn_beta=False, reset_mechanism="zero"
        )

        # Spiking Layer 2
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.ln2 = nn.LayerNorm(hidden_2)
        self.lif2 = snn.Leaky(
            beta=beta, threshold=1.0, spike_grad=spike_grad, 
            learn_beta=False, reset_mechanism="zero"
        )

        # Non-Spiking Output Readout (NO snnTorch threshold bug)
        self.fc_out = nn.Linear(hidden_2, n_beams)
        self.out_beta = 0.8  

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        batch, seq_len, _ = x.shape
        device = x.device

# Pure PyTorch initialization: perfectly matches the exact batch size every time
        mem1 = torch.zeros(batch, self.fc1.out_features, device=device)
        mem2 = torch.zeros(batch, self.fc2.out_features, device=device)
        mem_out = torch.zeros(batch, self.n_beams, device=device)

        logits_all, spk2_all = [], []

        for t in range(seq_len):
            xt = x[:, t, :]                             

            # Layer 1
            cur1 = self.ln1(self.fc1(xt)) * self.current_gain  
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_d = self.dropout(spk1)

            # Layer 2
            cur2 = self.ln2(self.fc2(spk1_d)) * self.current_gain  
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_d = self.dropout(spk2)

            # Pure PyTorch Manual Leaky Integration
            cur_out = self.fc_out(spk2_d)
            mem_out = (mem_out * self.out_beta) + cur_out

            logits_all.append(mem_out)
            spk2_all.append(spk2)

        return torch.stack(logits_all, dim=1), torch.stack(spk2_all, dim=1), (mem1, mem2, mem_out)


class BeamSwitchingController:
    def __init__(self, hysteresis_db: float = 3.0, min_hold_steps: int = 3):
        self.hysteresis = 10 ** (hysteresis_db / 10)
        self.min_hold, self.current_beam, self.hold_counter = min_hold_steps, None, 0

    def decide(self, beam_gains: np.ndarray) -> Tuple[int, bool]:
        best_beam = int(np.argmax(beam_gains))
        if self.current_beam is None:
            self.current_beam = best_beam
            return best_beam, True

        self.hold_counter += 1
        if beam_gains[best_beam] > beam_gains[self.current_beam] * self.hysteresis and self.hold_counter >= self.min_hold:
            self.current_beam = best_beam
            self.hold_counter = 0
            return best_beam, True
        return self.current_beam, False

    def reset(self): 
        self.current_beam, self.hold_counter = None, 0


class BeamSNNLoss(nn.Module):
    def __init__(self, n_beams: int, lambda_spk: float = 1e-3, lambda_topk: float = 0.1):
        super().__init__()
        self.ce_loss, self.lambda_spk, self.lambda_topk, self.n_beams = nn.CrossEntropyLoss(), lambda_spk, lambda_topk, n_beams

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spk_raster: torch.Tensor, topk_targets: Optional[torch.Tensor] = None):
        loss_ce = self.ce_loss(logits.reshape(-1, self.n_beams), targets.reshape(-1))
        spike_rate = spk_raster.float().mean()
        loss_spk = self.lambda_spk * (spike_rate - 0.15).pow(2)

        loss_topk = torch.tensor(0.0, device=logits.device)
        if topk_targets is not None:
            soft_labels = torch.zeros_like(logits)
            for k_i in range(topk_targets.shape[-1]):
                soft_labels.scatter_add_(-1, topk_targets[:, :, k_i].unsqueeze(-1), torch.full_like(soft_labels, 1.0 / (k_i + 1)))
            loss_topk = self.lambda_topk * -( (soft_labels / (soft_labels.sum(-1, keepdim=True) + 1e-9)) * torch.log_softmax(logits, dim=-1) ).sum(-1).mean()

        return loss_ce + loss_spk + loss_topk, {'total': (loss_ce + loss_spk + loss_topk).item(), 'spike_rate': spike_rate.item()}


def build_model(n_features: int, n_beams: int, device: torch.device) -> RecurrentBeamSNN:
    return RecurrentBeamSNN(n_features=n_features, n_beams=n_beams, hidden_1=256, hidden_2=128, beta=0.9, dropout=0.2).to(device)