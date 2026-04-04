"""
Training Loop + Evaluation for Predictive Beam Switching R-SNN
Includes: training, validation, per-trajectory evaluation, beam switching metrics
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from snn_model import RecurrentBeamSNN, BeamSNNLoss, BeamSwitchingController, build_model
from trajectory_generator import UserTrajectory


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    y_topk: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    X_t    = torch.tensor(X,      dtype=torch.float32)
    y_t    = torch.tensor(y,      dtype=torch.long)
    yk_t   = torch.tensor(y_topk, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t, yk_t)
    N = len(dataset)
    n_val  = int(N * val_split)
    n_test = int(N * test_split)
    n_train = N - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Data] Train={n_train} | Val={n_val} | Test={n_test} sequences")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss:   List[float] = field(default_factory=list)
    train_acc:  List[float] = field(default_factory=list)
    val_acc:    List[float] = field(default_factory=list)
    val_topk:   List[float] = field(default_factory=list)
    spike_rates:List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)


def train(
    model: RecurrentBeamSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 40,
    lr: float = 5e-4,
    device: torch.device = torch.device('cpu'),
    patience: int = 8,
    save_path: str = 'best_snn_beam.pt',
) -> TrainingHistory:

    criterion = BeamSNNLoss(n_beams=model.n_beams, lambda_spk=1e-3, lambda_topk=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history = TrainingHistory()
    best_val_loss = float('inf')
    no_improve = 0

    print(f"\n{'='*65}")
    print(f"  Training R-SNN  |  Epochs={n_epochs}  LR={lr}  Device={device}")
    print(f"{'='*65}")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'VlLoss':>8}  {'TrAcc':>7}  {'VlAcc':>7}  {'Top5':>6}  {'SpkR':>6}  {'Time':>6}")
    print(f"  {'-'*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total, tr_spk = 0.0, 0, 0, 0.0

        for X_b, y_b, yk_b in train_loader:
            X_b, y_b, yk_b = X_b.to(device), y_b.to(device), yk_b.to(device)
            optimizer.zero_grad()

            logits, spk_raster, _ = model(X_b)
            loss, stats = criterion(logits, y_b, spk_raster, yk_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            tr_loss   += stats['total'] * X_b.size(0)
            tr_spk    += stats['spike_rate'] * X_b.size(0)
            preds      = logits.argmax(dim=-1)
            tr_correct += (preds == y_b).sum().item()
            tr_total   += y_b.numel()

        tr_loss /= tr_total / X_b.shape[1]
        tr_acc   = tr_correct / tr_total
        tr_spk  /= tr_total / X_b.shape[1]

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_acc, val_topk = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.train_loss.append(tr_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(val_acc)
        history.val_topk.append(val_topk)
        history.spike_rates.append(tr_spk)
        history.lr_history.append(optimizer.param_groups[0]['lr'])

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {tr_loss:>8.4f}  {val_loss:>8.4f}  "
              f"{tr_acc:>7.3f}  {val_acc:>7.3f}  {val_topk:>6.3f}  "
              f"{tr_spk:>6.3f}  {elapsed:>5.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  [EarlyStopping] No improvement for {patience} epochs.")
                break

    print(f"\n  Best val loss: {best_val_loss:.4f}  →  saved to '{save_path}'")
    return history


@torch.no_grad()
def _evaluate(
    model: RecurrentBeamSNN,
    loader: DataLoader,
    criterion: BeamSNNLoss,
    device: torch.device,
    top_k: int = 5,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct1, correctk, total = 0.0, 0, 0, 0

    for X_b, y_b, yk_b in loader:
        X_b, y_b, yk_b = X_b.to(device), y_b.to(device), yk_b.to(device)
        logits, spk_raster, _ = model(X_b)
        loss, stats = criterion(logits, y_b, spk_raster, yk_b)

        total_loss += stats['total'] * y_b.numel()
        preds = logits.argmax(dim=-1)
        correct1 += (preds == y_b).sum().item()

        # Top-K: correct if predicted beam is in top-K labels
        topk_preds = logits.topk(top_k, dim=-1).indices   # [batch, seq, K]
        for k in range(top_k):
            correctk += (topk_preds[:, :, k] == y_b).sum().item()

        total += y_b.numel()

    return total_loss / total, correct1 / total, correctk / total


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    top1_acc:           float = 0.0
    top3_acc:           float = 0.0
    top5_acc:           float = 0.0
    beam_switch_rate:   float = 0.0
    avg_switches_per_traj: float = 0.0
    spectral_eff_gain:  float = 0.0
    avg_se_random:      float = 0.0
    avg_se_snn:         float = 0.0
    avg_se_oracle:      float = 0.0


@torch.no_grad()
def evaluate_on_trajectories(
    model: RecurrentBeamSNN,
    trajectories: List[UserTrajectory],
    device: torch.device,
    seq_len: int = 20,
) -> EvalMetrics:
    """
    Full evaluation over all trajectories.
    Computes: top-1/3/5 accuracy, beam switch rate, spectral efficiency.
    """
    model.eval()
    metrics = EvalMetrics()

    all_correct1, all_correct3, all_correct5 = [], [], []
    all_switches_snn, all_switches_oracle = [], []
    se_snn_all, se_oracle_all, se_random_all = [], [], []

    controller = BeamSwitchingController(hysteresis_db=3.0, min_hold_steps=3)

    for traj in trajectories:
        T = traj.n_steps
        X_full = torch.tensor(traj.channel_features, dtype=torch.float32).unsqueeze(0).to(device)
        # [1, T, F]

        # Run model in chunks
        pred_beams = np.zeros(T, dtype=int)
        hidden = None
        for start in range(0, T, seq_len):
            end = min(start + seq_len, T)
            chunk = X_full[:, start:end, :]
            logits, _, hidden = model(chunk, hidden)
            pred_beams[start:end] = logits.argmax(dim=-1).cpu().numpy()[0]

        # Accuracy
        gt = traj.beam_indices
        topk = traj.top_k_beams
        c1 = pred_beams == gt
        c3 = np.any(pred_beams[:, None] == topk[:, :3], axis=1)
        c5 = np.any(pred_beams[:, None] == topk[:, :5], axis=1)
        all_correct1.extend(c1.tolist())
        all_correct3.extend(c3.tolist())
        all_correct5.extend(c5.tolist())

        # Beam switches
        switches_snn    = np.sum(np.diff(pred_beams) != 0)
        switches_oracle = np.sum(np.diff(gt) != 0)
        all_switches_snn.append(switches_snn)
        all_switches_oracle.append(switches_oracle)

        # Spectral efficiency: SE = log2(1 + SNR * gain_norm)
        gains = traj.beam_gains  # [T, n_beams]
        snr_db = 20.0
        snr_lin = 10 ** (snr_db / 10)

        def se(beam_arr):
            g = gains[np.arange(T), beam_arr]
            g_norm = g / (gains.max(axis=1) + 1e-12)
            return np.log2(1 + snr_lin * g_norm).mean()

        random_beams = np.random.randint(0, traj.beam_gains.shape[1], T)
        se_snn_all.append(se(pred_beams))
        se_oracle_all.append(se(gt))
        se_random_all.append(se(random_beams))

    metrics.top1_acc = float(np.mean(all_correct1))
    metrics.top3_acc = float(np.mean(all_correct3))
    metrics.top5_acc = float(np.mean(all_correct5))
    metrics.beam_switch_rate = float(np.mean(all_switches_snn) / trajectories[0].n_steps)
    metrics.avg_switches_per_traj = float(np.mean(all_switches_snn))
    metrics.avg_se_snn    = float(np.mean(se_snn_all))
    metrics.avg_se_oracle = float(np.mean(se_oracle_all))
    metrics.avg_se_random = float(np.mean(se_random_all))
    metrics.spectral_eff_gain = (metrics.avg_se_snn - metrics.avg_se_random) / (metrics.avg_se_random + 1e-9)

    return metrics


def print_metrics(m: EvalMetrics):
    print(f"\n{'='*55}")
    print(f"  BEAM SWITCHING EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Top-1 Accuracy:          {m.top1_acc:.4f}  ({m.top1_acc*100:.1f}%)")
    print(f"  Top-3 Accuracy:          {m.top3_acc:.4f}  ({m.top3_acc*100:.1f}%)")
    print(f"  Top-5 Accuracy:          {m.top5_acc:.4f}  ({m.top5_acc*100:.1f}%)")
    print(f"  Avg Beam Switches/Traj:  {m.avg_switches_per_traj:.1f}")
    print(f"  Switch Rate:             {m.beam_switch_rate:.4f}")
    print(f"\n  Spectral Efficiency (bits/s/Hz):")
    print(f"    Random Baseline:       {m.avg_se_random:.3f}")
    print(f"    SNN Predicted:         {m.avg_se_snn:.3f}")
    print(f"    Oracle (ground truth): {m.avg_se_oracle:.3f}")
    print(f"    SE Gain vs Random:     {m.spectral_eff_gain*100:.1f}%")
    print(f"{'='*55}")