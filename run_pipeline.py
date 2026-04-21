"""
run_pipeline.py  —  6G Predictive Beam Switching via Recurrent SNN
DeepMIMO 140 GHz | SNNTorch | Multi-User Trajectories

Just run:  python run_pipeline.py
All outputs saved to the RESULTS_DIR folder below.
"""

import sys, os, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# --- IEEE PUBLICATION PLOT STYLING (LARGE FONT) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'font.size': 18,          # Base font size (was 14)
    'axes.titlesize': 20,     # Subplot titles (was 16)
    'axes.labelsize': 18,     # X/Y axis labels (was 14)
    'xtick.labelsize': 16,    # Axis tick numbers (was 12)
    'ytick.labelsize': 16,    # Axis tick numbers (was 12)
    'legend.fontsize': 16     # Legend text (was 12)
})
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
warnings.filterwarnings('ignore')

# ── CONFIGURATION — edit these ────────────────────────────────────────────────
DATA_DIR    = r"G:\Shree\6G Beam Switching enabled by SNN\6G Dataset creation\deepmimo_scenarios\O1_140"
RESULTS_DIR = r"G:\Shree\Predictive RSNN beam switching architecture\results"   # all outputs go here
N_TRAJ      = 250       # number of UE trajectories
N_STEPS     = 100      # time steps per trajectory
SEQ_LEN     = 20       # SNN input sequence length
N_BEAMS     = 64       # beam codebook size
EPOCHS      = 40       # training epochs
LR          = 5e-4     # learning rate
BATCH       = 32       # batch size
T_INDEX     = 0        # time snapshot index (0 = first available)
TX_INDEX    = 0        # BS index (0 = first)
LAMBDA_SPK  = 1e-3
LAMBDA_TOPK = 0.1
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepmimo_loader import (
    load_deepmimo_multifile, DeepMIMODataset,
    _synthetic_grid, _synthesize_channels, _generate_dft_codebook,
    compute_beam_gains
)
from trajectory_generator import generate_trajectories, trajectories_to_sequences
from snn_model import build_model
from trainer import build_dataloaders, train, evaluate_on_trajectories, print_metrics

os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_Ablation_Spk{LAMBDA_SPK}_TopK{LAMBDA_TOPK}"
RUN_DIR  = os.path.join(RESULTS_DIR, f"run_{RUN_ID}")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"\n{'='*65}")
print(f"  6G Predictive Beam Switching | R-SNN | SNNTorch")
print(f"  Results → {RUN_DIR}")
print(f"{'='*65}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASET
# ════════════════════════════════════════════════════════════════════════════

def build_demo_dataset(n_users=400, n_beams=64):
    print("[Demo] Building synthetic 140 GHz dataset ...")
    np.random.seed(42)
    ds = DeepMIMODataset(n_beams=n_beams)
    n_p = 5
    ds.user_locations = _synthetic_grid(n_users)
    ds.path_power  = np.random.uniform(-90, -50, (n_users, n_p))
    ds.path_delay  = np.random.exponential(5e-9,  (n_users, n_p))
    ds.path_phase  = np.random.uniform(-np.pi, np.pi, (n_users, n_p))
    ds.aod_az      = np.random.uniform(-60, 60,   (n_users, n_p))
    ds.aod_el      = np.random.uniform(-30, 30,   (n_users, n_p))
    ds.aoa_az      = np.random.uniform(-60, 60,   (n_users, n_p))
    ds.aoa_el      = np.random.uniform(-30, 30,   (n_users, n_p))
    ds.num_paths   = np.full(n_users, n_p)
    ds.beam_codebook = _generate_dft_codebook(n_beams, 64)
    ds.channels    = _synthesize_channels(ds, N_rx=4, N_tx=64)
    return ds

print(f"\n[Step 1] Loading dataset ...")
if os.path.isdir(DATA_DIR):
    ds = load_deepmimo_multifile(DATA_DIR, t_index=T_INDEX, tx_index=TX_INDEX, n_beams=N_BEAMS)
else:
    print(f"  WARNING: DATA_DIR not found → using synthetic demo data")
    ds = build_demo_dataset(n_users=400, n_beams=N_BEAMS)

# --- MASTER DATA SCRUBBER ---
print("\n[Step 1.5] Master Purge: Eradicating NaN/Inf artifacts from all MATLAB tensors...")
ds.user_locations = np.nan_to_num(ds.user_locations, nan=0.0, posinf=0.0, neginf=0.0)
ds.tx_location = np.nan_to_num(ds.tx_location, nan=0.0, posinf=0.0, neginf=0.0)
ds.path_power = np.nan_to_num(ds.path_power, nan=-200.0, posinf=-200.0, neginf=-200.0)
for attr in ['path_delay', 'path_phase', 'aod_az', 'aod_el', 'aoa_az', 'aoa_el']:
    setattr(ds, attr, np.nan_to_num(getattr(ds, attr), nan=0.0, posinf=0.0, neginf=0.0))
ds.channels = _synthesize_channels(ds, N_rx=4, N_tx=64)
ds.channels = np.nan_to_num(ds.channels, nan=0.0, posinf=0.0, neginf=0.0)
# ----------------------------

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — GENERATE TRAJECTORIES
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 2] Generating {N_TRAJ} trajectories × {N_STEPS} steps ...")
trajectories = generate_trajectories(ds, n_trajectories=N_TRAJ,
                                     n_steps=N_STEPS, dt=0.5, top_k=5, seed=42)


# ════════════════════════════════════════════════════════════════════════════
# PLOT A — TRAJECTORY DIAGRAM  (saved immediately after generation)
# ════════════════════════════════════════════════════════════════════════════

def plot_trajectory_diagram(trajectories, ds, save_path):
    MOB_COLORS = {
        'linear':      '#e74c3c',
        'random_walk': '#3498db',
        'L_shaped':    '#2ecc71',
        'circular':    '#f39c12',
        'highway':     '#9b59b6',
    }

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('white')

    gs_top = gridspec.GridSpec(1, 2, figure=fig, top=0.93, bottom=0.52, left=0.05, right=0.97, wspace=0.28)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig, top=0.46, bottom=0.05, left=0.05, right=0.97, wspace=0.32)

    # ── Panel A: All trajectories ──
    ax_main = fig.add_subplot(gs_top[0, 0])
    locs = ds.user_locations
    gains = compute_beam_gains(ds.channels, ds.beam_codebook)
    best_gain = gains.max(axis=1)
    best_norm = (best_gain - best_gain.min()) / (np.ptp(best_gain) + 1e-9)

    # Use a grayscale/blues colormap for background to make colored lines pop
    sc = ax_main.scatter(locs[:, 0], locs[:, 1], c=best_norm, cmap='Blues', s=12, alpha=0.3, zorder=1)
    plt.colorbar(sc, ax=ax_main, label='Normalised Best Beam Gain', pad=0.01)

    bx, by = ds.tx_location[0], ds.tx_location[1]
    ax_main.scatter(bx, by, s=300, marker='^', color='darkblue', edgecolors='black', lw=1.5, zorder=10)
    ax_main.annotate('BS', (bx, by), textcoords='offset points', xytext=(6, 6), color='darkblue', fontsize=12, fontweight='bold')

    for traj in trajectories:
        c = MOB_COLORS.get(traj.mobility_type, 'black')
        pos = traj.positions
        ax_main.plot(pos[:, 0], pos[:, 1], color=c, lw=1.5, alpha=0.85, zorder=3)
        ax_main.scatter(*pos[0, :2],  s=20, color=c, zorder=5, marker='o')
        ax_main.scatter(*pos[-1, :2], s=40, color='black', zorder=6, marker='x', lw=1.2)
        mid = len(pos) // 2
        if mid > 0:
            dx = pos[mid, 0] - pos[mid-1, 0]
            dy = pos[mid, 1] - pos[mid-1, 1]
            ax_main.annotate('', xy=(pos[mid,0]+dx, pos[mid,1]+dy), xytext=(pos[mid,0], pos[mid,1]), arrowprops=dict(arrowstyle='->', color=c, lw=1.5))

    patches = [mpatches.Patch(color=v, label=k.replace('_',' ').title()) for k, v in MOB_COLORS.items()]
    patches.append(mpatches.Patch(color='darkblue', label='Base Station'))
    ax_main.legend(handles=patches, loc='upper right', facecolor='white', edgecolor='black')
    ax_main.set_title('All 50 UE Trajectories  ○=start  ✕=end', fontweight='bold', pad=8)
    ax_main.set_xlabel('x (m)'); ax_main.set_ylabel('y (m)')

    # ── Panel B: Per-trajectory beam switch count ──
    ax_sw = fig.add_subplot(gs_top[0, 1])
    sw_counts = [np.sum(np.diff(t.beam_indices) != 0) for t in trajectories]
    mob_types = [t.mobility_type for t in trajectories]
    bar_colors = [MOB_COLORS.get(m, 'gray') for m in mob_types]
    bar_x = np.arange(len(trajectories))
    ax_sw.bar(bar_x, sw_counts, color=bar_colors, alpha=0.85, width=0.85, edgecolor='black')
    ax_sw.axhline(np.mean(sw_counts), color='black', ls='--', lw=2, label=f'Mean = {np.mean(sw_counts):.1f}')
    ax_sw.set_title('Beam Switch Count per Trajectory', fontweight='bold')
    ax_sw.set_xlabel('Trajectory Index'); ax_sw.set_ylabel('# Beam Switches')
    ax_sw.legend(facecolor='white', edgecolor='black')

    # ── Panel C: Beam index timeline ──
    ax_beam = fig.add_subplot(gs_bot[0, 0])
    sample_ids = [0, 10, 20, 30, 40]
    t_axis = np.arange(N_STEPS)
    for i, sid in enumerate(sample_ids):
        traj = trajectories[sid]
        c = MOB_COLORS.get(traj.mobility_type, 'black')
        ax_beam.step(t_axis, traj.beam_indices + (i*5), where='post', color=c, lw=2, label=f"T{sid} ({traj.mobility_type[:3]})")
    ax_beam.set_title('Beam Index vs Time (5 trajectories)', fontweight='bold')
    ax_beam.set_xlabel('Time Step'); ax_beam.set_ylabel('Beam Index')
    ax_beam.legend(facecolor='white', edgecolor='black')

    # ── Panel D: Beam gain heatmap ──
    ax_hm = fig.add_subplot(gs_bot[0, 1])
    cmap = plt.cm.viridis
    im = ax_hm.imshow(trajectories[0].beam_gains[:N_STEPS].T, aspect='auto', origin='lower', cmap=cmap)
    ax_hm.plot(np.arange(N_STEPS), trajectories[0].beam_indices, color='black', ls='--', lw=2, label='Best Beam')
    plt.colorbar(im, ax=ax_hm, label='Beam Gain')
    ax_hm.set_title('Beam Gain Heatmap — Trajectory 0', fontweight='bold')
    ax_hm.set_xlabel('Time Step'); ax_hm.set_ylabel('Beam Index')
    ax_hm.legend(facecolor='white', edgecolor='black')

    # ── Panel E: Velocity distribution ──
    ax_vel = fig.add_subplot(gs_bot[0, 2])
    vel_by_type = {}
    for traj in trajectories:
        vel_by_type.setdefault(traj.mobility_type, []).append(traj.velocity)
    for mob, vels in vel_by_type.items():
        c = MOB_COLORS.get(mob, 'black')
        ax_vel.scatter([mob.replace('_',' ')]*len(vels), vels, color=c, alpha=0.7, s=60, edgecolors='black', lw=0.5)
        ax_vel.plot([mob.replace('_',' ')]*2, [np.mean(vels)-np.std(vels), np.mean(vels)+np.std(vels)], color=c, lw=2)
        ax_vel.scatter(mob.replace('_',' '), np.mean(vels), color='black', s=100, zorder=10, marker='D')
    ax_vel.set_title('Velocity Distribution by Mobility Type', fontweight='bold')
    ax_vel.set_xlabel('Mobility Pattern'); ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.tick_params(axis='x', rotation=20)

    fig.suptitle('DeepMIMO 140 GHz | Multi-User Trajectory Generation | 6G Beam Switching', 
             fontsize=24, fontweight='bold', y=0.97)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — SEQUENCES + DATALOADERS
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 3] Preparing training sequences ...")
X, y, y_topk = trajectories_to_sequences(trajectories, seq_len=SEQ_LEN, stride=5)
n_features   = X.shape[-1]
print(f"  Feature dim: {n_features}  |  Sequences: {len(X)}")
train_loader, val_loader, test_loader = build_dataloaders(X, y, y_topk, batch_size=BATCH)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD MODEL
# ════════════════════════════════════════════════════════════════════════════

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Step 4] Building R-SNN on: {device}")
model = build_model(n_features=n_features, n_beams=N_BEAMS, device=device)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN
# ════════════════════════════════════════════════════════════════════════════

save_path = os.path.join(RUN_DIR, 'best_snn_beam.pt')
print(f"\n[Step 5] Training for up to {EPOCHS} epochs ...")
history = train(model, train_loader, val_loader,
                n_epochs=EPOCHS, lr=LR, device=device,
                patience=10, save_path=save_path,
                lambda_spk=LAMBDA_SPK, lambda_topk=LAMBDA_TOPK)


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — EVALUATE
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 6] Evaluating on all {N_TRAJ} trajectories ...")
model.load_state_dict(torch.load(save_path, map_location=device))
metrics = evaluate_on_trajectories(model, trajectories, device, seq_len=SEQ_LEN)
print_metrics(metrics)

# ════════════════════════════════════════════════════════════════════════════
# PLOT B — TRAINING CURVES + ACCURACY METRICS (PASTE THIS WHOLE BLOCK)
# ════════════════════════════════════════════════════════════════════════════

def plot_training_results(history, metrics, trajectories, save_path):
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35, top=0.92, bottom=0.08, left=0.06, right=0.97)

    def style_ax(ax):
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, color='gray')
        return ax

    ep = range(1, len(history.train_loss) + 1)

    # (1) Loss
    ax = style_ax(fig.add_subplot(gs[0, 0]))
    ax.plot(ep, history.train_loss, '#2563eb', lw=2.5, label='Train')
    ax.plot(ep, history.val_loss, '#dc2626', lw=2.5, ls='--', label='Val')
    ax.fill_between(ep, history.train_loss, history.val_loss, alpha=0.1, color='#8b5cf6')
    ax.set_title('Loss', fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(facecolor='white', edgecolor='black')

    # (2) Accuracy
    ax = style_ax(fig.add_subplot(gs[0, 1]))
    ax.plot(ep, [a*100 for a in history.train_acc], '#2563eb', lw=2.5, label='Train Top-1')
    ax.plot(ep, [a*100 for a in history.val_acc], '#dc2626', lw=2.5, ls='--', label='Val Top-1')
    ax.plot(ep, [a*100 for a in history.val_topk], '#059669', lw=2.5, ls='-.', label='Val Top-5')
    ax.set_title('Beam Prediction Accuracy', fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.legend(facecolor='white', edgecolor='black')

    # (3) Spike rate
    ax = style_ax(fig.add_subplot(gs[0, 2]))
    ax.plot(ep, [s*100 for s in history.spike_rates], '#9333ea', lw=2.5)
    ax.axhline(15, color='red', ls='--', lw=2, label='Target 15%')
    ax.axhspan(10, 20, alpha=0.15, color='#10b981')
    ax.set_title('Spike Rate (LIF Layer 2)', fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('Spike Rate (%)')
    ax.legend(facecolor='white', edgecolor='black')

    # (4) LR schedule
    ax = style_ax(fig.add_subplot(gs[0, 3]))
    ax.plot(ep, history.lr_history, '#ea580c', lw=2.5)
    ax.set_title('Learning Rate Schedule', fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.set_yscale('log')

    # (5) Top-K accuracy bars
    ax = style_ax(fig.add_subplot(gs[1, 0]))
    tk_vals = [metrics.top1_acc*100, metrics.top3_acc*100, metrics.top5_acc*100]
    tk_clrs = ['#3b82f6', '#10b981', '#f59e0b']
    bars = ax.bar(['Top-1','Top-3','Top-5'], tk_vals, color=tk_clrs, edgecolor='black', width=0.5, alpha=0.9)
    for b, v in zip(bars, tk_vals):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.1f}%', ha='center', va='bottom', color='black', fontweight='bold', fontsize=16)
    ax.set_ylim(0, 115); ax.set_title('Top-K Beam Accuracy', fontweight='bold'); ax.set_ylabel('Accuracy (%)')

    # (6) Spectral efficiency
    ax = style_ax(fig.add_subplot(gs[1, 1]))
    se_vals = [metrics.avg_se_random, metrics.avg_se_snn, metrics.avg_se_oracle]
    se_clrs = ['#ef4444', '#10b981', '#3b82f6']
    bars = ax.bar(['Random\nBaseline','R-SNN\nPredicted','Oracle\n(GT)'], se_vals, color=se_clrs, edgecolor='black', width=0.5, alpha=0.9)
    for b, v in zip(bars, se_vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', va='bottom', color='black', fontweight='bold', fontsize=16)
    gain_pct = metrics.spectral_eff_gain * 100
    ax.set_title(f'Spectral Efficiency (+{gain_pct:.1f}% vs random)', fontweight='bold'); ax.set_ylabel('SE (bits/s/Hz)')

    # (7) Beam switch count histogram
    ax = style_ax(fig.add_subplot(gs[1, 2]))
    sw = [np.sum(np.diff(t.beam_indices) != 0) for t in trajectories]
    ax.hist(sw, bins=15, color='#8b5cf6', edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(sw), color='red', ls='--', lw=2.5, label=f'Mean={np.mean(sw):.1f}')
    ax.set_title('Beam Switch Distribution', fontweight='bold'); ax.set_xlabel('# Switches'); ax.set_ylabel('Count')
    ax.legend(facecolor='white', edgecolor='black')

    # (8) Metrics summary text box
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor('white'); ax.axis('off')
    summary = (
        f"  SUMMARY\n"
        f"  {'─'*28}\n"
        f"  Top-1 Accuracy    {metrics.top1_acc*100:6.1f}%\n"
        f"  Top-3 Accuracy    {metrics.top3_acc*100:6.1f}%\n"
        f"  Top-5 Accuracy    {metrics.top5_acc*100:6.1f}%\n\n"
        f"  SE Random         {metrics.avg_se_random:6.3f} b/s/Hz\n"
        f"  SE R-SNN          {metrics.avg_se_snn:6.3f} b/s/Hz\n"
        f"  SE Oracle         {metrics.avg_se_oracle:6.3f} b/s/Hz\n"
        f"  SE Gain           +{metrics.spectral_eff_gain*100:.1f}%\n\n"
        f"  Avg Switches/Traj {metrics.avg_switches_per_traj:6.1f}\n"
        f"  Switch Rate       {metrics.beam_switch_rate:6.4f}\n\n"
        f"  Epochs trained    {len(history.train_loss):6d}\n"
        f"  Device            {'GPU' if torch.cuda.is_available() else 'CPU':>6}\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontfamily='monospace', fontsize=16, color='black', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='black', lw=1.5))

    fig.suptitle('R-SNN Training Results | DeepMIMO 140 GHz | 6G Beam Switching', fontsize=24, fontweight='bold', y=0.97)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — GENERATE PLOTS AND SAVE METRICS
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[Step 7] Generating IEEE-formatted plots...")

# 1. Trigger the Trajectory Plot
traj_plot = os.path.join(RUN_DIR, '1_trajectory_diagram.png')
plot_trajectory_diagram(trajectories, ds, traj_plot)
print(f"  [Plot] Trajectory diagram → {traj_plot}")

# 2. Trigger the Training Results Plot
result_plot = os.path.join(RUN_DIR, '2_training_results.png')
plot_training_results(history, metrics, trajectories, result_plot)
print(f"  [Plot] Training results   → {result_plot}")

# 3. Save Metrics Text
metrics_path = os.path.join(RUN_DIR, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Run ID: {RUN_ID}\n")
    f.write(f"Dataset: {DATA_DIR}\n")
    f.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n\n")
    f.write(f"=== CONFIGURATION ===\n")
    f.write(f"N_TRAJ={N_TRAJ}  N_STEPS={N_STEPS}  SEQ_LEN={SEQ_LEN}\n")
    f.write(f"N_BEAMS={N_BEAMS}  EPOCHS={EPOCHS}  LR={LR}  BATCH={BATCH}\n\n")
    f.write(f"=== BEAM PREDICTION ACCURACY ===\n")
    f.write(f"Top-1: {metrics.top1_acc*100:.2f}%\n")
    f.write(f"Top-3: {metrics.top3_acc*100:.2f}%\n")
    f.write(f"Top-5: {metrics.top5_acc*100:.2f}%\n\n")
    f.write(f"=== SPECTRAL EFFICIENCY ===\n")
    f.write(f"Random Baseline: {metrics.avg_se_random:.4f} bits/s/Hz\n")
    f.write(f"R-SNN Predicted: {metrics.avg_se_snn:.4f} bits/s/Hz\n")
    f.write(f"Oracle (GT):     {metrics.avg_se_oracle:.4f} bits/s/Hz\n")
    f.write(f"Gain vs Random:  +{metrics.spectral_eff_gain*100:.2f}%\n\n")
    f.write(f"=== BEAM SWITCHING ===\n")
    f.write(f"Avg switches/trajectory: {metrics.avg_switches_per_traj:.2f}\n")
    f.write(f"Switch rate:             {metrics.beam_switch_rate:.4f}\n\n")
    f.write(f"=== TRAINING HISTORY ===\n")
    f.write(f"Epochs completed: {len(history.train_loss)}\n")
    f.write(f"Best val loss:    {min(history.val_loss):.4f}\n")
    f.write(f"Best val Top-1:   {max(history.val_acc)*100:.2f}%\n")
    f.write(f"Best val Top-5:   {max(history.val_topk)*100:.2f}%\n")
    f.write(f"Final spike rate: {history.spike_rates[-1]*100:.2f}%\n")

print(f"  [Save] Metrics text      → {metrics_path}")

# 4. Save numpy arrays
np.save(os.path.join(RUN_DIR, 'train_loss.npy'),  np.array(history.train_loss))
np.save(os.path.join(RUN_DIR, 'val_loss.npy'),    np.array(history.val_loss))
np.save(os.path.join(RUN_DIR, 'train_acc.npy'),   np.array(history.train_acc))
np.save(os.path.join(RUN_DIR, 'val_acc.npy'),     np.array(history.val_acc))
np.save(os.path.join(RUN_DIR, 'spike_rates.npy'), np.array(history.spike_rates))
print(f"  [Save] Training arrays   → {RUN_DIR}/*.npy")

print(f"\n{'='*65}")
print(f"  ALL DONE")
print(f"  Results folder: {RUN_DIR}")
print(f"  ├── 1_trajectory_diagram.png")
print(f"  ├── 2_training_results.png")
print(f"  ├── best_snn_beam.pt")
print(f"  ├── metrics.txt")
print(f"  └── *.npy  (training history arrays)")
print(f"{'='*65}\n")