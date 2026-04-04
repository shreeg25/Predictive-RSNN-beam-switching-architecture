"""
DeepMIMO v4 Multi-File Loader
Assembles the full dataset from hundreds of per-parameter .mat files.

File naming convention:
    {param}_t{ttt}_tx{bbb}_r{rrr}.mat

where:
    param = aoa_az | aod_az | aoa_el | aod_el | delay | power |
            phase  | rx_pos | tx_pos | inter_pos | vertices | ...
    t     = time snapshot index (zero-padded, e.g. t003)
    tx    = transmitter / BS index (e.g. tx000)
    r     = UE row index (e.g. r000, r001 ...)
"""

import os
import re
import numpy as np
import scipy.io as sio
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# FILE INDEX
# ─────────────────────────────────────────────────────────────────────────────

FILE_PATTERN = re.compile(
    r'^(?P<param>[a-zA-Z0-9_]+?)_t(?P<t>\d+)_tx(?P<tx>\d+)_r(?P<r>\d+)\.mat$'
)


def build_file_index(data_dir: str) -> Dict:
    """
    Scan directory and build:
    index[param][t][tx][r] = absolute file path
    """
    data_dir = Path(data_dir)
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    n_matched = 0

    for fpath in sorted(data_dir.glob('*.mat')):
        m = FILE_PATTERN.match(fpath.name)
        if m:
            param = m.group('param')
            t     = int(m.group('t'))
            tx    = int(m.group('tx'))
            r     = int(m.group('r'))
            index[param][t][tx][r] = str(fpath)
            n_matched += 1

    print(f"[Loader] Indexed {n_matched} files | params={sorted(index.keys())}")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# SAFE MAT READER
# ─────────────────────────────────────────────────────────────────────────────

def _read_mat(fpath: str, key_hint: str = None) -> Optional[np.ndarray]:
    """
    Read a .mat file and return its primary array.
    Tries key_hint first, then falls back to first non-metadata key.
    """
    try:
        raw = sio.loadmat(fpath, squeeze_me=True)
        user_keys = [k for k in raw if not k.startswith('_')]
        if not user_keys:
            return None
        if key_hint and key_hint in user_keys:
            return np.atleast_1d(raw[key_hint]).astype(float)
        param_guess = Path(fpath).name.split('_t')[0]
        if param_guess in user_keys:
            return np.atleast_1d(raw[param_guess]).astype(float)
        best_key = max(user_keys, key=lambda k: np.array(raw[k]).size)
        val = raw[best_key]
        if isinstance(val, np.ndarray):
            return val.astype(float)
        return np.atleast_1d(float(val))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeepMIMODataset:
    user_locations: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tx_location:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    path_power:   np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    path_delay:   np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    path_phase:   np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    aoa_az:       np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    aoa_el:       np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    aod_az:       np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    aod_el:       np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    num_paths:    np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    channels:     np.ndarray = field(default_factory=lambda: np.zeros((0,4,64,1), dtype=complex))
    n_beams:      int = 64
    beam_codebook: Optional[np.ndarray] = None

    @property
    def n_users(self): return len(self.user_locations)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_deepmimo_multifile(
    data_dir: str,
    t_index:  int  = 0,
    tx_index: int  = 0,
    n_beams:  int  = 64,
    N_tx_ant: int  = 64,
    N_rx_ant: int  = 4,
    max_users: int = None,
    key_map:   dict = None,
) -> DeepMIMODataset:
    """
    Load one (time_snapshot, BS) slice from a DeepMIMO v4 multi-file dataset.
    """
    key_map = key_map or {}
    index   = build_file_index(data_dir)
    ds      = DeepMIMODataset(n_beams=n_beams)

    available_params = sorted(index.keys())
    if not available_params:
        raise ValueError(f"No matching .mat files found in: {data_dir}")

    anchor_param = available_params[0]
    available_t  = sorted(index[anchor_param].keys())
    t_key        = available_t[min(t_index, len(available_t)-1)]
    available_tx = sorted(index[anchor_param][t_key].keys())
    tx_key       = available_tx[min(tx_index, len(available_tx)-1)]

    print(f"[Loader] Loading: t={t_key:03d}  tx={tx_key:03d}  "
          f"(of {len(available_t)} snapshots, {len(available_tx)} BSs)")

    row_keys = sorted(index[anchor_param][t_key][tx_key].keys())
    if max_users:
        row_keys = row_keys[:max_users]

    print(f"[Loader] UE rows: {len(row_keys)}  "
          f"(r{row_keys[0]:03d} → r{row_keys[-1]:03d})")

    # Load each parameter across all rows
    param_arrays = {}
    for param in available_params:
        if t_key not in index[param] or tx_key not in index[param][t_key]:
            continue
        row_data = []
        for r in row_keys:
            fpath = index[param][t_key][tx_key].get(r)
            if fpath is None:
                row_data.append(None)
                continue
            arr = _read_mat(fpath, key_hint=key_map.get(param))
            row_data.append(arr)
        param_arrays[param] = row_data

    n_users = len(row_keys)
    ds.user_locations = _assemble_positions(param_arrays, n_users, 'rx_pos')

    tx_rows = param_arrays.get('tx_pos', [])
    if tx_rows and tx_rows[0] is not None:
        ds.tx_location = np.atleast_1d(tx_rows[0]).flatten()[:3]

    ds.path_power = _assemble_param(param_arrays, n_users, 'power')
    ds.path_delay = _assemble_param(param_arrays, n_users, 'delay')
    ds.path_phase = _assemble_param(param_arrays, n_users, 'phase')
    ds.aoa_az     = _assemble_param(param_arrays, n_users, 'aoa_az')
    ds.aoa_el     = _assemble_param(param_arrays, n_users, 'aoa_el')
    ds.aod_az     = _assemble_param(param_arrays, n_users, 'aod_az')
    ds.aod_el     = _assemble_param(param_arrays, n_users, 'aod_el')
    ds.num_paths  = np.array([
        int(np.count_nonzero(ds.path_power[u]))
        for u in range(n_users)
    ])

    ds.beam_codebook = _generate_dft_codebook(n_beams, N_tx_ant)
    ds.channels      = _synthesize_channels(ds, N_rx_ant, N_tx_ant)

    print(f"[Loader] ✓  users={n_users}  channels={ds.channels.shape}  "
          f"avg_paths={ds.num_paths.mean():.1f}")
    return ds


def load_deepmimo_temporal(
    data_dir: str,
    tx_index: int = 0,
    t_start:  int = 0,
    t_end:    int = None,
    n_beams:  int = 64,
    N_tx_ant: int = 64,
    N_rx_ant: int = 4,
    max_users: int = None,
) -> List[DeepMIMODataset]:
    """Load multiple time snapshots as a list — for temporal/mobility datasets."""
    index        = build_file_index(data_dir)
    anchor_param = sorted(index.keys())[0]
    all_t        = sorted(index[anchor_param].keys())
    t_slice      = all_t[t_start:t_end]
    print(f"[Loader] Loading {len(t_slice)} snapshots (tx={tx_index:03d})")

    datasets = []
    for i, _ in enumerate(t_slice):
        print(f"  [{i+1}/{len(t_slice)}]", end='\r')
        ds = load_deepmimo_multifile(
            data_dir, t_index=i, tx_index=tx_index,
            n_beams=n_beams, N_tx_ant=N_tx_ant, N_rx_ant=N_rx_ant,
            max_users=max_users,
        )
        datasets.append(ds)
    print(f"\n[Loader] ✓ {len(datasets)} snapshots loaded")
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_positions(param_arrays, n_users, param_key):
    rows = param_arrays.get(param_key, [])
    locs = np.zeros((n_users, 3))
    for i, arr in enumerate(rows[:n_users]):
        if arr is None:
            continue
        arr = arr.flatten().real
        n = min(len(arr), 3)
        locs[i, :n] = arr[:n]
    return locs


def _assemble_param(param_arrays, n_users, param_key, max_paths=25):
    rows = param_arrays.get(param_key, [])
    out  = np.zeros((n_users, max_paths))
    for i, arr in enumerate(rows[:n_users]):
        if arr is None:
            continue
        arr = np.atleast_1d(arr).flatten().real
        n = min(len(arr), max_paths)
        out[i, :n] = arr[:n]
    return out


def _generate_dft_codebook(n_beams, N_tx):
    angles   = np.linspace(-np.pi/2, np.pi/2, n_beams)
    codebook = np.zeros((n_beams, N_tx), dtype=complex)
    for i, a in enumerate(angles):
        idx = np.arange(N_tx)
        codebook[i] = np.exp(1j * np.pi * idx * np.sin(a)) / np.sqrt(N_tx)
    return codebook


def _ula_steering(N, az_deg, el_deg):
    az    = np.deg2rad(az_deg)
    el    = np.deg2rad(el_deg)
    phase = 2 * np.pi * 0.5 * np.arange(N) * np.cos(el) * np.sin(az)
    return np.exp(1j * phase) / np.sqrt(N)


def _synthesize_channels(ds: DeepMIMODataset, N_rx, N_tx):
    n = ds.n_users
    H = np.zeros((n, N_rx, N_tx, 1), dtype=complex)
    for u in range(n):
        n_p = max(1, int(ds.num_paths[u]))
        for p in range(n_p):
            pwr   = ds.path_power[u, p] if ds.path_power.shape[1] > p else -100
            phase = ds.path_phase[u, p] if ds.path_phase.shape[1] > p else 0
            az_d  = ds.aod_az[u, p]    if ds.aod_az.shape[1]    > p else 0
            el_d  = ds.aod_el[u, p]    if ds.aod_el.shape[1]    > p else 0
            az_a  = ds.aoa_az[u, p]    if ds.aoa_az.shape[1]    > p else 0
            el_a  = ds.aoa_el[u, p]    if ds.aoa_el.shape[1]    > p else 0
            amp   = 10 ** (pwr / 20) if pwr < 0 else np.sqrt(max(pwr, 0))
            a_tx  = _ula_steering(N_tx, az_d, el_d)
            a_rx  = _ula_steering(N_rx, az_a, el_a)
            H[u, :, :, 0] += amp * np.exp(1j * phase) * np.outer(a_rx, a_tx.conj())
    return H


def compute_beam_gains(H, codebook):
    n_users, _, _, _ = H.shape
    n_beams = codebook.shape[0]
    gains   = np.zeros((n_users, n_beams))
    for u in range(n_users):
        Hu = H[u, :, :, 0]
        for b in range(n_beams):
            gains[u, b] = float(np.abs(Hu @ codebook[b]).sum())
    return gains


def get_optimal_beams(H, codebook, top_k=5):
    gains = compute_beam_gains(H, codebook)
    return np.argsort(-gains, axis=1)[:, :top_k]


def _synthetic_grid(n_users: int) -> np.ndarray:
    """Generate a rectangular UE grid for demo/testing"""
    side = int(np.ceil(np.sqrt(n_users)))
    xs = np.linspace(0, 200, side)
    ys = np.linspace(0, 200, side)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel(), np.ones(side*side)*1.5], axis=1)
    return grid[:n_users]