"""
DeepMIMO v4 Dataset Inspector
Run this FIRST to map your dataset directory before training.

Usage:
    python inspect_dataset.py --data_dir /path/to/deepmimo/dataset
    python inspect_dataset.py --data_dir /path/to/deepmimo/dataset --peek
"""

import os
import re
import argparse
import numpy as np
import scipy.io as sio
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Filename pattern:  {param}_t{ttt}_tx{bbb}_r{rrr}.mat
# e.g.:  aoa_az_t003_tx000_r000.mat
#        delay_t006_tx000_r008.mat
# ─────────────────────────────────────────────────────────────────────────────
PATTERN = re.compile(
    r'^(?P<param>[a-zA-Z0-9_]+?)_t(?P<t>\d+)_tx(?P<tx>\d+)_r(?P<r>\d+)\.mat$'
)

# Known parameter types
PARAM_ROLES = {
    'aoa_az':    'AoA azimuth angle (degrees)',
    'aoa_el':    'AoA elevation angle (degrees)',
    'aod_az':    'AoD azimuth angle (degrees)',
    'aod_el':    'AoD elevation angle (degrees)',
    'delay':     'Path delay / ToA (seconds)',
    'power':     'Path power/gain (dBm or linear)',
    'phase':     'Path phase (radians)',
    'inter_pos': 'Interaction / scatter point positions',
    'rx_pos':    'UE (receiver) positions (x,y,z)',
    'tx_pos':    'BS (transmitter) positions (x,y,z)',
    'vertices':  'Scene geometry vertices',
    'channel':   'Complex channel matrix H',
    'los':       'Line-of-sight indicator',
    'num_paths': 'Number of active paths',
}


def scan_directory(data_dir: str) -> dict:
    """
    Scan all .mat files and build an index:
    {param -> {t -> {tx -> [r, ...]}}}
    """
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Directory not found: {data_dir}"

    files = list(data_dir.glob('*.mat'))
    if not files:  # try subfolders
        files = list(data_dir.glob('**/*.mat'))
    print(f"\n[Inspector] Found {len(files)} .mat files in: {data_dir}")

    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    unmatched = []

    for f in files:
        m = PATTERN.match(f.name)
        if m:
            param = m.group('param')
            t     = int(m.group('t'))
            tx    = int(m.group('tx'))
            r     = int(m.group('r'))
            index[param][t][tx].append(r)
        else:
            unmatched.append(f.name)

    # Sort all lists
    for param in index:
        for t in index[param]:
            for tx in index[param][t]:
                index[param][t][tx].sort()

    return dict(index), unmatched, files


def print_summary(index: dict, unmatched: list):
    """Print a structured summary of the dataset"""

    params      = sorted(index.keys())
    all_t       = sorted({t for p in index for t in index[p]})
    all_tx      = sorted({tx for p in index for t in index[p] for tx in index[p][t]})
    all_r       = sorted({r for p in index for t in index[p] for tx in index[p][t] for r in index[p][t][tx]})

    print(f"\n{'='*65}")
    print(f"  DEEPMIMO v4 DATASET STRUCTURE")
    print(f"{'='*65}")
    print(f"  Parameters found : {len(params)}")
    print(f"  Time snapshots   : {len(all_t)}  (t{min(all_t):03d} → t{max(all_t):03d})")
    print(f"  Transmitters (BS): {len(all_tx)}  (tx{min(all_tx):03d} → tx{max(all_tx):03d})")
    print(f"  UE row groups    : {len(all_r)}   (r{min(all_r):03d} → r{max(all_r):03d})")
    print(f"\n  {'Parameter':<16} {'Files':>6}  {'T range':>12}  {'TX':>4}  {'R':>4}  Description")
    print(f"  {'-'*65}")

    for param in params:
        t_vals  = sorted(index[param].keys())
        tx_vals = sorted({tx for t in index[param] for tx in index[param][t]})
        r_vals  = sorted({r for t in index[param] for tx in index[param][t] for r in index[param][t][tx]})
        n_files = sum(len(index[param][t][tx]) for t in index[param] for tx in index[param][t])
        t_str   = f"t{min(t_vals):03d}–t{max(t_vals):03d}"
        desc    = PARAM_ROLES.get(param, '(unknown)')
        print(f"  {param:<16} {n_files:>6}  {t_str:>12}  {len(tx_vals):>4}  {len(r_vals):>4}  {desc}")

    if unmatched:
        print(f"\n  Unmatched files ({len(unmatched)}): {unmatched[:5]}{'...' if len(unmatched)>5 else ''}")

    print(f"\n  Estimated total users ≈ {len(all_r)} rows × (users per row)")
    print(f"{'='*65}")
    return params, all_t, all_tx, all_r


def peek_files(data_dir: str, index: dict, n_peek: int = 3):
    """
    Open a few files for each parameter and print their internal keys + shapes.
    Critical for knowing the variable name inside each .mat file.
    """
    data_dir = Path(data_dir)
    print(f"\n{'='*65}")
    print(f"  PEEKING INSIDE FILES (first {n_peek} per parameter)")
    print(f"{'='*65}")

    internal_keys = {}   # param -> list of (key, shape, dtype)

    for param in sorted(index.keys()):
        t_vals = sorted(index[param].keys())
        tx_vals = sorted(index[param][t_vals[0]].keys())
        r_vals = index[param][t_vals[0]][tx_vals[0]]

        print(f"\n  ── {param} ──")
        param_keys = []

        for i, r in enumerate(r_vals[:n_peek]):
            fname = data_dir / f"{param}_t{t_vals[0]:03d}_tx{tx_vals[0]:03d}_r{r:03d}.mat"
            if not fname.exists():
                # Try zero-padded variations
                candidates = list(data_dir.glob(f"{param}_t*_tx*_r{r:03d}.mat"))
                fname = candidates[0] if candidates else fname

            try:
                raw = sio.loadmat(str(fname), squeeze_me=True)
                user_keys = {k: v for k, v in raw.items() if not k.startswith('_')}
                for k, v in user_keys.items():
                    arr = np.atleast_1d(v)
                    shape_str = str(arr.shape)
                    dtype_str = str(arr.dtype)
                    print(f"    [{i}] key='{k}'  shape={shape_str}  dtype={dtype_str}"
                          f"  val_range=[{arr.min():.4g}, {arr.max():.4g}]" if np.issubdtype(arr.dtype, np.number) else
                          f"    [{i}] key='{k}'  shape={shape_str}  dtype={dtype_str}")
                    param_keys.append({'key': k, 'shape': arr.shape, 'dtype': dtype_str})
            except Exception as e:
                print(f"    [{i}] ERROR reading {fname.name}: {e}")

        internal_keys[param] = param_keys

    print(f"\n{'='*65}")
    return internal_keys


def generate_config(index: dict, internal_keys: dict, data_dir: str, out_path: str = 'dataset_config.py'):
    """
    Auto-generate a dataset_config.py with the discovered structure.
    This is imported by the main loader.
    """
    params = sorted(index.keys())
    all_t  = sorted({t for p in index for t in index[p]})
    all_tx = sorted({tx for p in index for t in index[p] for tx in index[p][t]})
    all_r  = sorted({r for p in index for t in index[p] for tx in index[p][t] for r in index[p][t][tx]})

    # Infer internal key per parameter (most common key found)
    key_map = {}
    for param, key_list in internal_keys.items():
        if key_list:
            # Pick most frequent key name
            from collections import Counter
            cnt = Counter(kd['key'] for kd in key_list)
            key_map[param] = cnt.most_common(1)[0][0]
        else:
            key_map[param] = param  # fallback: same as param name

    lines = [
        '"""',
        'Auto-generated by inspect_dataset.py',
        'Edit DATA_DIR to your actual dataset path.',
        '"""',
        '',
        f'DATA_DIR = r"{data_dir}"',
        '',
        f'# Discovered index ranges',
        f'T_INDICES  = {all_t}',
        f'TX_INDICES = {all_tx}',
        f'R_INDICES  = {all_r}',
        '',
        f'# Parameters available',
        f'PARAMS = {params}',
        '',
        '# Internal MATLAB variable name inside each .mat file',
        '# (auto-detected — verify and correct if needed)',
        f'PARAM_KEY_MAP = {key_map}',
        '',
        '# Which parameters to use as SNN features',
        "FEATURE_PARAMS = ['power', 'aoa_az', 'aod_az', 'delay', 'rx_pos']",
        '',
        '# Beam codebook size',
        'N_BEAMS = 64',
        '',
        '# Number of BS antennas (adjust to your scenario)',
        'N_TX_ANTENNAS = 64',
        'N_RX_ANTENNAS = 4',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n[Inspector] ✓ Config written to: {out_path}")
    print(f"  → Review PARAM_KEY_MAP and DATA_DIR before running main.py")


def main():
    parser = argparse.ArgumentParser(description='DeepMIMO v4 Dataset Inspector')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--peek',     action='store_true',      help='Open files and print internal keys')
    parser.add_argument('--config',   type=str, default='dataset_config.py', help='Output config file path')
    args = parser.parse_args()

    index, unmatched, files = scan_directory(args.data_dir)
    params, all_t, all_tx, all_r = print_summary(index, unmatched)

    internal_keys = {}
    if args.peek:
        internal_keys = peek_files(args.data_dir, index, n_peek=2)

    generate_config(index, internal_keys, args.data_dir, out_path=args.config)
    print(f"\n[Inspector] Next step:")
    print(f"  python inspect_dataset.py --data_dir {args.data_dir} --peek")
    print(f"  # then:")
    print(f"  python main.py --data_dir {args.data_dir}")


if __name__ == '__main__':
    main()