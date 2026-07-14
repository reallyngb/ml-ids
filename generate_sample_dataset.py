"""
generate_sample_dataset.py
---------------------------
WHY THIS FILE EXISTS:
The real CIC-IDS2017 / CSE-CIC-IDS2018 datasets are several GB and are
downloaded from the University of New Brunswick (UNB) servers — see
README.md for links. This script generates a SYNTHETIC dataset with the
exact same column schema as real CICFlowMeter output, so you can run the
entire pipeline (Phases 3-11) right now without the download.

To switch to the real dataset later: download the CSVs, drop them in
data/raw/, and point config.yaml -> data.raw_filename at the file(s).
No other code changes are needed — every downstream script only cares
about column names, not where the data came from.

The synthetic generator creates BENIGN flows and 4 attack classes
(DoS, PortScan, Brute Force, Web Attack) with statistically distinct
feature distributions, so the ML models have real signal to learn from.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The ~30 core CICFlowMeter features most commonly used in CIC-IDS2017 work.
# Real CICFlowMeter output has ~80 columns; we use a representative subset
# so the synthetic data stays readable while exercising the full pipeline.
FEATURE_COLUMNS = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd PSH Flags", "SYN Flag Count",
    "ACK Flag Count", "URG Flag Count", "Average Packet Size",
    "Subflow Fwd Bytes", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Idle Mean",
]

ATTACK_PROFILES = {
    # Each profile shifts the mean/scale of key features relative to BENIGN,
    # mimicking real attack signatures (e.g. PortScan = tiny packets, huge count;
    # DoS = huge flow bytes/s, short duration).
    "BENIGN": dict(duration=(50000, 40000), fwd_pkts=(8, 5), pkt_len=(400, 200),
                   flow_bytes_s=(3000, 2000), syn=(1, 1)),
    "DoS": dict(duration=(500, 400), fwd_pkts=(150, 60), pkt_len=(60, 20),
                flow_bytes_s=(80000, 30000), syn=(1, 1)),
    "PortScan": dict(duration=(200, 150), fwd_pkts=(2, 1), pkt_len=(40, 10),
                      flow_bytes_s=(500, 300), syn=(1, 1)),
    "BruteForce": dict(duration=(20000, 8000), fwd_pkts=(20, 10), pkt_len=(120, 40),
                        flow_bytes_s=(4000, 1500), syn=(1, 1)),
    "WebAttack": dict(duration=(8000, 4000), fwd_pkts=(12, 6), pkt_len=(800, 300),
                       flow_bytes_s=(6000, 2500), syn=(0, 1)),
}


def _sample_flow(rng: np.random.Generator, profile: dict) -> dict:
    """Draw one synthetic flow's features from a profile's distributions.
    Uses abs()/clipping so features stay physically plausible (no negative
    durations or packet counts)."""
    duration = max(1, rng.normal(*profile["duration"]))
    fwd_pkts = max(1, int(rng.normal(*profile["fwd_pkts"])))
    bwd_pkts = max(0, int(fwd_pkts * rng.uniform(0.3, 1.2)))
    pkt_len = max(1, rng.normal(*profile["pkt_len"]))
    flow_bytes_s = max(0, rng.normal(*profile["flow_bytes_s"]))
    syn = int(rng.binomial(1, min(max(profile["syn"][0], 0), 1)))

    fwd_len_total = pkt_len * fwd_pkts
    bwd_len_total = pkt_len * bwd_pkts * rng.uniform(0.5, 1.5)

    return {
        "Flow Duration": duration,
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_len_total,
        "Total Length of Bwd Packets": bwd_len_total,
        "Fwd Packet Length Max": pkt_len * rng.uniform(1.0, 1.8),
        "Fwd Packet Length Min": pkt_len * rng.uniform(0.2, 0.8),
        "Fwd Packet Length Mean": pkt_len,
        "Bwd Packet Length Max": pkt_len * rng.uniform(0.8, 1.6),
        "Bwd Packet Length Min": pkt_len * rng.uniform(0.1, 0.6),
        "Bwd Packet Length Mean": pkt_len * rng.uniform(0.6, 1.1),
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": (fwd_pkts + bwd_pkts) / (duration / 1e6 + 1e-6),
        "Flow IAT Mean": duration / max(fwd_pkts, 1),
        "Flow IAT Std": rng.uniform(0, duration / 4),
        "Fwd IAT Mean": duration / max(fwd_pkts, 1),
        "Bwd IAT Mean": duration / max(bwd_pkts, 1) if bwd_pkts else 0,
        "Fwd PSH Flags": int(rng.binomial(1, 0.3)),
        "SYN Flag Count": syn,
        "ACK Flag Count": int(rng.binomial(1, 0.7)),
        "URG Flag Count": int(rng.binomial(1, 0.05)),
        "Average Packet Size": pkt_len,
        "Subflow Fwd Bytes": fwd_len_total,
        "Subflow Bwd Bytes": bwd_len_total,
        "Init_Win_bytes_forward": int(rng.integers(0, 65535)),
        "Init_Win_bytes_backward": int(rng.integers(0, 65535)),
        "act_data_pkt_fwd": max(0, fwd_pkts - 1),
        "min_seg_size_forward": int(rng.integers(20, 40)),
        "Active Mean": rng.uniform(0, duration / 2),
        "Idle Mean": rng.uniform(0, duration / 3),
    }


def generate(n_rows: int = 20000, benign_ratio: float = 0.75, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic flow dataset. benign_ratio controls class balance
    (real CIC-IDS2017 is similarly imbalanced, ~80% benign)."""
    rng = np.random.default_rng(seed)
    attack_labels = [l for l in ATTACK_PROFILES if l != "BENIGN"]

    n_benign = int(n_rows * benign_ratio)
    n_attack = n_rows - n_benign
    per_attack = n_attack // len(attack_labels)

    rows, labels = [], []
    for _ in range(n_benign):
        rows.append(_sample_flow(rng, ATTACK_PROFILES["BENIGN"]))
        labels.append("BENIGN")
    for label in attack_labels:
        for _ in range(per_attack):
            rows.append(_sample_flow(rng, ATTACK_PROFILES[label]))
            labels.append(label)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df["Label"] = labels

    # Add the metadata columns real CICFlowMeter output has, so drop_columns
    # logic in preprocessing.py has something real to remove.
    df.insert(0, "Flow ID", [f"flow_{i}" for i in range(len(df))])
    df.insert(1, "Source IP", [f"10.0.0.{rng.integers(1,254)}" for _ in range(len(df))])
    df.insert(2, "Destination IP", [f"192.168.1.{rng.integers(1,254)}" for _ in range(len(df))])
    df.insert(3, "Timestamp", pd.date_range("2026-01-01", periods=len(df), freq="s"))

    # Inject a few missing/infinite values, exactly like real CICFlowMeter output
    # (Flow Bytes/s and Flow Packets/s divide by duration and can produce inf/NaN
    # when duration is 0) — this gives Phase 3 cleaning code something real to do.
    n_dirty = max(5, len(df) // 500)
    dirty_idx = rng.choice(df.index, size=n_dirty, replace=False)
    df.loc[dirty_idx[: n_dirty // 2], "Flow Bytes/s"] = np.inf
    df.loc[dirty_idx[n_dirty // 2:], "Flow Packets/s"] = np.nan

    # Shuffle rows so classes aren't grouped together
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    cfg = get_config()
    out_path = resolve_path(f"{cfg['paths']['raw_data_dir']}/{cfg['data']['raw_filename']}")
    df = generate(n_rows=20000)
    df.to_csv(out_path, index=False)
    logger.info(f"Synthetic dataset written to {out_path} ({len(df)} rows, {df['Label'].nunique()} classes)")
    logger.info(f"Class distribution:\n{df['Label'].value_counts()}")
