"""
flow_builder.py
----------------
Phase 8 of the IDS pipeline: turns a stream of individual packets into
flow-level feature vectors that match the schema our models were trained on.

WHY THIS IS THE HARD PART OF A LIVE IDS:
Our models were trained on CICFlowMeter-style FLOW features (Flow Duration,
Total Fwd Packets, Flow Bytes/s, ...), not raw packets. A single incoming
packet tells us almost nothing; the signal is in how a group of packets
between the same two endpoints behaves over time. So we must:
  1. Group packets into flows, keyed by the 5-tuple
     (src_ip, dst_ip, src_port, dst_port, protocol)
  2. Track packet timing, sizes, and TCP flags per flow
  3. Every `flow_timeout_seconds` (config.yaml -> live_capture), "close" any
     flow that's gone quiet and compute its aggregate features
  4. Hand the finished feature vector to predict_live.py for classification

This is a SIMPLIFIED reimplementation of CICFlowMeter's logic (real
CICFlowMeter computes ~80 features with more nuance, e.g. separate active/
idle period detection). We compute the subset that matches FEATURE_COLUMNS
in generate_sample_dataset.py so a model trained on that schema can consume
live flows directly. If you swap in the real dataset with all 80 real
CICFlowMeter columns, extend `_finalize_flow()` accordingly.
"""

import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Flow:
    """Accumulates packet-level stats for one 5-tuple until it's finalized."""

    def __init__(self, key, first_packet_time):
        self.key = key  # (src_ip, dst_ip, src_port, dst_port, proto)
        self.start_time = first_packet_time
        self.last_seen = first_packet_time
        self.fwd_lengths = []
        self.bwd_lengths = []
        self.fwd_timestamps = []
        self.bwd_timestamps = []
        self.syn_count = 0
        self.ack_count = 0
        self.urg_count = 0
        self.psh_count = 0
        self.init_win_fwd = 0
        self.init_win_bwd = 0

    def add_packet(self, direction: str, length: int, ts: float, flags: dict, window: int = 0):
        self.last_seen = ts
        if direction == "fwd":
            self.fwd_lengths.append(length)
            self.fwd_timestamps.append(ts)
            if not self.init_win_fwd:
                self.init_win_fwd = window
        else:
            self.bwd_lengths.append(length)
            self.bwd_timestamps.append(ts)
            if not self.init_win_bwd:
                self.init_win_bwd = window

        self.syn_count += int(flags.get("S", False))
        self.ack_count += int(flags.get("A", False))
        self.urg_count += int(flags.get("U", False))
        self.psh_count += int(flags.get("P", False))

    def is_stale(self, now: float, timeout: float) -> bool:
        return (now - self.last_seen) > timeout

    def to_feature_dict(self) -> dict:
        """Compute aggregate flow features matching FEATURE_COLUMNS in
        generate_sample_dataset.py / preprocessing.py."""
        import numpy as np

        duration_us = max((self.last_seen - self.start_time) * 1e6, 1)
        fwd_pkts = len(self.fwd_lengths)
        bwd_pkts = len(self.bwd_lengths)
        all_lengths = self.fwd_lengths + self.bwd_lengths
        total_bytes = sum(all_lengths)

        def safe_stat(fn, arr, default=0):
            return fn(arr) if arr else default

        flow_iats = np.diff(sorted(self.fwd_timestamps + self.bwd_timestamps)) if (fwd_pkts + bwd_pkts) > 1 else [0]

        return {
            "Flow Duration": duration_us,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Total Length of Fwd Packets": sum(self.fwd_lengths),
            "Total Length of Bwd Packets": sum(self.bwd_lengths),
            "Fwd Packet Length Max": safe_stat(max, self.fwd_lengths),
            "Fwd Packet Length Min": safe_stat(min, self.fwd_lengths),
            "Fwd Packet Length Mean": safe_stat(lambda a: sum(a) / len(a), self.fwd_lengths),
            "Bwd Packet Length Max": safe_stat(max, self.bwd_lengths),
            "Bwd Packet Length Min": safe_stat(min, self.bwd_lengths),
            "Bwd Packet Length Mean": safe_stat(lambda a: sum(a) / len(a), self.bwd_lengths),
            "Flow Bytes/s": total_bytes / (duration_us / 1e6),
            "Flow Packets/s": (fwd_pkts + bwd_pkts) / (duration_us / 1e6),
            "Flow IAT Mean": float(np.mean(flow_iats)) * 1e6,
            "Flow IAT Std": float(np.std(flow_iats)) * 1e6,
            "Fwd IAT Mean": duration_us / max(fwd_pkts, 1),
            "Bwd IAT Mean": duration_us / max(bwd_pkts, 1) if bwd_pkts else 0,
            "Fwd PSH Flags": self.psh_count,
            "SYN Flag Count": self.syn_count,
            "ACK Flag Count": self.ack_count,
            "URG Flag Count": self.urg_count,
            "Average Packet Size": total_bytes / max(fwd_pkts + bwd_pkts, 1),
            "Subflow Fwd Bytes": sum(self.fwd_lengths),
            "Subflow Bwd Bytes": sum(self.bwd_lengths),
            "Init_Win_bytes_forward": self.init_win_fwd,
            "Init_Win_bytes_backward": self.init_win_bwd,
            "act_data_pkt_fwd": max(fwd_pkts - 1, 0),
            "min_seg_size_forward": 20,  # placeholder: real value needs TCP option parsing
            "Active Mean": duration_us / 2,
            "Idle Mean": 0,
        }


class FlowTable:
    """Thread-safe collection of in-progress flows, with a background
    thread that finalizes stale flows every `flow_timeout_seconds`."""

    def __init__(self, on_flow_complete, cfg: dict = None):
        self.cfg = cfg or get_config()
        self.timeout = self.cfg["live_capture"]["flow_timeout_seconds"]
        self.flows: dict = {}
        self.lock = threading.Lock()
        self.on_flow_complete = on_flow_complete  # callback(flow_key, feature_dict)
        self._stop_event = threading.Event()
        self._sweeper_thread = threading.Thread(target=self._sweep_loop, daemon=True)

    def start(self):
        self._sweeper_thread.start()
        logger.info(f"FlowTable sweeper started (timeout={self.timeout}s)")

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def make_key(src_ip, dst_ip, src_port, dst_port, proto):
        """Canonical 5-tuple key: sorted by IP so both directions of the
        same conversation map to one flow (fwd = original initiator)."""
        return (src_ip, dst_ip, src_port, dst_port, proto)

    def add_packet(self, src_ip, dst_ip, src_port, dst_port, proto, length, flags, window=0):
        now = time.time()
        # Determine direction: is this the "forward" (originating) direction
        # for the canonical key, or the "backward" (reply) direction?
        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto)

        with self.lock:
            if fwd_key in self.flows:
                self.flows[fwd_key].add_packet("fwd", length, now, flags, window)
            elif bwd_key in self.flows:
                self.flows[bwd_key].add_packet("bwd", length, now, flags, window)
            else:
                flow = Flow(fwd_key, now)
                flow.add_packet("fwd", length, now, flags, window)
                self.flows[fwd_key] = flow

    def _sweep_loop(self):
        while not self._stop_event.is_set():
            time.sleep(1)
            now = time.time()
            with self.lock:
                stale_keys = [k for k, f in self.flows.items() if f.is_stale(now, self.timeout)]
                for k in stale_keys:
                    flow = self.flows.pop(k)
                    try:
                        self.on_flow_complete(k, flow.to_feature_dict())
                    except Exception as e:
                        logger.error(f"on_flow_complete callback failed for {k}: {e}")
