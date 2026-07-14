"""
predict_live.py
----------------
Phase 8 (capture + prediction) and Phase 9 (alerting) of the IDS pipeline,
wired together into a running real-time detector.

Pipeline: Scapy sniff -> FlowTable (background thread aggregates into
5-second flow windows) -> on flow completion, scale features with the
SAME scaler fit during training -> predict with the best saved model ->
if malicious, raise an alert through every configured channel.

Run: `sudo python src/live/predict_live.py`  (root/CAP_NET_RAW needed to
sniff packets on Linux -- see README.md "Deployment" for the setcap
alternative to running as root).

This runs until Ctrl+C. It's intentionally single-file glue code: capture,
scoring, and alerting are all imported from their own modules above, so
each piece stays testable in isolation.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.live.flow_builder import FlowTable
from src.models.model_registry import load_best_model, load_scaler_and_columns
from src.alerts.alert_manager import AlertManager

logger = get_logger(__name__)


class LiveIDS:
    def __init__(self):
        self.cfg = get_config()
        self.model, self.model_name = load_best_model()
        self.scaler, self.feature_columns = load_scaler_and_columns()
        self.alert_manager = AlertManager(self.cfg)
        self.flow_table = FlowTable(on_flow_complete=self._on_flow_complete, cfg=self.cfg)

        self.stats = {"flows_analyzed": 0, "alerts_raised": 0}
        self._stats_lock = threading.Lock()

    # ---------- flow -> prediction ----------

    def _on_flow_complete(self, flow_key, feature_dict: dict):
        src_ip, dst_ip, src_port, dst_port, proto = flow_key

        # Build a single-row DataFrame with EXACTLY the columns/order the
        # scaler and model expect. Missing columns (features we don't
        # compute live, e.g. some CICFlowMeter statistics) default to 0 --
        # acceptable for a teaching-grade live detector; document this
        # limitation in README.md "Known Limitations".
        row = {col: feature_dict.get(col, 0) for col in self.feature_columns}
        X = pd.DataFrame([row], columns=self.feature_columns)
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0][1] if hasattr(self.model, "predict_proba") else float(pred)

        with self._stats_lock:
            self.stats["flows_analyzed"] += 1

        if pred == 1:
            with self._stats_lock:
                self.stats["alerts_raised"] += 1
            self.alert_manager.raise_alert(
                attack_type="Malicious Flow",  # binary model: use train_ml.py with binary=False for named attack types
                source_ip=src_ip, destination_ip=dst_ip,
                confidence=float(proba), model=self.model_name,
            )
        else:
            logger.debug(f"Benign flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port} (conf={proba:.3f})")

    # ---------- packet capture ----------

    def _packet_handler(self, packet):
        try:
            from scapy.layers.inet import IP, TCP, UDP
        except ImportError:
            return

        if IP not in packet:
            return
        ip_layer = packet[IP]
        length = len(packet)

        if TCP in packet:
            tcp = packet[TCP]
            flags = {
                "S": bool(tcp.flags & 0x02), "A": bool(tcp.flags & 0x10),
                "U": bool(tcp.flags & 0x20), "P": bool(tcp.flags & 0x08),
            }
            self.flow_table.add_packet(
                ip_layer.src, ip_layer.dst, tcp.sport, tcp.dport, "TCP",
                length, flags, window=int(tcp.window),
            )
        elif UDP in packet:
            udp = packet[UDP]
            self.flow_table.add_packet(
                ip_layer.src, ip_layer.dst, udp.sport, udp.dport, "UDP",
                length, {},
            )

    def run(self):
        from scapy.all import sniff

        self.flow_table.start()
        iface = self.cfg["live_capture"]["interface"]
        bpf = self.cfg["live_capture"]["bpf_filter"]

        logger.info(f"Starting live capture (interface={iface or 'default'}, filter='{bpf}') "
                    f"using model={self.model_name}. Press Ctrl+C to stop.")
        try:
            sniff(iface=iface, filter=bpf, prn=self._packet_handler, store=False)
        except KeyboardInterrupt:
            pass
        except PermissionError:
            logger.error(
                "Permission denied opening the network interface. Run with sudo, "
                "or grant CAP_NET_RAW: sudo setcap cap_net_raw+ep $(which python3)"
            )
        finally:
            self.flow_table.stop()
            logger.info(f"Stopped. Flows analyzed: {self.stats['flows_analyzed']}, "
                        f"alerts raised: {self.stats['alerts_raised']}")


if __name__ == "__main__":
    ids = LiveIDS()
    ids.run()
