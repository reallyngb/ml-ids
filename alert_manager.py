"""
alert_manager.py
-----------------
Phase 9 of the IDS pipeline: fan out a detected-attack event to every
configured channel (console, CSV, SQLite, Discord, Slack).

Design: one AlertManager instance is created once (in predict_live.py /
the dashboard) and its .raise_alert(...) method is called per detection.
Each channel is independent and wrapped in try/except so a broken webhook
URL never crashes the detection loop -- an IDS silently going down because
a Slack webhook 404'd would be worse than the missing Slack message itself.
"""

import sys
import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ANSI colour codes for console alerts -- no extra dependency needed
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


class AlertManager:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or get_config()
        self.alert_cfg = self.cfg["alerts"]

        self.csv_path = resolve_path(self.alert_cfg["csv_path"])
        self.sqlite_path = resolve_path(self.alert_cfg["sqlite_path"])

        self._init_csv()
        self._init_sqlite()

    # ---------- setup ----------

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "attack_type", "source_ip", "destination_ip",
                                  "confidence", "model"])

    def _init_sqlite(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                source_ip TEXT,
                destination_ip TEXT,
                confidence REAL,
                model TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ---------- public API ----------

    def raise_alert(self, attack_type: str, source_ip: str = "unknown",
                     destination_ip: str = "unknown", confidence: float = None,
                     model: str = "unknown"):
        """Record + broadcast one detection event across every channel."""
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp, "attack_type": attack_type,
            "source_ip": source_ip, "destination_ip": destination_ip,
            "confidence": confidence, "model": model,
        }

        if self.alert_cfg["console_colors"]:
            self._console_alert(record)
        self._csv_alert(record)
        self._sqlite_alert(record)
        self._discord_alert(record)
        self._slack_alert(record)

    # ---------- channels (each isolated so one failure doesn't break others) ----------

    def _console_alert(self, r: dict):
        try:
            conf_str = f"{r['confidence']:.2%}" if r["confidence"] is not None else "n/a"
            print(f"{_RED}[ALERT]{_RESET} {r['timestamp']} | "
                  f"{_YELLOW}{r['attack_type']}{_RESET} | "
                  f"{r['source_ip']} -> {r['destination_ip']} | "
                  f"confidence={conf_str} | model={r['model']}")
        except Exception as e:
            logger.error(f"Console alert failed: {e}")

    def _csv_alert(self, r: dict):
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([r["timestamp"], r["attack_type"], r["source_ip"],
                                  r["destination_ip"], r["confidence"], r["model"]])
        except Exception as e:
            logger.error(f"CSV alert failed: {e}")

    def _sqlite_alert(self, r: dict):
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.execute(
                "INSERT INTO alerts (timestamp, attack_type, source_ip, destination_ip, confidence, model) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (r["timestamp"], r["attack_type"], r["source_ip"], r["destination_ip"], r["confidence"], r["model"]),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"SQLite alert failed: {e}")

    def _discord_alert(self, r: dict):
        url = self.alert_cfg.get("discord_webhook_url")
        if not url:
            return
        try:
            content = (f"🚨 **{r['attack_type']}** detected\n"
                       f"`{r['source_ip']}` → `{r['destination_ip']}`\n"
                       f"confidence: {r['confidence']}, model: {r['model']}, time: {r['timestamp']}")
            requests.post(url, json={"content": content}, timeout=5)
        except Exception as e:
            logger.error(f"Discord webhook failed: {e}")

    def _slack_alert(self, r: dict):
        url = self.alert_cfg.get("slack_webhook_url")
        if not url:
            return
        try:
            text = (f":rotating_light: *{r['attack_type']}* detected — "
                    f"{r['source_ip']} → {r['destination_ip']} "
                    f"(confidence: {r['confidence']}, model: {r['model']}, time: {r['timestamp']})")
            requests.post(url, json={"text": text}, timeout=5)
        except Exception as e:
            logger.error(f"Slack webhook failed: {e}")

    # ---------- retrieval (used by the dashboard) ----------

    def recent_alerts(self, limit: int = 100):
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    # Manual smoke test: `python src/alerts/alert_manager.py`
    mgr = AlertManager()
    mgr.raise_alert("PortScan", source_ip="10.0.0.5", destination_ip="192.168.1.10",
                     confidence=0.94, model="xgboost")
    print("\nRecent alerts in SQLite:")
    for a in mgr.recent_alerts(5):
        print(a)
