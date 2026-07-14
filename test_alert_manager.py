"""
test_alert_manager.py
----------------------
Unit tests for src/alerts/alert_manager.py. Uses a temp dir for CSV/SQLite
so tests don't pollute the real logs/ directory or depend on run order.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alerts.alert_manager import AlertManager
from src.utils.config_loader import get_config


@pytest.fixture
def alert_manager(tmp_path):
    cfg = dict(get_config())  # shallow copy so we don't mutate the real config
    cfg["alerts"] = dict(cfg["alerts"])
    cfg["alerts"]["csv_path"] = str(tmp_path / "alerts.csv")
    cfg["alerts"]["sqlite_path"] = str(tmp_path / "alerts.db")
    cfg["alerts"]["discord_webhook_url"] = ""
    cfg["alerts"]["slack_webhook_url"] = ""
    return AlertManager(cfg)


def test_raise_alert_writes_to_csv(alert_manager):
    alert_manager.raise_alert("PortScan", source_ip="10.0.0.1", destination_ip="10.0.0.2",
                               confidence=0.9, model="xgboost")
    csv_path = Path(alert_manager.csv_path)
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "PortScan" in content
    assert "10.0.0.1" in content


def test_raise_alert_writes_to_sqlite(alert_manager):
    alert_manager.raise_alert("DoS", source_ip="1.1.1.1", destination_ip="2.2.2.2",
                               confidence=0.75, model="random_forest")
    recent = alert_manager.recent_alerts(limit=10)
    assert len(recent) == 1
    assert recent[0]["attack_type"] == "DoS"
    assert recent[0]["source_ip"] == "1.1.1.1"


def test_multiple_alerts_ordered_most_recent_first(alert_manager):
    alert_manager.raise_alert("PortScan", source_ip="1.1.1.1")
    alert_manager.raise_alert("DoS", source_ip="2.2.2.2")
    recent = alert_manager.recent_alerts(limit=10)
    assert recent[0]["attack_type"] == "DoS"      # most recent first
    assert recent[1]["attack_type"] == "PortScan"


def test_disabled_webhooks_do_not_raise(alert_manager):
    """With empty webhook URLs, _discord_alert/_slack_alert should no-op
    silently rather than crash the whole raise_alert() call."""
    alert_manager.raise_alert("BruteForce", source_ip="3.3.3.3")  # should not raise
