"""
app.py
------
Phase 10 of the IDS pipeline: Streamlit dashboard.

Shows:
  - Attack counter + total flows analyzed (from alert_manager's SQLite log)
  - Detection history table (most recent alerts)
  - Attack-type breakdown chart
  - Alerts-over-time chart
  - Model comparison table (from Phase 5's ml_model_comparison.csv)
  - Traffic statistics (benign vs malicious ratio)

Run: `streamlit run dashboard/app.py`

This dashboard reads from the SAME SQLite database that predict_live.py
writes to, so run predict_live.py in one terminal and this dashboard in
another to watch alerts appear live (Streamlit's autorefresh handles the
"live" feel via st.rerun on a timer).
"""

import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config_loader import get_config, resolve_path
from src.alerts.alert_manager import AlertManager

st.set_page_config(page_title="ML-IDS Dashboard", page_icon="🛡️", layout="wide")

cfg = get_config()


@st.cache_resource
def get_alert_manager():
    return AlertManager(cfg)


def load_model_comparison():
    path = resolve_path(cfg["paths"]["processed_data_dir"]) / "ml_model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    st.title("🛡️ ML-IDS Live Dashboard")
    st.caption("Real-time intrusion detection monitoring — reads alerts from the SQLite log written by predict_live.py")

    mgr = get_alert_manager()
    alerts = mgr.recent_alerts(limit=cfg["dashboard"]["max_rows_shown"])
    df_alerts = pd.DataFrame(alerts)

    # ---------- Top metrics ----------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Alerts Logged", len(df_alerts) if not df_alerts.empty else 0)
    with col2:
        unique_types = df_alerts["attack_type"].nunique() if not df_alerts.empty else 0
        st.metric("Distinct Attack Types", unique_types)
    with col3:
        unique_ips = df_alerts["source_ip"].nunique() if not df_alerts.empty else 0
        st.metric("Distinct Source IPs", unique_ips)
    with col4:
        avg_conf = f"{df_alerts['confidence'].mean():.1%}" if not df_alerts.empty else "n/a"
        st.metric("Avg. Confidence", avg_conf)

    st.divider()

    # ---------- Charts ----------
    left, right = st.columns(2)
    with left:
        st.subheader("Attack Type Breakdown")
        if not df_alerts.empty:
            st.bar_chart(df_alerts["attack_type"].value_counts())
        else:
            st.info("No alerts yet. Run `python src/live/predict_live.py` to start detecting.")

    with right:
        st.subheader("Alerts Over Time")
        if not df_alerts.empty:
            df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"])
            counts = df_alerts.set_index("timestamp").resample("1min").size()
            st.line_chart(counts)
        else:
            st.info("No time-series data yet.")

    st.divider()

    # ---------- Detection history table ----------
    st.subheader("Recent Detections (Packet/Flow Logs)")
    if not df_alerts.empty:
        st.dataframe(
            df_alerts[["timestamp", "attack_type", "source_ip", "destination_ip", "confidence", "model"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No detections logged yet.")

    st.divider()

    # ---------- Model comparison (from Phase 5) ----------
    st.subheader("Model Comparison (Training-Time Evaluation)")
    comparison_df = load_model_comparison()
    if comparison_df is not None:
        st.dataframe(
            comparison_df[["model", "precision", "recall", "f1", "roc_auc",
                            "detection_rate", "false_positive_rate", "train_time_s"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Run `python src/models/train_ml.py` to populate this table.")

    # ---------- Auto-refresh ----------
    st.caption(f"Auto-refreshing every {cfg['dashboard']['refresh_seconds']}s")
    time.sleep(cfg["dashboard"]["refresh_seconds"])
    st.rerun()


if __name__ == "__main__":
    main()
