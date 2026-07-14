"""
test_flow_builder.py
---------------------
Unit tests for src/live/flow_builder.py -- the packet-to-flow aggregation
logic that live capture depends on.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live.flow_builder import Flow, FlowTable


def test_flow_accumulates_forward_and_backward_packets():
    flow = Flow(key=("10.0.0.1", "10.0.0.2", 1234, 80, "TCP"), first_packet_time=time.time())
    flow.add_packet("fwd", length=500, ts=time.time(), flags={"S": True}, window=64000)
    flow.add_packet("bwd", length=300, ts=time.time(), flags={"A": True}, window=32000)

    feats = flow.to_feature_dict()
    assert feats["Total Fwd Packets"] == 1
    assert feats["Total Backward Packets"] == 1
    assert feats["Total Length of Fwd Packets"] == 500
    assert feats["Total Length of Bwd Packets"] == 300
    assert feats["SYN Flag Count"] == 1
    assert feats["ACK Flag Count"] == 1


def test_flow_feature_dict_has_no_negative_core_stats():
    """Duration, packet counts, and byte totals should never be negative --
    a regression here would silently corrupt every downstream prediction."""
    flow = Flow(key=("a", "b", 1, 2, "TCP"), first_packet_time=time.time())
    flow.add_packet("fwd", length=100, ts=time.time(), flags={})
    feats = flow.to_feature_dict()

    for key in ("Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets"):
        assert feats[key] >= 0, f"{key} was negative: {feats[key]}"


def test_flow_table_merges_bidirectional_packets_into_one_flow():
    """A request (A->B) and its reply (B->A) on the same ports must be
    recognized as ONE flow, not two -- this is the core of 5-tuple matching."""
    results = []
    ft = FlowTable(on_flow_complete=lambda k, f: results.append((k, f)))

    ft.add_packet("10.0.0.1", "10.0.0.2", 1234, 80, "TCP", 500, {"S": True})
    ft.add_packet("10.0.0.2", "10.0.0.1", 80, 1234, "TCP", 300, {"A": True})  # reply direction

    assert len(ft.flows) == 1, "forward and backward packets were split into separate flows"


def test_flow_table_keeps_separate_flows_for_different_conversations():
    results = []
    ft = FlowTable(on_flow_complete=lambda k, f: results.append((k, f)))

    ft.add_packet("10.0.0.1", "10.0.0.2", 1234, 80, "TCP", 500, {"S": True})
    ft.add_packet("10.0.0.3", "10.0.0.4", 5555, 443, "TCP", 200, {"S": True})

    assert len(ft.flows) == 2


def test_flow_feature_dict_matches_expected_schema():
    """The live feature dict must contain exactly the columns the trained
    models expect (see generate_sample_dataset.py FEATURE_COLUMNS) -- a
    mismatch here would break predict_live.py silently at inference time."""
    from src.data.generate_sample_dataset import FEATURE_COLUMNS

    flow = Flow(key=("a", "b", 1, 2, "TCP"), first_packet_time=time.time())
    flow.add_packet("fwd", length=100, ts=time.time(), flags={})
    feats = flow.to_feature_dict()

    missing = set(FEATURE_COLUMNS) - set(feats.keys())
    assert not missing, f"flow_builder is missing expected columns: {missing}"
