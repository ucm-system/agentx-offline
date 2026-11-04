"""Tests for the AgentX -> kvcache.ai (ClawPerf trace) converter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_agentx_to_kvcache import convert_trace


def _trace() -> dict:
    return {
        "id": "sess-1",
        "block_size": 64,
        "requests": [
            {"t": 0.0, "type": "n", "in": 128, "hash_ids": [1, 2]},
            {
                "t": 2.0,
                "type": "subagent",
                "agent_id": "agent_0",
                "requests": [
                    {"t": 2.5, "type": "s", "in": 64, "hash_ids": [9]},
                    {"t": 3.0, "type": "n", "in": 64, "hash_ids": [9, 10]},
                ],
            },
            {"t": 5.0, "type": "s", "in": 192, "hash_ids": [1, 2, 3]},
        ],
    }


def test_flatten_and_sort_by_time() -> None:
    rows = convert_trace(_trace())
    assert [r["input_length"] for r in rows] == [128, 64, 64, 192]
    assert [r["agentx"]["t"] for r in rows] == [0.0, 2.5, 3.0, 5.0]
    assert all(r["block_size"] == 64 for r in rows)


def test_user_ids_group_session_and_subagents() -> None:
    rows = convert_trace(_trace())
    assert rows[0]["user_id"] == "sess-1"
    assert rows[1]["user_id"] == "sess-1:agent_0"
    assert rows[2]["user_id"] == "sess-1:agent_0"
    assert rows[3]["user_id"] == "sess-1"


def test_hash_ids_are_preserved() -> None:
    rows = convert_trace(_trace())
    assert rows[0]["hash_ids"] == [1, 2]
    assert rows[3]["hash_ids"] == [1, 2, 3]
