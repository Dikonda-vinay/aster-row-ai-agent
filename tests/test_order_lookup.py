"""Unit tests for the order lookup tool.

These run with no network access and no LLM calls -- pure logic tests
against the supplied data/orders.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.tools.order_lookup import OrderLookupTool, normalize_order_id


@pytest.fixture(scope="module")
def tool() -> OrderLookupTool:
    return OrderLookupTool()


# --- normalization -----------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ORD-1007", "ORD-1007"),
        ("ord-1007", "ORD-1007"),
        ("  ORD-1007  ", "ORD-1007"),
        ("ORD-1007.", "ORD-1007"),
        ("ord-1007!", "ORD-1007"),
    ],
)
def test_normalize_harmless_variants(raw, expected):
    assert normalize_order_id(raw) == expected


# --- valid lookups -------------------------------------------------------

def test_valid_lookup_returns_ok(tool):
    result = tool.lookup("ORD-1007")
    assert result.lookup_status == "ok"
    assert result.data["order_id"] == "ORD-1007"


def test_valid_lookup_case_insensitive(tool):
    result = tool.lookup("ord-1007")
    assert result.lookup_status == "ok"
    assert result.data["order_id"] == "ORD-1007"


# --- privacy: internal fields must never appear -------------------------

INTERNAL_MARKERS = [
    "email",
    "shipping_address",
    "risk_score",
    "warehouse_note",
    "support_tags",
    "ava.morgan@example.test",
    "220 King Street",
]


def test_lookup_never_leaks_internal_fields(tool):
    result = tool.lookup("ORD-1007")
    payload_str = str(result.to_tool_output())
    for marker in INTERNAL_MARKERS:
        assert marker not in payload_str, f"leaked internal field: {marker}"


def test_lookup_never_leaks_customer_pii(tool):
    result = tool.lookup("ORD-1001")
    payload_str = str(result.to_tool_output())
    assert "customer" not in result.data  # top-level PII block must be absent entirely
    assert "Maya Reed" not in payload_str
    assert "maya.reed@example.test" not in payload_str


# --- not found / invalid -------------------------------------------------

def test_unknown_order_id(tool):
    result = tool.lookup("ORD-9999")
    assert result.lookup_status == "not_found"
    assert result.data is None


def test_missing_order_id(tool):
    result = tool.lookup("")
    assert result.lookup_status == "invalid_id"


def test_malformed_order_id_not_guessed(tool):
    result = tool.lookup("banana")
    assert result.lookup_status == "invalid_id"
    # must not silently resolve to some other real order
    assert result.data is None


# --- status precedence rules (data dictionary) ---------------------------

def test_cancelled_order_flags_stale_fields(tool):
    result = tool.lookup("ORD-1004")
    assert result.data["status"] == "cancelled"
    assert "stale" in result.guidance.lower() or "not" in result.guidance.lower()
    assert "cancelled" in result.guidance.lower()


def test_returned_order_flags_stale_fields(tool):
    result = tool.lookup("ORD-1008")
    assert result.data["status"] == "returned"
    assert "returned" in result.guidance.lower()


def test_shipped_without_eta_does_not_invent_date(tool):
    result = tool.lookup("ORD-1011")
    assert result.data["status"] == "shipped"
    assert result.data["estimated_delivery"] is None
    assert "unavailable" in result.guidance.lower() or "not calculate" in result.guidance.lower()


def test_exception_status_recommends_review(tool):
    result = tool.lookup("ORD-1010")
    assert result.data["status"] == "exception"
    assert "review" in result.guidance.lower()


def test_shipped_with_eta_has_no_special_warning_needed(tool):
    result = tool.lookup("ORD-1007")
    assert result.data["status"] == "shipped"
    assert result.data["estimated_delivery"] == "2026-08-22"
