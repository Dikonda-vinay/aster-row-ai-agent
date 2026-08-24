"""Order status lookup tool.

Design goals (straight from the assignment's failure modes):
  - The model never sees the raw orders.json. It only gets back what
    this function decides to return.
  - Internal-only fields (customer PII, risk score, warehouse notes,
    support tags) are stripped here, in code, not by asking the LLM
    nicely not to repeat them. A warehouse note containing an embedded
    "AI instruction" never reaches the model's context at all.
  - Status-precedence logic (cancelled/returned -> ignore stale ETA;
    shipped + null ETA -> say unavailable, don't invent a date;
    exception -> human review) is computed HERE, deterministically, as
    a `guidance` field. This means eval assertions on this behavior
    don't depend on the LLM reasoning correctly under a system prompt —
    the correct behavior is a property of the tool's output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app import config

ORDER_ID_RE = re.compile(r"^ORD-\d+$")

LookupStatus = Literal["ok", "not_found", "invalid_id"]


@dataclass
class OrderLookupResult:
    lookup_status: LookupStatus
    order_id_queried: str
    data: dict[str, Any] | None = None
    guidance: str = ""

    def to_tool_output(self) -> dict[str, Any]:
        """What actually gets serialized back to the model as a tool result.

        Only customer-safe fields ever appear here — see data/orders-data-dictionary.md.
        """
        payload: dict[str, Any] = {
            "lookup_status": self.lookup_status,
            "order_id_queried": self.order_id_queried,
            "guidance": self.guidance,
        }
        if self.data is not None:
            payload["order"] = self.data
        return payload


CUSTOMER_SAFE_ORDER_FIELDS = (
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
)


def normalize_order_id(raw: str) -> str:
    """Normalize harmless input differences: whitespace, case, stray punctuation.

    Does NOT attempt to guess or fuzzy-correct a substantially different ID.
    """
    cleaned = raw.strip().upper()
    cleaned = re.sub(r"[.,;:!?]+$", "", cleaned)  # trailing punctuation
    cleaned = re.sub(r"\s+", "", cleaned)  # internal stray whitespace, e.g. "ORD - 1007"
    cleaned = cleaned.replace("_", "-")
    return cleaned


def _sanitize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "final_sale": item.get("final_sale"),
        }
        for item in items
    ]


def _compute_guidance(order: dict[str, Any]) -> str:
    """Deterministic status-precedence guidance, per orders-data-dictionary.md."""
    status = order.get("status")

    if status in ("cancelled", "returned"):
        return (
            f"Order status is '{status}', which is authoritative. Do not present "
            "carrier, tracking, or estimated_delivery as evidence the order is still "
            "arriving \u2014 those fields may be stale operational leftovers. "
            f"Use customer_safe_message: {order.get('customer_safe_message')!r}"
        )

    if status == "shipped" and not order.get("estimated_delivery"):
        return (
            "Order has shipped, but the delivery estimate is unavailable. State that "
            "explicitly using the word 'unavailable'. Do not calculate or invent a date."
        )

    if status == "exception":
        return (
            "Order has an operational exception requiring support review. "
            "Recommend a human handoff; do not speculate about resolution."
        )

    if status in ("pending", "processing"):
        return (
            "Order has not shipped yet. Do not provide a carrier or tracking "
            "number \u2014 none exists yet."
        )

    return "No special handling required beyond presenting the fields as-is."


class OrderLookupTool:
    def __init__(self, orders_path: Path = config.ORDERS_PATH):
        self.orders_path = orders_path
        self._orders_by_id: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        raw = json.loads(self.orders_path.read_text(encoding="utf-8"))
        for order in raw.get("orders", []):
            self._orders_by_id[order["order_id"]] = order

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        if not raw_order_id or not raw_order_id.strip():
            return OrderLookupResult(
                lookup_status="invalid_id",
                order_id_queried=raw_order_id,
                guidance="No order ID was provided. Ask the customer for their order ID.",
            )

        normalized = normalize_order_id(raw_order_id)

        if not ORDER_ID_RE.match(normalized):
            return OrderLookupResult(
                lookup_status="invalid_id",
                order_id_queried=normalized,
                guidance=(
                    "The supplied value doesn't look like a valid order ID "
                    "(expected a format like ORD-1007). Ask the customer to double-check "
                    "it rather than guessing a different ID."
                ),
            )

        order = self._orders_by_id.get(normalized)
        if order is None:
            return OrderLookupResult(
                lookup_status="not_found",
                order_id_queried=normalized,
                guidance=(
                    "No order with this ID exists in the system. Tell the customer the "
                    "order was not found and ask them to double-check the ID or contact "
                    "support. Do not invent a status, carrier, or delivery estimate."
                ),
            )

        safe = {k: order.get(k) for k in CUSTOMER_SAFE_ORDER_FIELDS}
        safe["items"] = _sanitize_items(order.get("items", []))

        return OrderLookupResult(
            lookup_status="ok",
            order_id_queried=normalized,
            data=safe,
            guidance=_compute_guidance(order),
        )
