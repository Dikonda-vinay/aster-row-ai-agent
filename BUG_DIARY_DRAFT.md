# Bug Diary (working notes — will be folded into README.md)

## Bug 1: order_lookup guidance phrasing didn't match required customer-facing wording

**How reproduced:** Unit test `test_shipped_without_eta_does_not_invent_date` asserted the
deterministic `guidance` string for a shipped order with a null `estimated_delivery`
(ORD-1011) contained the word "unavailable" (the exact concept
`evaluation/visible-cases.json` requires in the final customer-facing answer for the
`shipped-without-eta` case). The assertion failed.

**Root cause:** `_compute_guidance()` in `app/tools/order_lookup.py` phrased the
no-ETA case as "no delivery estimate is available" — semantically identical, but it
never uses the literal word "unavailable". Since the LLM tends to echo the tool
guidance's wording, this created a real risk of the final answer missing the exact
required phrase, which is a deterministic eval assertion (`must_include_concepts`),
not a semantic one an LLM grader would forgive.

**Fix:** Reworded the guidance to explicitly say "...the delivery estimate is
unavailable..." and instructed the model to use that exact word.

**Regression test:** `tests/test_order_lookup.py::test_shipped_without_eta_does_not_invent_date`
now asserts `"unavailable" in result.guidance.lower()`.

---

(more entries to come as we build the evaluation harness and run it against the live agent)
