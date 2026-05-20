"""Structured exit-rule evaluator (P2.2).

Replaces the text-parsing ``check_exit_condition`` that did:
    if "macd" in plan and "below" in plan: ...
with a deterministic schema. The LLM now returns ``exit_rules`` as a list
of dicts:

    [
      {"indicator": "macd_4h",  "comparator": "<", "value": -200},
      {"indicator": "rsi14_5m", "comparator": ">", "value": 80}
    ]

ANY rule that evaluates True triggers a close (OR semantics — first hard
invalidation wins). Rules whose indicator is missing from the snapshot are
treated as "cannot evaluate" and skipped, not as triggers.

Permanent backward-compat fallback: when ``exit_rules`` is empty or absent,
the position falls back to TP/SL-only behaviour (no structured exit). The
caller logs ``exit_rules_missing`` once on entry so the frequency is
auditable in the diary.
"""

import logging
from typing import Iterable

_COMPARATORS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


def evaluate_exit_rules(rules: Iterable[dict] | None,
                         snapshot: dict) -> tuple[bool, str]:
    """Return ``(should_exit, reason)`` for *rules* against *snapshot*.

    Args:
        rules:    List of ``{indicator, comparator, value}`` dicts, or None.
        snapshot: Flat dict of indicator name → current value. Keys should
                  match the ``indicator`` field of each rule.

    Returns:
        (True, "rule_matched: ...") on the first matching rule.
        (False, "") when no rule matches or rules is empty.
    """
    if not rules:
        return False, ""
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        ind = rule.get("indicator")
        comp = rule.get("comparator")
        val_raw = rule.get("value")
        if not ind or comp not in _COMPARATORS or val_raw is None:
            continue
        snap_val = snapshot.get(ind)
        if snap_val is None:
            continue  # cannot evaluate — skip, do NOT treat as trigger
        try:
            current = float(snap_val)
            target = float(val_raw)
        except (TypeError, ValueError):
            continue
        try:
            if _COMPARATORS[comp](current, target):
                return True, f"rule_matched: {ind} {comp} {target} (actual={current})"
        except Exception as e:
            logging.warning("EXIT P2.2: comparator error on rule %r: %s", rule, e)
            continue
    return False, ""


def build_snapshot(market_section: dict, current_price: float | None = None) -> dict:
    """Flatten one ``market_sections`` entry into ``{indicator_key: value}``.

    Keys produced (when the underlying value exists):
      price
      <intraday_name>_5m   for each numeric scalar under "intraday"
      <long_name>_4h       for each numeric scalar under "long_term"
    """
    snap: dict = {}
    if current_price is not None:
        try:
            snap["price"] = float(current_price)
        except (TypeError, ValueError):
            pass

    for tf_key, suffix in (("intraday", "_5m"), ("long_term", "_4h")):
        block = market_section.get(tf_key) or {}
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            if v is None or isinstance(v, (list, dict)):
                continue
            try:
                snap[f"{k}{suffix}"] = float(v)
            except (TypeError, ValueError):
                continue
    return snap
