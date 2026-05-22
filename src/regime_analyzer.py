"""Slow-cadence market regime classifier (P4.1).

Runs every REGIME_REFRESH_MINUTES (default 60) on daily candles.
Produces a cached regime_brief.json with one entry per asset:

    {asset: {regime, trend, vol_regime, key_levels, range_position, computed_at}}

Regime labels: trending-up | trending-down | ranging | volatile

The fast loop reads this cache as read-only context injected into each prompt.
No additional LLM cost per cycle.
"""

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone


def _ema(values: list, period: int) -> list:
    """EMA seeded from the SMA of the first `period` values (standard seed approach)."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    # Seed: SMA of the first `period` bars provides a stable, bias-free starting point
    seed = sum(values[:period]) / period
    result = [seed]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1.0 - k))
    return result


def _realized_vol(closes: list, window: int) -> float | None:
    """Annualised realised volatility from log returns over `window` closes."""
    if len(closes) < window + 1:
        return None
    log_rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - window, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(log_rets) < 2:
        return None
    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
    return math.sqrt(var * 252)


def _swing_highs_lows(highs: list, lows: list, lookback: int = 2) -> tuple[list, list]:
    """Identify swing highs and lows with `lookback` bars on each side."""
    sh, sl = [], []
    n = len(highs)
    for i in range(lookback, n - lookback):
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            sh.append((i, highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            sl.append((i, lows[i]))
    return sh, sl


def _classify(asset: str, candles: list) -> dict:
    """Classify market regime from daily candles. Returns a regime dict."""
    if len(candles) < 60:
        return {"regime": "unknown", "error": "insufficient_candles"}

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    current_price = closes[-1]

    # EMA50 and EMA200 on daily closes
    ema50_series = _ema(closes, 50)
    ema200_series = _ema(closes, 200)
    ema50 = ema50_series[-1] if ema50_series else None
    ema200 = ema200_series[-1] if ema200_series else None

    # HH / LL structure over last 20 daily bars
    recent_highs = highs[-22:]
    recent_lows = lows[-22:]
    sh, sl = _swing_highs_lows(recent_highs, recent_lows, lookback=1)
    hh_count = sum(
        1 for j in range(1, len(sh))
        if sh[j][1] > sh[j - 1][1]
    )
    ll_count = sum(
        1 for j in range(1, len(sl))
        if sl[j][1] < sl[j - 1][1]
    )

    # 30-day range position
    recent_30_highs = highs[-30:]
    recent_30_lows = lows[-30:]
    high_30d = max(recent_30_highs) if recent_30_highs else current_price
    low_30d = min(recent_30_lows) if recent_30_lows else current_price
    range_30d = high_30d - low_30d
    range_pos = (current_price - low_30d) / range_30d if range_30d > 0 else 0.5

    # Realised vol regime: 14d vs 60d
    vol_14 = _realized_vol(closes, 14)
    vol_60 = _realized_vol(closes, 60)
    vol_ratio = (vol_14 / vol_60) if (vol_14 and vol_60 and vol_60 > 0) else None
    if vol_ratio is None:
        vol_regime = "unknown"
    elif vol_ratio > 1.3:
        vol_regime = "expanding"
    elif vol_ratio < 0.7:
        vol_regime = "compressing"
    else:
        vol_regime = "normal"

    # Key levels: nearest swing high (resistance) and swing low (support) in 60d
    sh_60, sl_60 = _swing_highs_lows(highs[-62:], lows[-62:], lookback=2)
    resistance = min(
        (p for _, p in sh_60 if p > current_price),
        default=round(current_price * 1.05, 4)
    )
    support = max(
        (p for _, p in sl_60 if p < current_price),
        default=round(current_price * 0.95, 4)
    )

    # EMA trend direction
    if ema50 is not None and ema200 is not None:
        ema_diff_pct = abs(ema50 - ema200) / ema200 * 100
        if ema50 > ema200 and ema_diff_pct > 1.0:
            trend = "up"
        elif ema50 < ema200 and ema_diff_pct > 1.0:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "unknown"

    # 2-sigma deviation from EMA50 for volatile detection
    dev_from_ema50 = None
    if ema50 and vol_14 and vol_14 > 0:
        daily_std = ema50 * (vol_14 / math.sqrt(252))
        if daily_std > 0:
            dev_from_ema50 = abs(current_price - ema50) / daily_std

    # Regime classification
    is_volatile = (
        vol_regime == "expanding"
        or (dev_from_ema50 is not None and dev_from_ema50 > 2.0)
    )
    if is_volatile:
        regime = "volatile"
    elif trend == "up" and hh_count >= ll_count and range_pos > 0.5:
        regime = "trending-up"
    elif trend == "down" and ll_count >= hh_count and range_pos < 0.5:
        regime = "trending-down"
    else:
        regime = "ranging"

    return {
        "regime": regime,
        "trend": trend,
        "vol_regime": vol_regime,
        "vol_ratio": round(vol_ratio, 3) if vol_ratio else None,
        "range_position": round(range_pos, 3),
        "hh_count": hh_count,
        "ll_count": ll_count,
        "key_levels": {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "ema50d": round(ema50, 4) if ema50 else None,
            "ema200d": round(ema200, 4) if ema200 else None,
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "stale": False,
    }


async def refresh_regime(hyperliquid, assets: list, output_path: str = "regime_brief.json") -> dict:
    """Fetch 200 daily candles per asset, classify regimes, write cache.

    Preserves the previous cached value with stale=True when a fetch fails.
    Writes atomically via a temp file to avoid partial reads by the fast loop.
    """
    # Load existing cache to preserve stale values on failure
    existing: dict = {}
    try:
        with open(output_path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    brief: dict = {}
    for asset in assets:
        try:
            candles = await hyperliquid.get_candles(asset, "1d", 200)
            if not candles or len(candles) < 60:
                raise ValueError(f"only {len(candles or [])} daily candles returned")
            brief[asset] = _classify(asset, candles)
            logging.info("P4.1 regime: %s → %s", asset, brief[asset]["regime"])
        except Exception as e:
            logging.warning("P4.1 regime fetch failed for %s: %s — using stale cache", asset, e)
            prev = existing.get(asset, {})
            if prev:
                prev = dict(prev)
                prev["stale"] = True
                brief[asset] = prev
            else:
                brief[asset] = {"regime": "unknown", "stale": True, "error": str(e)}

    # Atomic write: write to temp file then rename
    try:
        dir_name = os.path.dirname(os.path.abspath(output_path)) or "."
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_name, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(brief, tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, output_path)
    except Exception as e:
        logging.warning("P4.1: failed to write regime_brief.json: %s", e)

    return brief
