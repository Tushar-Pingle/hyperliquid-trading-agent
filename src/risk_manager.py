"""Centralized risk management for the trading agent.

All safety guards are enforced here, independent of LLM decisions.
The LLM cannot override these limits — they are hard-coded checks
applied before every trade execution.
"""

import json
import logging
from datetime import datetime, timezone

from src.config_loader import CONFIG

# ---------------------------------------------------------------------------
# Coin normalisation — P1.1
# ---------------------------------------------------------------------------

# Maps lowercase(display_label) → canonical API symbol.
# Only add entries where the UI label differs from the API symbol.
_COIN_ALIASES: dict[str, str] = {
    "wtioil": "xyz:CL",      # Hyperliquid UI shows "WTIOIL" for xyz:CL
    "xyz:wtioil": "xyz:CL",  # belt-and-suspenders
}


def normalize_coin(symbol: str) -> str:
    """Return the canonical API symbol for *symbol*.

    Handles two classes of alias:
    1. ``"COIN (xyz)"`` suffix — the UI appends ``(xyz)`` to HIP-3 assets.
       Strip the suffix and prepend ``xyz:`` to get the API form.
    2. Known rename aliases (e.g. ``WTIOIL`` → ``xyz:CL``).

    Symbols that already match the API form (e.g. ``xyz:CL``, ``BTC``) pass
    through unchanged.
    """
    s = symbol.strip()
    # Strip trailing "(xyz)" marker that the Hyperliquid UI appends
    if s.lower().endswith("(xyz)"):
        base = s[:-5].strip()
        s = base if base.startswith("xyz:") else f"xyz:{base}"
    # Apply rename aliases (case-insensitive lookup)
    return _COIN_ALIASES.get(s.lower(), s)


class RiskManager:
    """Enforces risk limits on every trade before execution."""

    COOLDOWNS_PATH = "cooldowns.json"

    def __init__(self):
        self.max_position_pct = float(CONFIG.get("max_position_pct") or 10)
        self.max_loss_per_position_pct = float(CONFIG.get("max_loss_per_position_pct") or 20)
        self.max_leverage = float(CONFIG.get("max_leverage") or 10)
        self.max_total_exposure_pct = float(CONFIG.get("max_total_exposure_pct") or 50)
        self.daily_loss_circuit_breaker_pct = float(CONFIG.get("daily_loss_circuit_breaker_pct") or 10)
        self.mandatory_sl_pct = float(CONFIG.get("mandatory_sl_pct") or 5)
        self.max_concurrent_positions = int(CONFIG.get("max_concurrent_positions") or 10)
        self.min_balance_reserve_pct = float(CONFIG.get("min_balance_reserve_pct") or 20)

        # P1.1 — stacking guard
        self.allow_scale_in = bool(CONFIG.get("stacking_allow_scale_in", False))

        # P2.1 — ATR-scaled sizing thresholds (env-driven, no hardcoded magic numbers)
        self.atr_ratio_high = float(CONFIG.get("atr_ratio_high") or 1.5)
        self.atr_ratio_low = float(CONFIG.get("atr_ratio_low") or 0.7)
        self.atr_low_size_mult = float(CONFIG.get("atr_low_size_mult") or 0.5)
        self.atr_high_size_mult = float(CONFIG.get("atr_high_size_mult") or 1.0)

        # P2.6 — minimum reward:risk on entries
        self.min_rr = float(CONFIG.get("min_rr") or 1.5)

        # P2.7 — low-conviction volume gate
        self.min_vol_spike_ratio = float(CONFIG.get("min_vol_spike_ratio") or 0.5)

        # P3.2 — SL too-tight gate: R must be ≥ this fraction of ATR14_4h
        self.min_r_as_atr_fraction = float(CONFIG.get("min_r_as_atr_fraction") or 0.3)

        # P4.2 — hard-reject new entries in volatile regime
        self.regime_gate_volatile = bool(CONFIG.get("regime_gate_volatile", True))

        # P1.2 — per-asset cooldown
        self.cooldown_bars = int(CONFIG.get("cooldown_bars") or 3)
        self.interval_sec = 300.0  # default 5 m; set by set_interval() after arg parsing
        self._cooldowns: dict = {}  # {canonical_coin: {last_action_ts, last_action}}

        # P2.3 — asset-aware price precision. Set by main() via set_price_rounder().
        # Signature: (asset: str, price: float) -> float
        self._price_rounder = None

        # Daily tracking
        self.daily_high_value = None
        self.daily_high_date = None
        self.circuit_breaker_active = False
        self.circuit_breaker_date = None

    # ------------------------------------------------------------------
    # Interval setter — called once from main() after args are parsed
    # ------------------------------------------------------------------

    def set_interval(self, interval_sec: float) -> None:
        self.interval_sec = float(interval_sec)

    def set_price_rounder(self, rounder) -> None:
        """Install an asset-aware price-rounding callable (P2.3).

        ``rounder(asset, price) -> float``. When unset, ``enforce_stop_loss``
        falls back to 2-decimal rounding for backward compatibility.
        """
        self._price_rounder = rounder

    # ------------------------------------------------------------------
    # Cooldown persistence — P1.2
    # ------------------------------------------------------------------

    def load_cooldowns(self) -> None:
        """Load persisted cooldowns from disk. Best-effort — never raises.

        Creates an empty ``cooldowns.json`` on first run so the file always
        exists after startup.  Absence of the file after boot is therefore a
        real error, not an ambiguous "no trades yet" state.
        """
        try:
            with open(self.COOLDOWNS_PATH) as f:
                self._cooldowns = json.load(f)
            logging.info("RISK: Loaded cooldowns for %d asset(s)", len(self._cooldowns))
        except FileNotFoundError:
            self._cooldowns = {}
            self._save_cooldowns()  # create the file immediately
            logging.info("RISK P1.2: Created empty cooldowns.json (first run)")
        except Exception as e:
            logging.warning("RISK: Failed to load cooldowns (starting fresh): %s", e)
            self._cooldowns = {}

    def _save_cooldowns(self) -> None:
        """Persist cooldowns to disk. Best-effort — never raises, never blocks exits."""
        try:
            with open(self.COOLDOWNS_PATH, "w") as f:
                json.dump(self._cooldowns, f, default=str)
        except Exception as e:
            logging.warning("RISK: Failed to save cooldowns (non-fatal): %s", e)

    def record_cooldown(self, coin: str, action: str) -> None:
        """Record a trade action timestamp for *coin* and persist.

        Best-effort — never raises, never blocks an exit.
        """
        try:
            canonical = normalize_coin(coin)
            self._cooldowns[canonical] = {
                "last_action_ts": datetime.now(timezone.utc).isoformat(),
                "last_action": action,
            }
            self._save_cooldowns()
            logging.info("RISK P1.2: Cooldown started for %s (%s)", canonical, action)
        except Exception as e:
            logging.warning("RISK: record_cooldown failed for %s (non-fatal): %s", coin, e)

    def seed_cooldown(self, coin: str) -> None:
        """Seed a startup cooldown for an already-held position.

        Called at startup for every position on the exchange so that the bot
        cannot immediately stack or flip an existing position on the first
        cycle after a restart.  Only seeds if no cooldown is already recorded.
        """
        try:
            canonical = normalize_coin(coin)
            if canonical not in self._cooldowns:
                self._cooldowns[canonical] = {
                    "last_action_ts": datetime.now(timezone.utc).isoformat(),
                    "last_action": "startup_seed",
                }
                self._save_cooldowns()
                logging.info(
                    "RISK P1.2: Seeded startup cooldown for %s (existing position)", canonical
                )
        except Exception as e:
            logging.warning("RISK: seed_cooldown failed for %s (non-fatal): %s", coin, e)

    # ------------------------------------------------------------------
    # P1.1 — Stacking guard
    # ------------------------------------------------------------------

    def check_stacking(self, coin: str, is_buy: bool,
                        positions: list) -> tuple[bool, str]:
        """Reject any new entry when a same-direction position already exists.

        Args:
            coin:      Asset symbol from the LLM decision (may be a display label).
            is_buy:    True for a long entry, False for a short entry.
            positions: Live position list from ``account_state["positions"]``.

        Returns:
            (allowed, reason) — reason is ``""`` when allowed.
        """
        if self.allow_scale_in:
            return True, ""

        canonical = normalize_coin(coin)
        for pos in positions:
            pos_coin = normalize_coin(pos.get("coin") or "")
            if pos_coin != canonical:
                continue
            try:
                szi = float(pos.get("szi") or 0)
            except (TypeError, ValueError):
                continue
            if szi == 0:
                continue
            pos_is_long = szi > 0
            if pos_is_long == is_buy:
                direction = "long" if pos_is_long else "short"
                return False, (
                    f"stacking_blocked: {canonical} already has a {direction} "
                    f"position (szi={szi:.6f})"
                )
        return True, ""

    # ------------------------------------------------------------------
    # P2.7 — Low-conviction volume gate
    # ------------------------------------------------------------------

    def check_volume_conviction(self, trade: dict) -> tuple[bool, str]:
        """Block new entries when the entry-timeframe vol_spike_ratio is dead.

        The 45h live run repeatedly showed the LLM identifying "catastrophically
        weak volume" / "dead tape" conditions and trading anyway. The data is
        already in the LLM prompt — this gate just enforces it.

        Reads ``trade["vol_spike_ratio"]`` (the primary timeframe ratio, set by
        main.py before calling validate_trade). When the field is missing or
        unparseable, the gate is bypassed — do NOT reject for missing data,
        only for confirmed-low data.
        """
        raw = trade.get("vol_spike_ratio")
        if raw is None:
            return True, ""  # missing — bypass, don't manufacture a block
        try:
            ratio = float(raw)
        except (TypeError, ValueError):
            return True, ""
        if ratio <= 0:
            return True, ""  # zero / negative is "no data", not "low vol"
        if ratio < self.min_vol_spike_ratio:
            return False, (
                f"low_conviction_volume: {ratio:.2f} < {self.min_vol_spike_ratio} "
                f"(entry-timeframe vol_spike_ratio)"
            )
        return True, ""

    # ------------------------------------------------------------------
    # P1.2 — Cooldown guard
    # ------------------------------------------------------------------

    def check_cooldown(self, coin: str, now: datetime) -> tuple[bool, str]:
        """Reject a new entry if the asset is inside its cooldown window.

        The cooldown window is ``COOLDOWN_BARS × interval_sec`` seconds after
        the last open, close, or flip on the asset.

        Args:
            coin: Asset symbol (may be a display label).
            now:  Current UTC datetime.

        Returns:
            (allowed, reason) — reason is ``""`` when allowed.
        """
        canonical = normalize_coin(coin)
        entry = self._cooldowns.get(canonical)
        if not entry:
            return True, ""
        last_ts_str = entry.get("last_action_ts")
        if not last_ts_str:
            return True, ""
        try:
            last_ts = datetime.fromisoformat(last_ts_str)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
        except Exception:
            return True, ""  # Unparseable timestamp — don't block

        cooldown_sec = self.cooldown_bars * self.interval_sec
        elapsed = (now - last_ts).total_seconds()
        remaining = cooldown_sec - elapsed
        if remaining > 0:
            return False, (
                f"cooldown_active: {canonical} last touched {elapsed:.0f}s ago, "
                f"{remaining:.0f}s remaining "
                f"({self.cooldown_bars} bars × {self.interval_sec:.0f}s)"
            )
        return True, ""

    def _reset_daily_if_needed(self, account_value: float):
        """Reset daily high watermark at UTC day boundary."""
        today = datetime.now(timezone.utc).date()
        if self.daily_high_date != today:
            self.daily_high_value = account_value
            self.daily_high_date = today
            self.circuit_breaker_active = False
            self.circuit_breaker_date = None
        elif account_value > self.daily_high_value:
            self.daily_high_value = account_value

    # ------------------------------------------------------------------
    # Individual checks — each returns (allowed: bool, reason: str)
    # ------------------------------------------------------------------

    def check_position_size(self, alloc_usd: float, account_value: float) -> tuple[bool, str]:
        """Single position cannot exceed max_position_pct of account."""
        if account_value <= 0:
            return False, "Account value is zero or negative"
        max_alloc = account_value * (self.max_position_pct / 100.0)
        if alloc_usd > max_alloc:
            return False, (
                f"Allocation ${alloc_usd:.2f} exceeds {self.max_position_pct}% "
                f"of account (${max_alloc:.2f})"
            )
        return True, ""

    def check_total_exposure(self, positions: list[dict], new_alloc: float,
                              account_value: float) -> tuple[bool, str]:
        """Sum of all position notionals + new allocation cannot exceed max_total_exposure_pct."""
        current_exposure = 0.0
        for pos in positions:
            qty = abs(float(pos.get("quantity") or pos.get("szi") or 0))
            entry = float(pos.get("entry_price") or pos.get("entryPx") or 0)
            current_exposure += qty * entry
        total = current_exposure + new_alloc
        max_exposure = account_value * (self.max_total_exposure_pct / 100.0)
        if total > max_exposure:
            return False, (
                f"Total exposure ${total:.2f} would exceed {self.max_total_exposure_pct}% "
                f"of account (${max_exposure:.2f})"
            )
        return True, ""

    def check_leverage(self, alloc_usd: float, collateral: float,
                        current_notional: float = 0.0) -> tuple[bool, str]:
        """Aggregate account leverage (existing + new) cannot exceed max_leverage.

        C4: Computes leverage as ``(current_perp_notional + new_alloc) / collateral``
        so we never let a new trade push total leverage past the configured cap,
        even on unified accounts where perp `withdrawable` reads as zero.
        """
        if collateral <= 0:
            return False, "Collateral is zero or negative"
        projected_notional = current_notional + alloc_usd
        effective_lev = projected_notional / collateral
        if effective_lev > self.max_leverage:
            return False, (
                f"Projected total leverage {effective_lev:.2f}x (existing ${current_notional:.2f} + "
                f"new ${alloc_usd:.2f} on ${collateral:.2f} collateral) exceeds max {self.max_leverage}x"
            )
        return True, ""

    def check_daily_drawdown(self, account_value: float) -> tuple[bool, str]:
        """Activate circuit breaker if account drops max % from daily high."""
        self._reset_daily_if_needed(account_value)
        if self.circuit_breaker_active:
            return False, "Daily loss circuit breaker is active — no new trades until tomorrow (UTC)"
        if self.daily_high_value and self.daily_high_value > 0:
            drawdown_pct = ((self.daily_high_value - account_value) / self.daily_high_value) * 100
            if drawdown_pct >= self.daily_loss_circuit_breaker_pct:
                self.circuit_breaker_active = True
                self.circuit_breaker_date = datetime.now(timezone.utc).date()
                return False, (
                    f"Daily drawdown {drawdown_pct:.2f}% exceeds circuit breaker "
                    f"threshold of {self.daily_loss_circuit_breaker_pct}%"
                )
        return True, ""

    def check_concurrent_positions(self, current_count: int) -> tuple[bool, str]:
        """Limit number of simultaneous open positions."""
        if current_count >= self.max_concurrent_positions:
            return False, (
                f"Already at max concurrent positions ({self.max_concurrent_positions})"
            )
        return True, ""

    def check_balance_reserve(self, balance: float, initial_balance: float) -> tuple[bool, str]:
        """Don't trade if balance falls below reserve threshold."""
        if initial_balance <= 0:
            return True, ""
        min_balance = initial_balance * (self.min_balance_reserve_pct / 100.0)
        if balance < min_balance:
            return False, (
                f"Balance ${balance:.2f} below minimum reserve "
                f"${min_balance:.2f} ({self.min_balance_reserve_pct}% of initial)"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Stop-loss enforcement
    # ------------------------------------------------------------------

    def enforce_stop_loss(self, sl_price: float | None, entry_price: float,
                           is_buy: bool, asset: str = "") -> float:
        """Ensure every trade has a stop-loss. Auto-set if missing.

        Uses the asset-aware price rounder (P2.3) when available so a
        low-priced asset (e.g. DOGE at 0.08) doesn't get its SL rounded to 0.08
        away from a 0.0795 entry. Falls back to 2 decimals when the rounder
        isn't installed (standalone tests / boot before main wiring).
        """
        if sl_price is not None:
            return sl_price
        sl_distance = entry_price * (self.mandatory_sl_pct / 100.0)
        raw = entry_price - sl_distance if is_buy else entry_price + sl_distance
        if self._price_rounder is not None and asset:
            try:
                return float(self._price_rounder(asset, raw))
            except Exception as e:
                logging.warning("RISK P2.3: price rounder failed for %s (%s); using 2-decimal fallback", asset, e)
        return round(raw, 2)

    # ------------------------------------------------------------------
    # Force-close losing positions
    # ------------------------------------------------------------------

    def check_losing_positions(self, positions: list[dict]) -> list[dict]:
        """Return positions that should be force-closed due to excessive loss.

        P2.4: loss_pct is now ``|pnl| / margin``, not ``|pnl| / notional``.
        Margin = notional / leverage. At 10× leverage a 2% adverse price move
        is a 20% margin hit — and 20% is what MAX_LOSS_PER_POSITION_PCT now
        means. The old formula understated loss by the leverage factor and
        let positions run to a 100%+ margin loss before tripping.

        Leverage source:
          1. ``pos["leverage"]["value"]`` (cross OR isolated) — Hyperliquid
             surfaces effective leverage here for both modes
          2. Fallback when field missing/malformed: ``MAX_LEVERAGE`` from config
             (NEVER 1.0 — that would under-trigger force-close on real positions)

        Args:
            positions: List of position dicts with keys:
                coin/symbol, szi/quantity, entryPx/entry_price,
                pnl/unrealized_pnl, leverage

        Returns:
            List of positions that exceed the max loss threshold.
        """
        to_close = []
        for pos in positions:
            coin = pos.get("coin") or pos.get("symbol")
            entry_px = float(pos.get("entryPx") or pos.get("entry_price") or 0)
            size = float(pos.get("szi") or pos.get("quantity") or 0)
            pnl = float(pos.get("pnl") or pos.get("unrealized_pnl") or 0)

            if entry_px == 0 or size == 0:
                continue

            notional = abs(size) * entry_px
            if notional == 0:
                continue

            # P2.4: extract leverage with safe fallback
            leverage = None
            lev_raw = pos.get("leverage")
            if isinstance(lev_raw, dict):
                lev_val = lev_raw.get("value")
                if lev_val is not None:
                    try:
                        leverage = float(lev_val)
                    except (TypeError, ValueError):
                        leverage = None
            elif lev_raw is not None:
                try:
                    leverage = float(lev_raw)
                except (TypeError, ValueError):
                    leverage = None

            fallback_used = False
            if leverage is None or leverage <= 0:
                leverage = max(1.0, self.max_leverage)
                fallback_used = True
                logging.info(
                    "RISK P2.4: leverage field missing/malformed for %s — "
                    "fallback to MAX_LEVERAGE=%.1f",
                    coin, leverage,
                )

            margin = notional / leverage
            if margin == 0:
                continue

            loss_pct = abs(pnl / margin) * 100 if pnl < 0 else 0

            if loss_pct >= self.max_loss_per_position_pct:
                logging.warning(
                    "RISK: Force-closing %s — margin loss %.2f%% exceeds max %.2f%% "
                    "(pnl=$%.2f, margin=$%.2f, leverage=%.1fx%s)",
                    coin, loss_pct, self.max_loss_per_position_pct,
                    pnl, margin, leverage,
                    " [fallback]" if fallback_used else "",
                )
                to_close.append({
                    "coin": coin,
                    "size": abs(size),
                    "is_long": size > 0,
                    "loss_pct": round(loss_pct, 2),
                    "pnl": round(pnl, 2),
                    "margin": round(margin, 2),
                    "leverage": leverage,
                    "leverage_fallback": fallback_used,
                })
        return to_close

    # ------------------------------------------------------------------
    # Composite validation — run all checks before a trade
    # ------------------------------------------------------------------

    def validate_trade(self, trade: dict, account_state: dict,
                        initial_balance: float,
                        regime_context: dict | None = None) -> tuple[bool, str, dict]:
        """Run all safety checks on a proposed trade.

        Args:
            trade: LLM trade decision with keys:
                asset, action, allocation_usd, tp_price, sl_price
            account_state: Current account with keys:
                balance, total_value, positions
            initial_balance: Starting balance for reserve check
            regime_context: Optional regime brief for this asset (P4.2).
                Keys: regime, vol_regime, stale, etc.

        Returns:
            (allowed, reason, adjusted_trade)
            adjusted_trade may have modified sl_price if it was missing.
        """
        action = trade.get("action", "hold")
        if action == "hold":
            return True, "", trade

        coin = trade.get("asset", "")
        is_buy = action == "buy"
        positions = account_state.get("positions", [])
        now = datetime.now(timezone.utc)

        # P1.2 — cooldown check (before any other guard; exits bypass this)
        ok, reason = self.check_cooldown(coin, now)
        if not ok:
            return False, reason, trade

        # P1.1 — stacking guard (hard block, no scale-in carve-out)
        ok, reason = self.check_stacking(coin, is_buy, positions)
        if not ok:
            return False, reason, trade

        # P4.2 — volatile-regime gate (after cheap memory checks, before data-dependent checks)
        if self.regime_gate_volatile and regime_context:
            if regime_context.get("regime") == "volatile" and not regime_context.get("stale"):
                return False, "regime_blocked_volatile", trade

        # P2.7 — low-conviction volume gate (block entries on dead tape)
        ok, reason = self.check_volume_conviction(trade)
        if not ok:
            return False, reason, trade

        alloc_usd = float(trade.get("allocation_usd", 0))
        if alloc_usd <= 0:
            return False, "Zero or negative allocation", trade

        # P2.1: ATR-scaled sizing — shrink in high vol, normal in low vol.
        # No "boost in low vol" (HIGH_SIZE_MULT defaults to 1.0): the point of
        # vol-scaling is to reduce risk in chop, not add risk in calm markets.
        # atr_ratio = short-term ATR / long-term ATR (atr3_4h / atr14_4h)
        try:
            atr_ratio = float(trade.get("atr_ratio") or 0)
        except (TypeError, ValueError):
            atr_ratio = 0
        if atr_ratio > 0:
            if atr_ratio > self.atr_ratio_high:
                scale = self.atr_low_size_mult
                regime = "HIGH_VOL"
            elif atr_ratio < self.atr_ratio_low:
                scale = self.atr_high_size_mult
                regime = "LOW_VOL"
            else:
                scale = 1.0
                regime = "NORMAL_VOL"
            if scale != 1.0:
                new_alloc = alloc_usd * scale
                logging.info(
                    "RISK P2.1: atr_ratio=%.2f (%s) → scaling alloc $%.2f × %.2f = $%.2f",
                    atr_ratio, regime, alloc_usd, scale, new_alloc,
                )
                alloc_usd = new_alloc
                trade = {**trade, "allocation_usd": alloc_usd}

        # Hyperliquid minimum order size is $10
        if alloc_usd < 11.0:
            alloc_usd = 11.0
            trade = {**trade, "allocation_usd": alloc_usd}
            logging.info("RISK: Bumped allocation to $11 (Hyperliquid $10 minimum)")

        account_value = float(account_state.get("total_value", 0))
        balance = float(account_state.get("balance", 0))
        # C4: prefer effective_collateral (full spot USDC) over withdrawable
        # for leverage calculations on unified accounts.
        collateral = float(account_state.get("effective_collateral", balance) or balance)
        current_perp_notional = float(account_state.get("perp_notional", 0) or 0)
        # positions and is_buy already set above for P1.1/P1.2 checks

        # 1. Daily drawdown circuit breaker
        ok, reason = self.check_daily_drawdown(account_value)
        if not ok:
            return False, reason, trade

        # 2. Balance reserve
        ok, reason = self.check_balance_reserve(balance, initial_balance)
        if not ok:
            return False, reason, trade

        # 3. Position size limit
        ok, reason = self.check_position_size(alloc_usd, account_value)
        if not ok:
            # Cap allocation instead of rejecting
            max_alloc = account_value * (self.max_position_pct / 100.0)
            # But never below Hyperliquid's $10 minimum
            if max_alloc < 11.0:
                max_alloc = 11.0
            logging.warning("RISK: Capping allocation from $%.2f to $%.2f", alloc_usd, max_alloc)
            alloc_usd = max_alloc
            trade = {**trade, "allocation_usd": alloc_usd}

        # 4. Total exposure
        ok, reason = self.check_total_exposure(positions, alloc_usd, account_value)
        if not ok:
            return False, reason, trade

        # 5. Leverage check (C4: uses true collateral and current notional)
        ok, reason = self.check_leverage(alloc_usd, collateral, current_perp_notional)
        if not ok:
            return False, reason, trade

        # 6. Concurrent positions
        active_count = sum(
            1 for p in positions
            if abs(float(p.get("szi") or p.get("quantity") or 0)) > 0
        )
        ok, reason = self.check_concurrent_positions(active_count)
        if not ok:
            return False, reason, trade

        # 7. Enforce mandatory stop-loss
        current_price = float(trade.get("current_price", 0))
        entry_price = current_price if current_price > 0 else 1.0
        sl_price = trade.get("sl_price")
        enforced_sl = self.enforce_stop_loss(sl_price, entry_price, is_buy, asset=coin)
        if sl_price is None:
            logging.info("RISK: Auto-setting SL at %.2f (%.1f%% from entry)",
                        enforced_sl, self.mandatory_sl_pct)
        trade = {**trade, "sl_price": enforced_sl}

        # 8. P3.2 — SL too-tight gate (R < MIN_R_AS_ATR_FRACTION × ATR)
        ok, reason = self.check_sl_too_tight(trade, entry_price, is_buy)
        if not ok:
            return False, reason, trade

        # 9. P2.6 — minimum reward:risk ratio (uses tp2_price when partial TP active)
        ok, reason = self.check_min_rr(trade, entry_price, is_buy)
        if not ok:
            return False, reason, trade

        return True, "", trade

    def check_min_rr(self, trade: dict, entry_price: float,
                      is_buy: bool) -> tuple[bool, str]:
        """Reject entries whose reward:risk ratio is below MIN_RR (P2.6).

        Long  R:R = (tp - entry) / (entry - sl)
        Short R:R = (entry - tp) / (sl - entry)

        Trades without a TP defined are not gated by this check — TP is
        recommended but not mandatory (SL is). Allowing TP=None preserves
        the existing behaviour where the LLM can leave a position open-ended.
        """
        try:
            # P3.2: when partial TP is active, validate against the further target (TP2)
            tp_raw = trade.get("tp2_price") or trade.get("tp_price")
            sl_raw = trade.get("sl_price")
        except Exception:
            return True, ""
        if tp_raw is None:
            return True, ""  # no TP defined — nothing to gate against
        try:
            tp = float(tp_raw)
            sl = float(sl_raw)
            entry = float(entry_price)
        except (TypeError, ValueError):
            return True, ""  # malformed values — skip rather than block

        if is_buy:
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp

        if risk <= 0:
            # SL on wrong side of entry — separate problem, not an R:R issue
            return False, f"sl_wrong_side: entry={entry:.6f} sl={sl:.6f} ({'long' if is_buy else 'short'})"
        if reward <= 0:
            return False, f"tp_wrong_side: entry={entry:.6f} tp={tp:.6f} ({'long' if is_buy else 'short'})"

        rr = reward / risk
        if rr < self.min_rr:
            return False, (
                f"min_rr_not_met: {rr:.2f} < {self.min_rr} "
                f"(entry={entry:.6f} tp={tp:.6f} sl={sl:.6f})"
            )
        return True, ""

    def check_sl_too_tight(self, trade: dict, entry_price: float,
                           is_buy: bool) -> tuple[bool, str]:
        """Reject entries where R < MIN_R_AS_ATR_FRACTION × ATR14_4h (P3.2).

        Prevents the partial-TP system from placing TP1/TP2 so close to entry
        that spread or noise fills them immediately. Skipped when ATR is absent.
        """
        try:
            atr = trade.get("atr14_4h")
            if atr is None:
                return True, ""  # no ATR data — skip rather than block
            atr = float(atr)
            if atr <= 0:
                return True, ""
            sl_raw = trade.get("sl_price")
            if sl_raw is None:
                return True, ""  # mandatory SL not yet enforced — skip
            sl = float(sl_raw)
            entry = float(entry_price)
            R = abs(entry - sl)
            min_r = self.min_r_as_atr_fraction * atr
            if R < min_r:
                return False, (
                    f"sl_too_tight: R={R:.6f} < {self.min_r_as_atr_fraction}×ATR={min_r:.6f} "
                    f"(atr14_4h={atr:.6f})"
                )
        except (TypeError, ValueError):
            pass
        return True, ""

    def get_risk_summary(self) -> dict:
        """Return current risk parameters for inclusion in LLM context."""
        return {
            "max_position_pct": self.max_position_pct,
            "max_loss_per_position_pct": self.max_loss_per_position_pct,
            "max_leverage": self.max_leverage,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "daily_loss_circuit_breaker_pct": self.daily_loss_circuit_breaker_pct,
            "mandatory_sl_pct": self.mandatory_sl_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "min_balance_reserve_pct": self.min_balance_reserve_pct,
            "circuit_breaker_active": self.circuit_breaker_active,
            "cooldown_bars": self.cooldown_bars,
            "cooldown_sec": self.cooldown_bars * self.interval_sec,
            "stacking_blocked": not self.allow_scale_in,
        }
