"""High-level Hyperliquid exchange client with async retry helpers.

This module wraps the Hyperliquid `Exchange` and `Info` SDK classes to provide a
single entry point for submitting trades, managing orders, and retrieving market
state.  It normalizes retry behaviour, adds logging, and caches metadata so that
the trading agent can depend on predictable, non-blocking IO.
"""

import asyncio
import logging
import aiohttp
from typing import TYPE_CHECKING
from src.config_loader import CONFIG
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants  # For MAINNET/TESTNET
from eth_account import Account as _Account
from eth_account.signers.local import LocalAccount
from websocket._exceptions import WebSocketConnectionClosedException
import socket

if TYPE_CHECKING:
    # Type stubs for linter - eth_account's type stubs are incorrect
    class Account:
        @staticmethod
        def from_key(_private_key: str) -> LocalAccount: ...
        @staticmethod
        def from_mnemonic(_mnemonic: str) -> LocalAccount: ...
        @staticmethod
        def enable_unaudited_hdwallet_features() -> None: ...
else:
    Account = _Account

class HyperliquidAPI:
    """Facade around Hyperliquid SDK clients with async convenience methods.

    The class owns wallet credentials, connection configuration, and provides
    coroutine helpers that keep retry semantics and logging consistent across
    the trading agent.
    """

    def __init__(self):
        """Initialize wallet credentials and instantiate exchange clients.

        Raises:
            ValueError: If neither a private key nor mnemonic is present in the
                configuration.
        """
        self._meta_cache = None
        self._hip3_meta_cache = {}  # {dex_name: meta_response}
        self._perp_dex_offsets = {}  # {dex_name: asset_index_offset}
        if "hyperliquid_private_key" in CONFIG and CONFIG["hyperliquid_private_key"]:
            self.wallet = Account.from_key(CONFIG["hyperliquid_private_key"])
        elif "mnemonic" in CONFIG and CONFIG["mnemonic"]:
            Account.enable_unaudited_hdwallet_features()
            self.wallet = Account.from_mnemonic(CONFIG["mnemonic"])
        else:
            raise ValueError("Either HYPERLIQUID_PRIVATE_KEY/LIGHTER_PRIVATE_KEY or MNEMONIC must be provided")
        # Choose base URL: allow override via env-config; fallback to network selection
        network = (CONFIG.get("hyperliquid_network") or "mainnet").lower()
        base_url = CONFIG.get("hyperliquid_base_url")
        if not base_url:
            if network == "testnet":
                base_url = getattr(constants, "TESTNET_API_URL", constants.MAINNET_API_URL)
            else:
                base_url = constants.MAINNET_API_URL
        self.base_url = base_url
        # Account address: the main wallet that holds funds.
        # The agent wallet (private key) is just the authorized signer.
        self.account_address = CONFIG.get("hyperliquid_vault_address")
        # The address to query for state — main account if set, otherwise the signer
        self.query_address = self.account_address or self.wallet.address
        self._build_clients()

    def _build_clients(self):
        """Instantiate exchange and info client instances for the active base URL."""
        self.info = Info(self.base_url)
        self.exchange = Exchange(self.wallet, self.base_url, account_address=self.account_address)

    def _apply_hip3_meta_to_clients(self, meta: dict, offset: int) -> None:
        """Register HIP-3 dex assets into SDK Info objects so order placement works."""
        for info_obj in (self.info, self.exchange.info):
            if hasattr(info_obj, 'set_perp_meta'):
                try:
                    info_obj.set_perp_meta(meta, offset)
                except Exception as e:
                    logging.warning("set_perp_meta failed on info object: %s", e)

    def _reset_clients(self):
        """Recreate SDK clients after connection failures while logging failures."""
        try:
            self._build_clients()
            logging.warning("Hyperliquid clients re-instantiated after connection issue")
        except (ValueError, AttributeError, RuntimeError) as e:
            logging.error("Failed to reset Hyperliquid clients: %s", e)
            return
        # Re-apply any previously loaded HIP-3 dex metadata after the reset
        for dex, data in self._hip3_meta_cache.items():
            if isinstance(data, list) and len(data) >= 1:
                offset = self._perp_dex_offsets.get(dex, 110000)
                self._apply_hip3_meta_to_clients(data[0], offset)

    async def _retry(self, fn, *args, max_attempts: int = 3, backoff_base: float = 0.5, reset_on_fail: bool = True, to_thread: bool = True, **kwargs):
        """Retry helper with exponential backoff and optional thread offloading.

        Args:
            fn: Callable to invoke, either sync (supports `asyncio.to_thread`) or
                async depending on ``to_thread``. The callable should raise
                exceptions rather than returning sentinel values.
            *args: Positional arguments forwarded to ``fn``.
            max_attempts: Maximum number of attempts before surfacing the last
                exception.
            backoff_base: Initial delay in seconds, doubled after each failure.
            reset_on_fail: Whether to rebuild Hyperliquid clients after a
                failure.
            to_thread: If ``True`` the callable is executed in a worker thread.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            Result produced by ``fn``.

        Raises:
            Exception: Propagates any exception raised by ``fn`` after retries.
        """
        last_err = None
        for attempt in range(max_attempts):
            try:
                if to_thread:
                    return await asyncio.to_thread(fn, *args, **kwargs)
                return await fn(*args, **kwargs)
            except (WebSocketConnectionClosedException, aiohttp.ClientError, ConnectionError, TimeoutError, socket.timeout) as e:
                last_err = e
                logging.warning("HL call failed (attempt %s/%s): %s", attempt + 1, max_attempts, e)
                if reset_on_fail:
                    self._reset_clients()
                await asyncio.sleep(backoff_base * (2 ** attempt))
                continue
            except (RuntimeError, ValueError, KeyError, AttributeError) as e:
                # Unknown errors: don't spin forever, but allow a quick reset once
                last_err = e
                logging.warning("HL call unexpected error (attempt %s/%s): %s", attempt + 1, max_attempts, e)
                if reset_on_fail and attempt == 0:
                    self._reset_clients()
                    await asyncio.sleep(backoff_base)
                    continue
                break
        raise last_err if last_err else RuntimeError("Hyperliquid retry: unknown error")

    def round_size(self, asset, amount):
        """Round order size to the asset precision defined by market metadata.

        Args:
            asset: Symbol of the market whose contract size we are rounding to.
            amount: Desired contract size before rounding.

        Returns:
            The input ``amount`` rounded to the market's ``szDecimals`` precision.
        """
        # Check main dex cache first
        meta = self._meta_cache[0] if hasattr(self, '_meta_cache') and self._meta_cache else None
        if meta:
            universe = meta.get("universe", [])
            asset_info = next((u for u in universe if u.get("name") == asset), None)
            if asset_info:
                decimals = asset_info.get("szDecimals", 8)
                return round(amount, decimals)
        # Check HIP-3 dex cache
        if ":" in asset:
            dex = asset.split(":")[0]
            dex_data = self._hip3_meta_cache.get(dex) if hasattr(self, '_hip3_meta_cache') else None
            if dex_data and isinstance(dex_data, list) and len(dex_data) >= 1:
                dex_meta = dex_data[0]  # [meta_dict, asset_ctxs_list]
                universe = dex_meta.get("universe", [])
                asset_info = next((u for u in universe if u.get("name") == asset), None)
                if asset_info:
                    decimals = asset_info.get("szDecimals", 8)
                    return round(amount, decimals)
        return round(amount, 8)

    async def place_buy_order(self, asset, amount, slippage=0.01):
        """Submit a market buy order with exchange-side rounding and retry logic.

        Args:
            asset: Market symbol to open.
            amount: Contract size to open before rounding.
            slippage: Maximum acceptable slippage expressed as a decimal.

        Returns:
            Raw SDK response from :meth:`Exchange.market_open`.
        """
        amount = self.round_size(asset, amount)
        if ":" in asset:
            # HIP-3 dex asset: pass the current price explicitly so the SDK doesn't
            # try to fetch it via all_mids() which doesn't cover HIP-3 assets.
            px = await self.get_current_price(asset)
            return await self._retry(lambda: self.exchange.market_open(asset, True, amount, px, slippage))
        return await self._retry(lambda: self.exchange.market_open(asset, True, amount, None, slippage))

    async def place_sell_order(self, asset, amount, slippage=0.01):
        """Submit a market sell order with exchange-side rounding and retry logic.

        Args:
            asset: Market symbol to open.
            amount: Contract size to open before rounding.
            slippage: Maximum acceptable slippage expressed as a decimal.

        Returns:
            Raw SDK response from :meth:`Exchange.market_open`.
        """
        amount = self.round_size(asset, amount)
        if ":" in asset:
            # HIP-3 dex asset: pass the current price explicitly so the SDK doesn't
            # try to fetch it via all_mids() which doesn't cover HIP-3 assets.
            px = await self.get_current_price(asset)
            return await self._retry(lambda: self.exchange.market_open(asset, False, amount, px, slippage))
        return await self._retry(lambda: self.exchange.market_open(asset, False, amount, None, slippage))

    async def place_limit_buy(self, asset, amount, limit_price, tif="Gtc"):
        """Submit a limit buy order.

        Args:
            asset: Market symbol.
            amount: Contract size before rounding.
            limit_price: Limit price for the order.
            tif: Time-in-force — "Gtc" (good-til-canceled), "Ioc" (immediate-or-cancel),
                 or "Alo" (add-liquidity-only / post-only).

        Returns:
            Raw SDK response from :meth:`Exchange.order`.
        """
        amount = self.round_size(asset, amount)
        order_type = {"limit": {"tif": tif}}
        return await self._retry(lambda: self.exchange.order(asset, True, amount, limit_price, order_type))

    async def place_limit_sell(self, asset, amount, limit_price, tif="Gtc"):
        """Submit a limit sell order.

        Args:
            asset: Market symbol.
            amount: Contract size before rounding.
            limit_price: Limit price for the order.
            tif: Time-in-force — "Gtc", "Ioc", or "Alo".

        Returns:
            Raw SDK response from :meth:`Exchange.order`.
        """
        amount = self.round_size(asset, amount)
        order_type = {"limit": {"tif": tif}}
        return await self._retry(lambda: self.exchange.order(asset, False, amount, limit_price, order_type))

    async def place_take_profit(self, asset, is_buy, amount, tp_price):
        """Create a reduce-only trigger order that executes a take-profit exit.

        Args:
            asset: Market symbol to trade.
            is_buy: ``True`` if the original position is long; dictates close
                direction.
            amount: Contract size to close.
            tp_price: Trigger price for the take-profit order.

        Returns:
            Raw SDK response from `Exchange.order`.
        """
        amount = self.round_size(asset, amount)
        order_type = {"trigger": {"triggerPx": tp_price, "isMarket": True, "tpsl": "tp"}}
        return await self._retry(lambda: self.exchange.order(asset, not is_buy, amount, tp_price, order_type, True))

    async def place_stop_loss(self, asset, is_buy, amount, sl_price):
        """Create a reduce-only trigger order that executes a stop-loss exit.

        Args:
            asset: Market symbol to trade.
            is_buy: ``True`` if the original position is long; dictates close
                direction.
            amount: Contract size to close.
            sl_price: Trigger price for the stop-loss order.

        Returns:
            Raw SDK response from `Exchange.order`.
        """
        amount = self.round_size(asset, amount)
        order_type = {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}}
        return await self._retry(lambda: self.exchange.order(asset, not is_buy, amount, sl_price, order_type, True))

    async def cancel_order(self, asset, oid):
        """Cancel a single order by identifier for a given asset.

        Args:
            asset: Market symbol associated with the order.
            oid: Hyperliquid order identifier to cancel.

        Returns:
            Raw SDK response from :meth:`Exchange.cancel`.
        """
        return await self._retry(lambda: self.exchange.cancel(asset, oid))

    async def cancel_all_orders(self, asset, max_attempts: int = 3, poll_delay: float = 0.2):
        """Cancel every open order for ``asset`` and verify by polling.

        Some order types (notably UI-set position-attached TP/SL with
        ``tpsl="tp"/"sl"``) may not appear immediately in the
        frontend_open_orders feed. This retries up to ``max_attempts`` times
        with a short delay so stale triggers don't linger after we move on.
        """
        total_cancelled = 0
        last_remaining = 0
        for attempt in range(max_attempts):
            try:
                open_orders = await self._retry(
                    lambda: self.info.frontend_open_orders(self.query_address)
                )
            except (RuntimeError, ValueError, KeyError, ConnectionError) as e:
                logging.error("cancel_all_orders fetch failed for %s: %s", asset, e)
                return {"status": "error", "message": str(e), "cancelled_count": total_cancelled}

            asset_orders = [o for o in open_orders if o.get("coin") == asset]
            if not asset_orders:
                if attempt > 0:
                    logging.info(
                        "cancel_all_orders %s: clean after %d attempt(s)",
                        asset, attempt + 1
                    )
                return {"status": "ok", "cancelled_count": total_cancelled, "attempts": attempt + 1}

            last_remaining = len(asset_orders)
            for order in asset_orders:
                oid = order.get("oid")
                if oid:
                    try:
                        await self.cancel_order(asset, oid)
                        total_cancelled += 1
                    except Exception as e:
                        logging.warning("Failed to cancel %s oid %s: %s", asset, oid, e)
            await asyncio.sleep(poll_delay)

        logging.warning(
            "cancel_all_orders %s: %d order(s) still remain after %d attempts",
            asset, last_remaining, max_attempts
        )
        return {
            "status": "partial",
            "cancelled_count": total_cancelled,
            "remaining": last_remaining,
            "attempts": max_attempts,
        }

    async def get_open_orders(self):
        """Fetch and normalize open orders associated with the wallet.

        Returns:
            List of order dictionaries augmented with ``triggerPx`` when present.
        """
        try:
            orders = await self._retry(lambda: self.info.frontend_open_orders(self.query_address))
            # Normalize trigger price if present in orderType
            for o in orders:
                try:
                    ot = o.get("orderType")
                    if isinstance(ot, dict) and "trigger" in ot:
                        trig = ot.get("trigger") or {}
                        if "triggerPx" in trig:
                            o["triggerPx"] = float(trig["triggerPx"])
                except (ValueError, KeyError, TypeError):
                    continue
            return orders
        except (RuntimeError, ValueError, KeyError, ConnectionError) as e:
            logging.error("Get open orders error: %s", e)
            return []

    async def get_recent_fills(self, limit: int = 50):
        """Return the most recent fills when supported by the SDK variant.

        Args:
            limit: Maximum number of fills to return.

        Returns:
            List of fill dictionaries or an empty list if unsupported.
        """
        try:
            # Some SDK versions expose user_fills; fall back gracefully if absent
            if hasattr(self.info, 'user_fills'):
                fills = await self._retry(lambda: self.info.user_fills(self.query_address))
            elif hasattr(self.info, 'fills'):
                fills = await self._retry(lambda: self.info.fills(self.query_address))
            else:
                return []
            if isinstance(fills, list):
                return fills[-limit:]
            return []
        except (RuntimeError, ValueError, KeyError, ConnectionError, AttributeError) as e:
            logging.error("Get recent fills error: %s", e)
            return []

    def extract_oids(self, order_result):
        """Extract resting or filled order identifiers from an exchange response.

        Args:
            order_result: Raw order response payload returned by the exchange.

        Returns:
            List of order identifiers present in resting or filled status entries.
        """
        oids = []
        try:
            statuses = order_result["response"]["data"]["statuses"]
            for st in statuses:
                if "resting" in st and "oid" in st["resting"]:
                    oids.append(st["resting"]["oid"])
                if "filled" in st and "oid" in st["filled"]:
                    oids.append(st["filled"]["oid"])
        except (KeyError, TypeError, ValueError):
            pass
        return oids

    async def get_user_state(self):
        """Retrieve wallet state with enriched position PnL calculations.

        Supports both standard and unified accounts.

        Returns:
            Dictionary with keys:
                balance — withdrawable USDC (perp) or available spot USDC.
                total_value — account value including unrealized PnL.
                positions — enriched position dicts with pnl and notional.
                effective_collateral — C4: total USDC backing positions
                    (spot_total for unified accounts; this is what should
                    drive leverage calculations).
                perp_notional — C4: sum of |size| * mark_price across
                    active perp positions. Used with effective_collateral
                    to compute true account leverage.
        """
        state = await self._retry(lambda: self.info.user_state(self.query_address))
        positions = list(state.get("assetPositions", []))
        total_value = float(state.get("accountValue", 0.0))

        # HIP-3: info.user_state only returns MAIN perp dex positions. Positions
        # held on HIP-3 builder-deployed perp dexes (e.g. "xyz") require an
        # explicit dex= parameter on clearinghouseState. Without this, the bot
        # is blind to HIP-3 positions — C1, force-close, leverage math, and
        # reconcile all silently miss them, which is how the WTIOIL stack of
        # 6+ contracts built up on a $267 account.
        for dex_name in list(self._hip3_meta_cache.keys()):
            try:
                hip3_state = await self._retry(
                    lambda d=dex_name: self.info.post("/info", {
                        "type": "clearinghouseState",
                        "user": self.query_address,
                        "dex": d,
                    })
                )
                hip3_positions = hip3_state.get("assetPositions", []) if isinstance(hip3_state, dict) else []
                if hip3_positions:
                    positions.extend(hip3_positions)
                # Add HIP-3 account value to total (each dex has its own margin pool)
                hip3_value = float(hip3_state.get("marginSummary", {}).get("accountValue", 0.0) or 0.0) if isinstance(hip3_state, dict) else 0.0
                if hip3_value:
                    total_value += hip3_value
            except Exception as e:
                logging.warning("Failed to fetch HIP-3 %s clearinghouseState: %s", dex_name, e)

        enriched_positions = []
        perp_notional = 0.0
        for pos_wrap in positions:
            pos = pos_wrap["position"]
            entry_px = float(pos.get("entryPx", 0) or 0)
            size = float(pos.get("szi", 0) or 0)
            side = "long" if size > 0 else "short"
            current_px = await self.get_current_price(pos["coin"]) if entry_px and size else 0.0
            pnl = (current_px - entry_px) * abs(size) if side == "long" else (entry_px - current_px) * abs(size)
            pos["pnl"] = pnl
            pos["notional_entry"] = abs(size) * entry_px
            pos["notional_mark"] = abs(size) * (current_px or entry_px)
            perp_notional += pos["notional_mark"]
            enriched_positions.append(pos)
        balance = float(state.get("withdrawable", 0.0))
        effective_collateral = balance

        # Unified account: funds live in spot USDC and act as cross-margin for
        # perp positions. The perp `withdrawable`/`accountValue` may be ~0 or
        # only reflect a sliver of unrealized PnL even when spot has plenty.
        # Always query spot USDC; use whichever side reports more buying power
        # (no-op for standard perp accounts where spot USDC is empty).
        try:
            spot_state = await self._retry(
                lambda: self.info.spot_user_state(self.query_address)
            )
            for bal in spot_state.get("balances", []):
                if bal.get("coin") == "USDC":
                    spot_total = float(bal.get("total", 0) or 0)
                    spot_hold = float(bal.get("hold", 0) or 0)
                    spot_available = spot_total - spot_hold
                    if spot_available > balance:
                        balance = spot_available
                    # C4: effective collateral = full spot USDC (the hold
                    # portion is margin posted to back current positions,
                    # but it still counts as collateral for those positions).
                    if spot_total > effective_collateral:
                        effective_collateral = spot_total
                    # Total value = all spot USDC (including held margin)
                    # plus current unrealized PnL.
                    perp_pnl = sum(p.get("pnl", 0.0) for p in enriched_positions)
                    spot_total_value = spot_total + perp_pnl
                    if spot_total_value > total_value:
                        total_value = spot_total_value
                    break
        except Exception as e:
            logging.warning("Failed to fetch spot state for unified account: %s", e)

        if not total_value:
            total_value = balance + sum(max(p.get("pnl", 0.0), 0.0) for p in enriched_positions)
        return {
            "balance": balance,
            "total_value": total_value,
            "positions": enriched_positions,
            "effective_collateral": effective_collateral,
            "perp_notional": perp_notional,
        }

    async def get_current_price(self, asset):
        """Return the latest mid-price for ``asset``.

        Supports both main dex assets (e.g. "BTC") and HIP-3 assets
        (e.g. "xyz:GOLD"). For HIP-3 assets, queries the dex-specific
        allMids endpoint.

        Args:
            asset: Market symbol to query.

        Returns:
            Mid-price as a float, or ``0.0`` when unavailable.
        """
        if ":" in asset:
            # HIP-3 asset — need dex-specific allMids
            dex = asset.split(":")[0]
            mids = await self._retry(
                lambda: self.info.post("/info", {"type": "allMids", "dex": dex})
            )
        else:
            mids = await self._retry(self.info.all_mids)
        return float(mids.get(asset, 0.0))

    async def _get_dex_offset(self, dex: str) -> int:
        """Return the asset index offset for a HIP-3 perp dex.

        Args:
            dex: HIP-3 dex name (e.g. "xyz").

        Returns:
            Integer offset used by the SDK for asset index calculation.
        """
        if dex in self._perp_dex_offsets:
            return self._perp_dex_offsets[dex]
        try:
            all_dexes = await self._retry(lambda: self.info.perp_dexs())
            # all_dexes[0] is the default "" dex; HIP-3 dexes follow from index 1
            for i, perp_dex in enumerate(all_dexes[1:]):
                name = perp_dex.get("name") if isinstance(perp_dex, dict) else perp_dex
                if name == dex:
                    offset = 110000 + i * 10000
                    self._perp_dex_offsets[dex] = offset
                    return offset
        except Exception as e:
            logging.warning("Could not fetch perp_dexs for offset lookup (%s): %s", dex, e)
        self._perp_dex_offsets[dex] = 110000
        return 110000

    async def get_meta_and_ctxs(self, dex=None):
        """Return cached meta/context information, fetching once per lifecycle.

        Args:
            dex: Optional HIP-3 dex name (e.g. "xyz"). None for main dex.

        Returns:
            Cached metadata response.
        """
        if dex:
            if dex not in self._hip3_meta_cache:
                response = await self._retry(
                    lambda: self.info.post("/info", {"type": "metaAndAssetCtxs", "dex": dex})
                )
                if isinstance(response, list) and len(response) >= 2:
                    self._hip3_meta_cache[dex] = response
                    # Register HIP-3 assets in SDK clients so order placement works
                    offset = await self._get_dex_offset(dex)
                    self._apply_hip3_meta_to_clients(response[0], offset)
            return self._hip3_meta_cache.get(dex)
        if not self._meta_cache:
            response = await self._retry(self.info.meta_and_asset_ctxs)
            self._meta_cache = response
        return self._meta_cache

    async def get_open_interest(self, asset):
        """Return open interest for ``asset`` if it exists in cached metadata.

        Args:
            asset: Market symbol to query (supports HIP-3 "dex:asset" format).

        Returns:
            Rounded open interest or ``None`` if unavailable.
        """
        try:
            dex = asset.split(":")[0] if ":" in asset else None
            data = await self.get_meta_and_ctxs(dex=dex)
            if isinstance(data, list) and len(data) >= 2:
                meta, asset_ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                asset_idx = next((i for i, u in enumerate(universe) if u.get("name") == asset), None)
                if asset_idx is not None and asset_idx < len(asset_ctxs):
                    oi = asset_ctxs[asset_idx].get("openInterest")
                    return round(float(oi), 2) if oi else None
            return None
        except (RuntimeError, ValueError, KeyError, ConnectionError, TypeError) as e:
            logging.error("OI fetch error for %s: %s", asset, e)
            return None

    async def get_candles(self, asset, interval="5m", count=100):
        """Fetch historical candle data for any Hyperliquid perp market.

        Args:
            asset: Market symbol (e.g. "BTC", "ETH", "OIL", "GOLD", "SPX").
            interval: Candle interval string (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            count: Number of candles to fetch (max 5000).

        Returns:
            List of dicts with keys: t, open, high, low, close, volume.
        """
        import time as _time

        # Map interval to approximate milliseconds to compute startTime
        interval_ms_map = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
            "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000,
            "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
        }
        interval_ms = interval_ms_map.get(interval, 300_000)
        end_time = int(_time.time() * 1000)
        start_time = end_time - (count * interval_ms)

        if ":" in asset:
            # HIP-3 asset — SDK candles_snapshot can't resolve dex:asset names,
            # so use the raw post endpoint directly
            raw = await self._retry(
                lambda: self.info.post("/info", {
                    "type": "candleSnapshot",
                    "req": {"coin": asset, "interval": interval,
                            "startTime": start_time, "endTime": end_time}
                })
            )
        else:
            raw = await self._retry(
                lambda: self.info.candles_snapshot(asset, interval, start_time, end_time)
            )
        candles = []
        for c in raw:
            candles.append({
                "t": c.get("t"),
                "open": float(c.get("o", 0)),
                "high": float(c.get("h", 0)),
                "low": float(c.get("l", 0)),
                "close": float(c.get("c", 0)),
                "volume": float(c.get("v", 0)),
            })
        return candles

    async def get_funding_rate(self, asset):
        """Return the most recent funding rate for ``asset`` if available.

        Args:
            asset: Market symbol to query.

        Returns:
            Funding rate as a float or ``None`` when not present.
        """
        try:
            dex = asset.split(":")[0] if ":" in asset else None
            data = await self.get_meta_and_ctxs(dex=dex)
            if isinstance(data, list) and len(data) >= 2:
                meta, asset_ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                asset_idx = next((i for i, u in enumerate(universe) if u.get("name") == asset), None)
                if asset_idx is not None and asset_idx < len(asset_ctxs):
                    funding = asset_ctxs[asset_idx].get("funding")
                    return round(float(funding), 8) if funding else None
            return None
        except (RuntimeError, ValueError, KeyError, ConnectionError, TypeError) as e:
            logging.error("Funding fetch error for %s: %s", asset, e)
            return None
