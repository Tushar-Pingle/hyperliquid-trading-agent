"""Entry-point script that wires together the trading agent, data feeds, and API."""

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.agent.decision_maker import TradingAgent
from src.indicators.local_indicators import compute_all, last_n, latest
from src.risk_manager import RiskManager, normalize_coin
from src.trading.hyperliquid_api import HyperliquidAPI
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone
import math  # For Sharpe
import time
from dotenv import load_dotenv
import os
import json
from aiohttp import web
from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def is_hip3_frozen(now: datetime | None = None) -> bool:
    """S2: HIP-3 perp dexes (oil/gold/spx/silver) freeze their oracle over the
    traditional-markets weekend. Block new entries and auto-close existing
    positions from Fri 16:50 UTC through Sun 22:00 UTC."""
    now = now or datetime.now(timezone.utc)
    wd = now.weekday()  # Mon=0 .. Fri=4 Sat=5 Sun=6
    if wd == 4 and (now.hour > 16 or (now.hour == 16 and now.minute >= 50)):
        return True
    if wd == 5:
        return True
    if wd == 6 and now.hour < 22:
        return True
    return False


def get_interval_seconds(interval_str):
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Hyperliquid")
    parser.add_argument("--assets", type=str, nargs="+", required=False, help="Assets to trade, e.g., BTC ETH")
    parser.add_argument("--interval", type=str, required=False, help="Interval period, e.g., 1h")
    args = parser.parse_args()

    # Allow assets/interval via .env (CONFIG) if CLI not provided
    from src.config_loader import CONFIG
    assets_env = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")
    if (not args.assets or len(args.assets) == 0) and assets_env:
        # Support space or comma separated
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    hyperliquid = HyperliquidAPI()
    agent = TradingAgent(hyperliquid=hyperliquid)
    risk_mgr = RiskManager()

    # P1.2 — tell risk_mgr the bar duration so cooldown_bars × interval = real seconds
    interval_sec = get_interval_seconds(args.interval)
    risk_mgr.set_interval(interval_sec)
    risk_mgr.load_cooldowns()

    # P1.3 — warn when loop interval is shorter than recommended (cuts fee drag)
    if interval_sec < 900:  # 15 m
        logging.warning(
            "P1.3: INTERVAL=%s (%ds) is below the recommended 15m minimum. "
            "Shorter intervals increase fee drag — consider INTERVAL=15m.",
            args.interval, interval_sec,
        )


    start_time = datetime.now(timezone.utc)
    invocation_count = 0
    trade_log = []  # For Sharpe: list of returns
    ACTIVE_TRADES_PATH = "active_trades.json"
    active_trades = []  # persisted to ACTIVE_TRADES_PATH — loaded/reconciled on startup
    recent_events = deque(maxlen=200)
    diary_path = "diary.jsonl"
    initial_account_value = None
    # Perp mid-price history sampled each loop (authoritative, avoids spot/perp basis mismatch)
    price_history = {}

    print(f"Starting trading agent for assets: {args.assets} at interval: {args.interval}")

    def add_event(msg: str):
        """Log an informational event and push it into the recent events deque."""
        logging.info(msg)

    def save_active_trades():
        """Persist active_trades list to disk (H5)."""
        try:
            with open(ACTIVE_TRADES_PATH, "w") as f:
                json.dump(active_trades, f, default=str)
        except Exception as e:
            logging.warning("Failed to save active_trades: %s", e)

    async def run_loop():
        """Main trading loop that gathers data, calls the agent, and executes trades."""
        nonlocal invocation_count, initial_account_value

        # Pre-load meta cache for correct order sizing
        await hyperliquid.get_meta_and_ctxs()
        # Pre-load HIP-3 dex meta for any dex:asset in the asset list
        hip3_dexes = set()
        for a in args.assets:
            if ":" in a:
                hip3_dexes.add(a.split(":")[0])
        for dex in hip3_dexes:
            await hyperliquid.get_meta_and_ctxs(dex=dex)
            add_event(f"Loaded HIP-3 meta for dex: {dex}")

        # H5: Load persisted active_trades and reconcile against live exchange state
        try:
            with open(ACTIVE_TRADES_PATH, "r") as f:
                loaded_trades = json.load(f)
            if loaded_trades:
                recon_state = await hyperliquid.get_user_state()
                recon_orders = await hyperliquid.get_open_orders()
                assets_with_pos = {
                    p.get('coin') for p in recon_state.get('positions', [])
                    if abs(float(p.get('szi') or 0)) > 0
                }
                assets_with_orders = {o.get('coin') for o in recon_orders if o.get('coin')}
                all_live_oids = {o.get('oid') for o in recon_orders if o.get('oid')}
                for tr in loaded_trades:
                    asset_name = tr.get('asset')
                    tp_oid_tr = tr.get('tp_oid')
                    sl_oid_tr = tr.get('sl_oid')
                    has_pos = asset_name in assets_with_pos
                    has_orders = asset_name in assets_with_orders
                    has_oid = (tp_oid_tr and tp_oid_tr in all_live_oids) or (sl_oid_tr and sl_oid_tr in all_live_oids)
                    if has_pos or has_orders or has_oid:
                        active_trades.append(tr)
                        add_event(f"H5: Restored active trade for {asset_name} from disk")
                    else:
                        add_event(f"H5: Dropped stale active trade for {asset_name} (no position/orders on exchange)")
                if active_trades:
                    save_active_trades()
        except FileNotFoundError:
            add_event("H5: No active_trades.json found — starting fresh")
        except Exception as e:
            add_event(f"H5: Error loading active_trades: {e}")

        # P1.2 — seed cooldowns for any positions already held on the exchange
        # so the bot cannot immediately stack or flip on the first cycle after
        # a restart without serving a full cooldown window first.
        try:
            seed_state = await hyperliquid.get_user_state()
            for _sp in seed_state.get("positions", []):
                _coin = _sp.get("coin") or ""
                try:
                    _szi = float(_sp.get("szi") or 0)
                except (TypeError, ValueError):
                    _szi = 0.0
                if _szi != 0:
                    risk_mgr.seed_cooldown(_coin)
        except Exception as _seed_err:
            add_event(f"P1.2: cooldown seed error (non-fatal): {_seed_err}")

        while True:
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # Global account state
            state = await hyperliquid.get_user_state()
            total_value = state.get('total_value') or state['balance'] + sum(p.get('pnl', 0) for p in state['positions'])
            sharpe = calculate_sharpe(trade_log)

            account_value = total_value
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = ((account_value - initial_account_value) / initial_account_value * 100.0) if initial_account_value else 0.0

            positions = []
            for pos_wrap in state['positions']:
                pos = pos_wrap
                coin = pos.get('coin')
                current_px = await hyperliquid.get_current_price(coin) if coin else None
                positions.append({
                    "symbol": coin,
                    "quantity": round_or_none(pos.get('szi'), 6),
                    "entry_price": round_or_none(pos.get('entryPx'), 2),
                    "current_price": round_or_none(current_px, 2),
                    "liquidation_price": round_or_none(pos.get('liquidationPx') or pos.get('liqPx'), 2),
                    "unrealized_pnl": round_or_none(pos.get('pnl'), 4),
                    "leverage": pos.get('leverage')
                })

            # --- RISK: Force-close positions that exceed max loss ---
            try:
                positions_to_close = risk_mgr.check_losing_positions(state['positions'])
                for ptc in positions_to_close:
                    coin = ptc["coin"]
                    size = ptc["size"]
                    is_long = ptc["is_long"]
                    add_event(f"RISK FORCE-CLOSE: {coin} at {ptc['loss_pct']}% loss (PnL: ${ptc['pnl']})")
                    try:
                        if is_long:
                            await hyperliquid.place_sell_order(coin, size)
                        else:
                            await hyperliquid.place_buy_order(coin, size)
                        await hyperliquid.cancel_all_orders(coin)
                        risk_mgr.record_cooldown(coin, "force_close")  # P1.2
                        # Remove from active trades
                        for tr in active_trades[:]:
                            if tr.get('asset') == coin:
                                active_trades.remove(tr)
                        save_active_trades()  # H5
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": coin,
                                "action": "risk_force_close",
                                "loss_pct": ptc["loss_pct"],
                                "pnl": ptc["pnl"],
                            }) + "\n")
                    except Exception as fc_err:
                        add_event(f"Force-close error for {coin}: {fc_err}")
            except Exception as risk_err:
                add_event(f"Risk check error: {risk_err}")

            # --- S2: HIP-3 weekend auto-close ---
            # When the HIP-3 oracle freeze window is active (Fri 16:50 → Sun
            # 22:00 UTC), close any open HIP-3 positions immediately. New
            # entries are blocked separately in the trade-execution path.
            if is_hip3_frozen():
                for pos in state['positions']:
                    coin = pos.get('coin') or ''
                    if ':' not in coin:
                        continue
                    try:
                        size = float(pos.get('szi') or 0)
                    except (TypeError, ValueError):
                        size = 0
                    if size == 0:
                        continue
                    add_event(f"S2 WEEKEND CLOSE: {coin} (size={size}) — HIP-3 oracle frozen window")
                    try:
                        if size > 0:
                            await hyperliquid.place_sell_order(coin, abs(size))
                        else:
                            await hyperliquid.place_buy_order(coin, abs(size))
                        await hyperliquid.cancel_all_orders(coin)
                        risk_mgr.record_cooldown(coin, "weekend_close")  # P1.2
                        for tr in active_trades[:]:
                            if tr.get('asset') == coin:
                                active_trades.remove(tr)
                        save_active_trades()
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": coin,
                                "action": "hip3_weekend_close",
                                "size": abs(size),
                            }) + "\n")
                    except Exception as ex:
                        add_event(f"S2 weekend close failed for {coin}: {ex}")

            recent_diary = []
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        entry = json.loads(line)
                        recent_diary.append(entry)
            except Exception:
                pass

            open_orders_struct = []
            try:
                open_orders = await hyperliquid.get_open_orders()
                for o in open_orders[:50]:
                    open_orders_struct.append({
                        "coin": o.get('coin'),
                        "oid": o.get('oid'),
                        "is_buy": o.get('isBuy'),
                        "size": round_or_none(o.get('sz'), 6),
                        "price": round_or_none(o.get('px'), 2),
                        "trigger_price": round_or_none(o.get('triggerPx'), 2),
                        "order_type": o.get('orderType')
                    })
            except Exception:
                open_orders = []

            # Reconcile active trades — C3: require 2 consecutive cycles of
            # "no position AND no orders" before purging, to avoid false
            # closures caused by cancel-fetch race conditions.
            try:
                # Positions: only count those with notional > $0.01 (filters dust).
                assets_with_positions = set()
                for pos in state['positions']:
                    try:
                        szi = abs(float(pos.get('szi') or 0))
                        entry = float(pos.get('entryPx') or 0)
                        if szi > 0 and szi * entry > 0.01:
                            assets_with_positions.add(pos.get('coin'))
                    except Exception:
                        continue
                assets_with_orders = {o.get('coin') for o in (open_orders or []) if o.get('coin')}
                ORPHAN_CYCLES_REQUIRED = 2
                for tr in active_trades[:]:
                    asset = tr.get('asset')
                    if asset not in assets_with_positions and asset not in assets_with_orders:
                        tr['_orphan_cycles'] = tr.get('_orphan_cycles', 0) + 1
                        if tr['_orphan_cycles'] >= ORPHAN_CYCLES_REQUIRED:
                            add_event(f"Reconciling stale active trade for {asset} (orphan for {tr['_orphan_cycles']} cycles)")
                            active_trades.remove(tr)
                            save_active_trades()  # H5
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "reconcile_close",
                                    "reason": "no_position_no_orders",
                                    "orphan_cycles": tr['_orphan_cycles'],
                                    "opened_at": tr.get('opened_at')
                                }) + "\n")
                        else:
                            add_event(f"Tentative orphan for {asset} (cycle {tr['_orphan_cycles']}/{ORPHAN_CYCLES_REQUIRED}) — deferring reconcile")
                    else:
                        # Reset counter if the asset reappears on exchange
                        if tr.get('_orphan_cycles', 0) > 0:
                            tr['_orphan_cycles'] = 0
            except Exception:
                pass

            recent_fills_struct = []
            try:
                fills = await hyperliquid.get_recent_fills(limit=50)
                for f_entry in fills[-20:]:
                    try:
                        t_raw = f_entry.get('time') or f_entry.get('timestamp')
                        timestamp = None
                        if t_raw is not None:
                            try:
                                t_int = int(t_raw)
                                if t_int > 1e12:
                                    timestamp = datetime.fromtimestamp(t_int / 1000, tz=timezone.utc).isoformat()
                                else:
                                    timestamp = datetime.fromtimestamp(t_int, tz=timezone.utc).isoformat()
                            except Exception:
                                timestamp = str(t_raw)
                        recent_fills_struct.append({
                            "timestamp": timestamp,
                            "coin": f_entry.get('coin') or f_entry.get('asset'),
                            "is_buy": f_entry.get('isBuy'),
                            "size": round_or_none(f_entry.get('sz') or f_entry.get('size'), 6),
                            "price": round_or_none(f_entry.get('px') or f_entry.get('price'), 2)
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance": round_or_none(state['balance'], 2),
                "account_value": round_or_none(account_value, 2),
                "sharpe_ratio": round_or_none(sharpe, 3),
                "positions": positions,
                "active_trades": [
                    {
                        "asset": tr.get('asset'),
                        "is_long": tr.get('is_long'),
                        "amount": round_or_none(tr.get('amount'), 6),
                        "entry_price": round_or_none(tr.get('entry_price'), 2),
                        "tp_oid": tr.get('tp_oid'),
                        "sl_oid": tr.get('sl_oid'),
                        "exit_plan": tr.get('exit_plan'),
                        "entry_thesis": tr.get('entry_thesis'),  # S8
                        "opened_at": tr.get('opened_at')
                    }
                    for tr in active_trades
                ],
                "open_orders": open_orders_struct,
                "recent_diary": recent_diary,
                "recent_fills": recent_fills_struct,
            }

            # Gather data for ALL assets first (using Hyperliquid candles + local indicators)
            market_sections = []
            asset_prices = {}
            asset_atr_ratios = {}  # S4: short/long ATR ratio for vol-scaled sizing
            for asset in args.assets:
                try:
                    current_price = await hyperliquid.get_current_price(asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({"t": datetime.now(timezone.utc).isoformat(), "mid": round_or_none(current_price, 2)})
                    oi = await hyperliquid.get_open_interest(asset)
                    funding = await hyperliquid.get_funding_rate(asset)

                    # Fetch candles from Hyperliquid and compute indicators locally
                    candles_5m = await hyperliquid.get_candles(asset, "5m", 100)
                    candles_4h = await hyperliquid.get_candles(asset, "4h", 100)

                    intra = compute_all(candles_5m)
                    lt = compute_all(candles_4h)

                    # S1: volume-spike ratio (current bar volume / 20-bar SMA volume)
                    def _vol_spike(candles):
                        vols = [c.get("volume", 0) for c in candles]
                        if len(vols) < 20:
                            return None
                        sma_v = sum(vols[-20:]) / 20.0
                        if sma_v <= 0:
                            return None
                        return round(vols[-1] / sma_v, 3)
                    vol_spike_5m = _vol_spike(candles_5m)
                    vol_spike_4h = _vol_spike(candles_4h)

                    # S4: ATR ratio (short-term / long-term on 4h)
                    atr3_lt = latest(lt.get("atr3", []))
                    atr14_lt = latest(lt.get("atr14", []))
                    if atr3_lt and atr14_lt and atr14_lt > 0:
                        asset_atr_ratios[asset] = round(atr3_lt / atr14_lt, 3)

                    recent_mids = [entry["mid"] for entry in list(price_history.get(asset, []))[-10:]]
                    funding_annualized = round(funding * 24 * 365 * 100, 2) if funding else None

                    market_sections.append({
                        "asset": asset,
                        "current_price": round_or_none(current_price, 2),
                        "intraday": {
                            "ema20": round_or_none(latest(intra.get("ema20", [])), 2),
                            "macd": round_or_none(latest(intra.get("macd", [])), 2),
                            "rsi7": round_or_none(latest(intra.get("rsi7", [])), 2),
                            "rsi14": round_or_none(latest(intra.get("rsi14", [])), 2),
                            "vol_spike_ratio": vol_spike_5m,
                            "series": {
                                "ema20": round_series(last_n(intra.get("ema20", []), 10), 2),
                                "macd": round_series(last_n(intra.get("macd", []), 10), 2),
                                "rsi7": round_series(last_n(intra.get("rsi7", []), 10), 2),
                                "rsi14": round_series(last_n(intra.get("rsi14", []), 10), 2),
                            }
                        },
                        "long_term": {
                            "ema20": round_or_none(latest(lt.get("ema20", [])), 2),
                            "ema50": round_or_none(latest(lt.get("ema50", [])), 2),
                            "atr3": round_or_none(atr3_lt, 2),
                            "atr14": round_or_none(atr14_lt, 2),
                            "atr_ratio_short_over_long": asset_atr_ratios.get(asset),
                            "vol_spike_ratio": vol_spike_4h,
                            "macd_series": round_series(last_n(lt.get("macd", []), 10), 2),
                            "rsi_series": round_series(last_n(lt.get("rsi14", []), 10), 2),
                        },
                        "open_interest": round_or_none(oi, 2),
                        "funding_rate": round_or_none(funding, 8),
                        "funding_annualized_pct": funding_annualized,
                        "hip3_market_frozen": (':' in asset and is_hip3_frozen()),
                        "recent_mid_prices": recent_mids,
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # --- S3: Continuous exit-condition checking ---
            # Before asking the LLM, evaluate each active trade's exit_plan
            # against current indicators. If invalidated, market-close and
            # cancel its TP/SL — removes the LLM-talking-itself-back-in
            # failure mode.
            for tr in active_trades[:]:
                try:
                    if not tr.get('exit_plan'):
                        continue
                    asset = tr.get('asset')
                    if not asset:
                        continue
                    should_exit = await check_exit_condition(tr, hyperliquid)
                    if not should_exit:
                        continue
                    add_event(f"S3 EXIT {asset}: invalidation triggered by exit_plan — closing")
                    try:
                        amt = abs(float(tr.get('amount') or 0))
                        if amt > 0:
                            if tr.get('is_long'):
                                await hyperliquid.place_sell_order(asset, amt)
                            else:
                                await hyperliquid.place_buy_order(asset, amt)
                        await hyperliquid.cancel_all_orders(asset)
                        risk_mgr.record_cooldown(asset, "exit_invalidation")  # P1.2
                        active_trades.remove(tr)
                        save_active_trades()
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset": asset,
                                "action": "exit_invalidation",
                                "exit_plan": tr.get('exit_plan'),
                            }) + "\n")
                    except Exception as ex:
                        add_event(f"S3 exit close failed for {asset}: {ex}")
                except Exception:
                    continue

            # Single LLM call with all assets
            context_payload = OrderedDict([
                ("invocation", {
                    "minutes_since_start": round(minutes_since_start, 2),
                    "current_time": datetime.now(timezone.utc).isoformat(),
                    "invocation_count": invocation_count
                }),
                ("account", dashboard),
                ("risk_limits", risk_mgr.get_risk_summary()),
                ("market_data", market_sections),
                ("instructions", {
                    "assets": args.assets,
                    "requirement": "Decide actions for all assets and return a strict JSON object matching the schema."
                })
            ])
            context = json.dumps(context_payload, default=json_default)
            add_event(f"Combined prompt length: {len(context)} chars for {len(args.assets)} assets")
            with open("prompts.log", "a") as f:
                f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n{json.dumps(context_payload, indent=2, default=json_default)}\n")

            def _is_failed_outputs(outs):
                """Return True when outputs are missing or clearly invalid."""
                if not isinstance(outs, dict):
                    return True
                decisions = outs.get("trade_decisions")
                if not isinstance(decisions, list) or not decisions:
                    return True
                try:
                    return all(
                        isinstance(o, dict)
                        and (o.get('action') == 'hold')
                        and ('parse error' in (o.get('rationale', '').lower()))
                        for o in decisions
                    )
                except Exception:
                    return True

            try:
                outputs = agent.decide_trade(args.assets, context)
                if not isinstance(outputs, dict):
                    add_event(f"Invalid output format (expected dict): {outputs}")
                    outputs = {}
            except Exception as e:
                import traceback
                add_event(f"Agent error: {e}")
                add_event(f"Traceback: {traceback.format_exc()}")
                outputs = {}

            # Retry once on failure/parse error with a stricter instruction prefix
            if _is_failed_outputs(outputs):
                add_event("Retrying LLM once due to invalid/parse-error output")
                context_retry_payload = OrderedDict([
                    ("retry_instruction", "Return ONLY the JSON array per schema with no prose."),
                    ("original_context", context_payload)
                ])
                context_retry = json.dumps(context_retry_payload, default=json_default)
                try:
                    outputs = agent.decide_trade(args.assets, context_retry)
                    if not isinstance(outputs, dict):
                        add_event(f"Retry invalid format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Retry agent error: {e}")
                    add_event(f"Retry traceback: {traceback.format_exc()}")
                    outputs = {}

            reasoning_text = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
            if reasoning_text:
                add_event(f"LLM reasoning summary: {reasoning_text}")

            # Log full cycle decisions for the dashboard
            cycle_decisions = []
            for d in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                cycle_decisions.append({
                    "asset": d.get("asset"),
                    "action": d.get("action", "hold"),
                    "allocation_usd": d.get("allocation_usd", 0),
                    "rationale": d.get("rationale", ""),
                })
            cycle_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": invocation_count,
                "reasoning": reasoning_text[:2000] if reasoning_text else "",
                "decisions": cycle_decisions,
                "account_value": round_or_none(account_value, 2),
                "balance": round_or_none(state['balance'], 2),
                "positions_count": len([p for p in state['positions'] if abs(float(p.get('szi') or 0)) > 0]),
            }
            try:
                with open("decisions.jsonl", "a") as f:
                    f.write(json.dumps(cycle_log) + "\n")
            except Exception:
                pass

            # Execute trades for each asset
            for output in outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []:
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue
                    action = output.get("action")
                    current_price = asset_prices.get(asset, 0)
                    action = output["action"]
                    rationale = output.get("rationale", "")
                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")
                    if action in ("buy", "sell"):
                        is_buy = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue

                        # S2: Block new HIP-3 entries during weekend freeze
                        if ':' in asset and is_hip3_frozen():
                            add_event(f"S2 BLOCK {asset}: HIP-3 oracle frozen window — no new entries")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "hip3_weekend_block",
                                    "requested_action": action,
                                    "requested_alloc_usd": alloc_usd,
                                }) + "\n")
                            continue

                        # S4: Pass ATR ratio to risk manager for vol-scaled sizing
                        if asset in asset_atr_ratios:
                            output["atr_ratio"] = asset_atr_ratios[asset]

                        # S7: For HIP-3 assets, check orderbook liquidity. If
                        # spread > 0.5% or top-of-book depth < $500, cap the
                        # allocation hard so we don't pay massive slippage on
                        # thin synthetic perps.
                        if ":" in asset:
                            try:
                                ob = await hyperliquid.get_orderbook(asset)
                                if ob:
                                    spread_pct = ob.get("spread_pct") or 0
                                    side_depth = ob.get("ask_depth_usd" if is_buy else "bid_depth_usd") or 0
                                    if spread_pct > 0.5 or side_depth < 500:
                                        cap_pct = 5.0
                                        max_alloc = (state.get('total_value') or 0) * (cap_pct / 100.0)
                                        if max_alloc > 0 and alloc_usd > max_alloc:
                                            add_event(
                                                f"S7 {asset}: thin book (spread={spread_pct:.2f}%, "
                                                f"depth=${side_depth:.0f}) — capping alloc "
                                                f"${alloc_usd:.2f}→${max_alloc:.2f}"
                                            )
                                            alloc_usd = max(max_alloc, 11.0)
                                            output["allocation_usd"] = alloc_usd
                            except Exception as ob_err:
                                add_event(f"S7 orderbook check failed for {asset}: {ob_err}")

                        # --- C1 / P1.1: Pre-trade position-existence check ---
                        # Hard block on same-direction stacking — no scale-in
                        # carve-out.  Opposite-direction → flip (H6).
                        # risk_mgr.validate_trade() also enforces check_stacking()
                        # as a second line of defence (catches normalised aliases).
                        existing_szi = 0.0
                        existing_entry = 0.0
                        for pos in state.get('positions', []):
                            if normalize_coin(pos.get('coin') or '') == normalize_coin(asset):
                                try:
                                    existing_szi = float(pos.get('szi') or 0)
                                    existing_entry = float(pos.get('entryPx') or 0)
                                except (TypeError, ValueError):
                                    existing_szi = 0.0
                                break
                        if existing_szi != 0:
                            existing_is_long = existing_szi > 0
                            existing_notional = abs(existing_szi) * (existing_entry or current_price)
                            if existing_is_long == is_buy:
                                # P1.1: Hard block — same-direction position exists.
                                # No scale-in permitted (this was the WTIOIL stacking bug).
                                add_event(
                                    f"SKIP {asset}: stacking_blocked — "
                                    f"existing {('long' if existing_is_long else 'short')} "
                                    f"szi={existing_szi:.6f} notional=${existing_notional:.2f}"
                                )
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "stacking_blocked",
                                        "existing_szi": existing_szi,
                                        "existing_notional": round(existing_notional, 2),
                                        "requested_alloc_usd": alloc_usd,
                                    }) + "\n")
                                continue
                            else:
                                # H6: Opposite direction — flip: close existing, then open new
                                old_side = 'long' if existing_is_long else 'short'
                                new_side = 'long' if is_buy else 'short'
                                add_event(f"FLIP {asset}: closing {old_side} (notional ${existing_notional:.2f}) to open {new_side}")
                                flip_ok = False
                                try:
                                    close_size = abs(existing_szi)
                                    if existing_is_long:
                                        await hyperliquid.place_sell_order(asset, close_size)
                                    else:
                                        await hyperliquid.place_buy_order(asset, close_size)
                                    await hyperliquid.cancel_all_orders(asset)
                                    # Poll to confirm position is closed (up to 5 × 0.5s)
                                    for _ in range(5):
                                        await asyncio.sleep(0.5)
                                        check_state = await hyperliquid.get_user_state()
                                        still_open = any(
                                            p.get('coin') == asset and abs(float(p.get('szi') or 0)) > 0
                                            for p in check_state.get('positions', [])
                                        )
                                        if not still_open:
                                            flip_ok = True
                                            break
                                    if not flip_ok:
                                        add_event(f"FLIP {asset}: position still open after close — skipping new entry this cycle")
                                        with open(diary_path, "a") as f:
                                            f.write(json.dumps({
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                                "asset": asset,
                                                "action": "flip_close_timeout",
                                                "old_side": old_side,
                                                "new_side": new_side,
                                            }) + "\n")
                                        continue
                                    # Remove stale active_trade entry
                                    for tr in active_trades[:]:
                                        if tr.get('asset') == asset:
                                            active_trades.remove(tr)
                                    save_active_trades()  # H5
                                    risk_mgr.record_cooldown(asset, "flip")  # P1.2
                                    with open(diary_path, "a") as f:
                                        f.write(json.dumps({
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "asset": asset,
                                            "action": "flip_closed",
                                            "old_side": old_side,
                                            "new_side": new_side,
                                            "closed_size": close_size,
                                        }) + "\n")
                                    add_event(f"FLIP {asset}: {old_side} closed — proceeding to open {new_side}")
                                    # Fall through: proceed to open the new direction below
                                except Exception as flip_err:
                                    add_event(f"FLIP {asset}: close failed ({flip_err}) — skipping new entry")
                                    continue
                                if not flip_ok:
                                    continue

                        # --- RISK: Validate trade before execution ---
                        output["current_price"] = current_price
                        allowed, reason, output = risk_mgr.validate_trade(
                            output, state, initial_account_value or 0
                        )
                        if not allowed:
                            add_event(f"RISK BLOCKED {asset}: {reason}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "risk_blocked",
                                    "reason": reason,
                                    "original_alloc_usd": alloc_usd,
                                }) + "\n")
                            continue
                        # Use potentially adjusted values from risk manager
                        alloc_usd = float(output.get("allocation_usd", alloc_usd))
                        amount = alloc_usd / current_price

                        # Cancel any stale TP/SL orders for this asset before
                        # opening a new position (guards against duplicates after
                        # service restarts that reset in-memory active_trades).
                        await hyperliquid.cancel_all_orders(asset)

                        # --- P1.3: Entry order — post-only limit (default) or market ---
                        entry_order_type_cfg = (
                            CONFIG.get("entry_order_type") or "limit"
                        ).lower()
                        entry_limit_timeout = int(
                            CONFIG.get("entry_limit_timeout_sec") or 90
                        )
                        actual_size = amount
                        filled = False
                        order = None
                        order_type = "limit" if entry_order_type_cfg != "market" else "market"
                        limit_price = None

                        if entry_order_type_cfg != "market":
                            # Post-only limit entry: join best bid (buy) or best ask (sell).
                            # If unfilled after timeout → cancel and skip.  No market fallback.
                            try:
                                ob = await hyperliquid.get_orderbook(asset)
                            except Exception as ob_err:
                                ob = None
                                add_event(f"P1.3 orderbook error {asset}: {ob_err}")
                            if not ob:
                                add_event(
                                    f"P1.3 SKIP {asset}: orderbook unavailable — "
                                    "cannot place post-only limit entry"
                                )
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "limit_entry_no_orderbook",
                                    }) + "\n")
                                continue

                            limit_price = ob["best_bid"] if is_buy else ob["best_ask"]
                            if is_buy:
                                order = await hyperliquid.place_limit_buy(
                                    asset, amount, limit_price, tif="Alo"
                                )
                            else:
                                order = await hyperliquid.place_limit_sell(
                                    asset, amount, limit_price, tif="Alo"
                                )

                            entry_oids = hyperliquid.extract_oids(order)
                            entry_oid = entry_oids[0] if entry_oids else None

                            if not entry_oid:
                                add_event(
                                    f"P1.3 SKIP {asset}: post-only limit rejected "
                                    f"(no oid — order likely crossed spread at {limit_price})"
                                )
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "limit_entry_rejected",
                                        "limit_price": limit_price,
                                    }) + "\n")
                                continue

                            add_event(
                                f"P1.3: LIMIT {action.upper()} {asset} "
                                f"{amount:.6f} @ {limit_price} post-only "
                                f"(oid={entry_oid}, timeout={entry_limit_timeout}s)"
                            )

                            # Poll until filled or timeout — check every 5 s.
                            #
                            # Design notes:
                            # - poll_start is tracked separately so the SKIP log
                            #   prints the real elapsed time, not a hardcoded value.
                            # - _seen_in_feed guards against a false-early-exit caused
                            #   by feed latency: if the order has never appeared in
                            #   frontend_open_orders yet, we give it a 10 s grace
                            #   period before treating "not visible" as "rejected".
                            # - Partial fills: if the order disappears but a position
                            #   of any size exists, treat it as filled (actual_size
                            #   reflects what was actually executed).
                            poll_start = time.monotonic()
                            deadline = poll_start + entry_limit_timeout
                            _seen_in_feed = False

                            while time.monotonic() < deadline:
                                await asyncio.sleep(5)
                                elapsed = time.monotonic() - poll_start
                                try:
                                    cur_orders = await hyperliquid.get_open_orders()
                                    still_open = any(
                                        o.get("oid") == entry_oid for o in cur_orders
                                    )

                                    if still_open:
                                        _seen_in_feed = True
                                        continue  # order resting — keep waiting

                                    # Order not visible in feed.
                                    # Check for position (full or partial fill).
                                    pos_check = await hyperliquid.get_user_state()
                                    for _pp in pos_check.get("positions", []):
                                        if (
                                            normalize_coin(_pp.get("coin") or "")
                                            == normalize_coin(asset)
                                        ):
                                            _sz = abs(float(_pp.get("szi") or 0))
                                            if _sz > 0:
                                                actual_size = _sz  # partial fill ok
                                                filled = True
                                            break

                                    if filled:
                                        break  # position confirmed — done

                                    # No position and order not in feed.
                                    # Exit early only once we're past the grace period
                                    # (avoids false-skip due to propagation delay).
                                    if _seen_in_feed or elapsed >= 10:
                                        break  # order definitely gone, no fill

                                    # else: feed lag grace period — keep polling

                                except Exception as _poll_err:
                                    add_event(f"P1.3 poll error {asset}: {_poll_err}")
                                    # Keep polling on errors until timeout

                            elapsed_sec = time.monotonic() - poll_start
                            if not filled:
                                # Order unfilled — cancel any resting portion and skip.
                                # No market fallback.
                                try:
                                    await hyperliquid.cancel_order(asset, entry_oid)
                                except Exception as _ce:
                                    add_event(f"P1.3 cancel error {asset}: {_ce}")
                                add_event(
                                    f"P1.3 SKIP {asset}: limit entry unfilled after "
                                    f"{elapsed_sec:.0f}s — cancelled, no market fallback"
                                )
                                with open(diary_path, "a") as f:
                                    f.write(json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "asset": asset,
                                        "action": "limit_entry_timeout",
                                        "limit_price": limit_price,
                                        "elapsed_sec": round(elapsed_sec, 1),
                                        "timeout_sec": entry_limit_timeout,
                                    }) + "\n")
                                continue

                        else:
                            # Market entry path (ENTRY_ORDER_TYPE=market)
                            order = (
                                await hyperliquid.place_buy_order(asset, amount)
                                if is_buy
                                else await hyperliquid.place_sell_order(asset, amount)
                            )
                            order_type = "market"

                            # H7/M9: Poll actual filled position size
                            max_polls = 5 if ":" in asset else 3
                            poll_delay = 0.5 if ":" in asset else 0.35
                            for _poll in range(max_polls):
                                await asyncio.sleep(poll_delay)
                                try:
                                    pos_state = await hyperliquid.get_user_state()
                                    for _p in pos_state.get("positions", []):
                                        if _p.get("coin") == asset:
                                            _szi = abs(float(_p.get("szi") or 0))
                                            if _szi > 0:
                                                actual_size = _szi
                                                filled = True
                                                break
                                except Exception:
                                    pass
                                if filled:
                                    break
                            if not filled:
                                fills_check = await hyperliquid.get_recent_fills(limit=10)
                                for fc in reversed(fills_check):
                                    try:
                                        if (
                                            fc.get("coin") == asset
                                            or fc.get("asset") == asset
                                        ):
                                            filled = True
                                            break
                                    except Exception:
                                        continue

                        add_event(
                            f"{action.upper()} {asset} amount {actual_size:.6f} "
                            f"at ~{current_price} (filled={filled}, "
                            f"entry={order_type}{'@'+str(limit_price) if limit_price else ''})"
                        )

                        trade_log.append({"type": action, "price": current_price, "amount": actual_size, "exit_plan": output["exit_plan"], "filled": filled})

                        # H8: Place TP and SL — only register in active_trades when both succeed
                        tp_oid = None
                        sl_oid = None
                        orders_ok = True
                        try:
                            if output.get("tp_price"):
                                tp_order = await hyperliquid.place_take_profit(asset, is_buy, actual_size, output["tp_price"])
                                tp_oids = hyperliquid.extract_oids(tp_order)
                                tp_oid = tp_oids[0] if tp_oids else None
                                if tp_oid is None:
                                    add_event(f"WARNING: TP for {asset} returned no oid — response: {tp_order}")
                                    orders_ok = False
                                else:
                                    add_event(f"TP placed {asset} at {output['tp_price']} (oid={tp_oid})")
                            if output.get("sl_price"):
                                sl_order = await hyperliquid.place_stop_loss(asset, is_buy, actual_size, output["sl_price"])
                                sl_oids = hyperliquid.extract_oids(sl_order)
                                sl_oid = sl_oids[0] if sl_oids else None
                                if sl_oid is None:
                                    add_event(f"WARNING: SL for {asset} returned no oid — response: {sl_order}")
                                    orders_ok = False
                                else:
                                    add_event(f"SL placed {asset} at {output['sl_price']} (oid={sl_oid})")
                        except Exception as tpsl_err:
                            add_event(f"TP/SL placement error for {asset}: {tpsl_err}")
                            orders_ok = False

                        if not orders_ok:
                            # H8: Cancel any partial orders and do NOT register the trade
                            add_event(f"H8: TP/SL incomplete for {asset} — cancelling all orders, trade not registered")
                            await hyperliquid.cancel_all_orders(asset)
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": "tpsl_failed",
                                    "entry_action": action,
                                    "tp_oid": tp_oid,
                                    "sl_oid": sl_oid,
                                }) + "\n")
                        else:
                            # All confirmed — register in active_trades and persist
                            for existing in active_trades[:]:
                                if existing.get('asset') == asset:
                                    try:
                                        active_trades.remove(existing)
                                    except ValueError:
                                        pass
                            active_trades.append({
                                "asset": asset,
                                "is_long": is_buy,
                                "amount": actual_size,
                                "entry_price": current_price,
                                "tp_oid": tp_oid,
                                "sl_oid": sl_oid,
                                "exit_plan": output["exit_plan"],
                                "entry_thesis": output.get("rationale", "") or "",  # S8
                                "opened_at": datetime.now(timezone.utc).isoformat()
                            })
                            save_active_trades()  # H5
                            risk_mgr.record_cooldown(asset, action)  # P1.2
                            if rationale:
                                add_event(f"Post-trade rationale for {asset}: {rationale}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset": asset,
                                    "action": action,
                                    "order_type": order_type,
                                    "limit_price": limit_price,
                                    "allocation_usd": alloc_usd,
                                    "amount": actual_size,
                                    "entry_price": current_price,
                                    "tp_price": output.get("tp_price"),
                                    "tp_oid": tp_oid,
                                    "sl_price": output.get("sl_price"),
                                    "sl_oid": sl_oid,
                                    "exit_plan": output.get("exit_plan", ""),
                                    "rationale": output.get("rationale", ""),
                                    "order_result": str(order),
                                    "opened_at": datetime.now(timezone.utc).isoformat(),
                                    "filled": filled
                                }) + "\n")
                    else:
                        add_event(f"Hold {asset}: {output.get('rationale', '')}")
                        # Write hold to diary
                        with open(diary_path, "a") as f:
                            diary_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "asset": asset,
                                "action": "hold",
                                "rationale": output.get("rationale", "")
                            }
                            f.write(json.dumps(diary_entry) + "\n")
                except Exception as e:
                    import traceback
                    add_event(f"Execution error {asset}: {e}")
                    add_event(f"Traceback: {traceback.format_exc()}")

            await asyncio.sleep(get_interval_seconds(args.interval))

    async def handle_diary(request):
        """Return diary entries as JSON or newline-delimited text."""
        try:
            raw = request.query.get('raw')
            download = request.query.get('download')
            if raw or download:
                if not os.path.exists(diary_path):
                    return web.Response(text="", content_type="text/plain")
                with open(diary_path, "r") as f:
                    data = f.read()
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename=diary.jsonl"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(request.query.get('limit', '200'))
            with open(diary_path, "r") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            entries = [json.loads(l) for l in lines[start:]]
            return web.json_response({"entries": entries})
        except FileNotFoundError:
            return web.json_response({"entries": []})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path = request.query.get('path', 'llm_requests.log')
            download = request.query.get('download')
            limit_param = request.query.get('limit')
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_param and (limit_param.lower() == 'all' or limit_param == '-1')):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_param) if limit_param else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def start_api(app):
        """Register HTTP endpoints for observing diary entries and logs."""
        app.router.add_get('/diary', handle_diary)
        app.router.add_get('/logs', handle_logs)

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        app = web.Application()
        await start_api(app)
        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        await run_loop()

    def calculate_total_return(state, trade_log):
        """Compute percent return relative to an assumed initial balance."""
        initial = 10000
        current = state['balance'] + sum(p.get('pnl', 0) for p in state.get('positions', []))
        return ((current - initial) / initial) * 100 if initial else 0

    def calculate_sharpe(returns):
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get('pnl', 0) if 'pnl' in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    async def check_exit_condition(trade, hyperliquid_api):
        """Evaluate whether a given trade's exit plan triggers a close."""
        plan = (trade.get("exit_plan") or "").lower()
        if not plan:
            return False
        try:
            candles_4h = await hyperliquid_api.get_candles(trade["asset"], "4h", 60)
            indicators = compute_all(candles_4h)
            if "macd" in plan and "below" in plan:
                macd_val = latest(indicators.get("macd", []))
                threshold = float(plan.split("below")[-1].strip())
                return macd_val is not None and macd_val < threshold
            if "close above ema50" in plan:
                ema50_val = latest(indicators.get("ema50", []))
                current = await hyperliquid_api.get_current_price(trade["asset"])
                return ema50_val is not None and current > ema50_val
        except Exception:
            return False
        return False

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
