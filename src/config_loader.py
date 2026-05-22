"""Centralized environment variable loading for the trading agent configuration."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Fetch an environment variable with optional default and required validation."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw}") from exc


def _get_json(name: str, default: dict | None = None) -> dict | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Environment variable {name} must be a JSON object")
        return parsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON for {name}: {raw}") from exc


def _get_list(name: str, default: list[str] | None = None) -> list[str] | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    # Support JSON-style lists
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise RuntimeError(f"Environment variable {name} must be a list if using JSON syntax")
            return [str(item).strip().strip('"\'') for item in parsed if str(item).strip()]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON list for {name}: {raw}") from exc
    # Fallback: comma separated string
    values = []
    for item in raw.split(","):
        cleaned = item.strip().strip('"\'')
        if cleaned:
            values.append(cleaned)
    return values or default


CONFIG = {
    # Hyperliquid
    "hyperliquid_private_key": _get_env("HYPERLIQUID_PRIVATE_KEY") or _get_env("LIGHTER_PRIVATE_KEY"),
    "mnemonic": _get_env("MNEMONIC"),
    "hyperliquid_base_url": _get_env("HYPERLIQUID_BASE_URL"),
    "hyperliquid_network": _get_env("HYPERLIQUID_NETWORK", "mainnet"),
    "hyperliquid_vault_address": _get_env("HYPERLIQUID_VAULT_ADDRESS"),  # Main wallet address (agent signs on behalf)

    # LLM — Anthropic Claude API (primary)
    "anthropic_api_key": _get_env("ANTHROPIC_API_KEY", required=True),
    "llm_model": _get_env("LLM_MODEL", "claude-sonnet-4-20250514"),
    "sanitize_model": _get_env("SANITIZE_MODEL", "claude-haiku-4-5-20251001"),
    "max_tokens": _get_int("MAX_TOKENS", 4096),
    "enable_tool_calling": _get_bool("ENABLE_TOOL_CALLING", False),

    # Extended thinking (Claude)
    "thinking_enabled": _get_bool("THINKING_ENABLED", False),
    "thinking_budget_tokens": _get_int("THINKING_BUDGET_TOKENS", 10000),

    # Runtime controls
    "assets": _get_env("ASSETS"),  # e.g., "BTC ETH SOL OIL GOLD SPX"
    "interval": _get_env("INTERVAL"),  # e.g., "5m", "1h"

    # Risk management
    "max_position_pct": _get_env("MAX_POSITION_PCT", "20"),
    "max_loss_per_position_pct": _get_env("MAX_LOSS_PER_POSITION_PCT", "20"),
    "max_leverage": _get_env("MAX_LEVERAGE", "10"),
    "max_total_exposure_pct": _get_env("MAX_TOTAL_EXPOSURE_PCT", "80"),
    "daily_loss_circuit_breaker_pct": _get_env("DAILY_LOSS_CIRCUIT_BREAKER_PCT", "25"),
    "mandatory_sl_pct": _get_env("MANDATORY_SL_PCT", "5"),
    "max_concurrent_positions": _get_env("MAX_CONCURRENT_POSITIONS", "10"),
    "min_balance_reserve_pct": _get_env("MIN_BALANCE_RESERVE_PCT", "10"),

    # API server
    "api_host": _get_env("API_HOST", "0.0.0.0"),
    "api_port": _get_env("APP_PORT") or _get_env("API_PORT") or "3000",

    # P1.2 — per-asset cooldown
    "cooldown_bars": _get_env("COOLDOWN_BARS", "3"),           # bars of silence after any open/close/flip

    # P1.3 — fee reduction / limit entries
    "entry_order_type": _get_env("ENTRY_ORDER_TYPE", "limit"), # "limit" (post-only) or "market"
    "entry_limit_timeout_sec": _get_env("ENTRY_LIMIT_TIMEOUT_SEC", "90"),  # cancel unfilled limit after N s

    # P1.1 — stacking guard escape hatch (leave false unless you deliberately want scale-in)
    "stacking_allow_scale_in": _get_bool("STACKING_ALLOW_SCALE_IN", False),

    # P2.1 — ATR-scaled position sizing
    # atr_ratio = short-term ATR / long-term ATR (atr3 / atr14 on 4h)
    "atr_ratio_high": _get_env("ATR_RATIO_HIGH", "1.5"),     # above this: high vol regime
    "atr_ratio_low": _get_env("ATR_RATIO_LOW", "0.7"),       # below this: low vol regime
    "atr_low_size_mult": _get_env("LOW_SIZE_MULT", "0.5"),   # high vol → shrink allocation
    "atr_high_size_mult": _get_env("HIGH_SIZE_MULT", "1.0"), # low vol → normal (no boost above cap)

    # P2.5 — trade log + Sharpe
    "min_sharpe_sample": _get_env("MIN_SHARPE_SAMPLE", "10"),
    "sharpe_window": _get_env("SHARPE_WINDOW", "50"),

    # P2.6 — minimum reward:risk on entries
    "min_rr": _get_env("MIN_RR", "1.5"),

    # P2.7 — low-conviction volume gate
    "min_vol_spike_ratio": _get_env("MIN_VOL_SPIKE_RATIO", "0.5"),

    # P2.5.4 — concise mode: shorter LLM output on hold-all cycles
    "concise_mode": _get_bool("CONCISE_MODE", True),

    # P3.1 — trailing stop loss
    "trailing_stop_enabled": _get_bool("TRAILING_STOP_ENABLED", True),
    "trail_activate_r": _get_env("TRAIL_ACTIVATE_R", "1.0"),
    "trail_distance_atr": _get_env("TRAIL_DISTANCE_ATR", "1.0"),

    # Legacy / optional
    "taapi_api_key": _get_env("TAAPI_API_KEY"),
    "openrouter_api_key": _get_env("OPENROUTER_API_KEY"),
}
