"""
配置管理模块
从 .env 文件和环境变量加载所有配置项。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

# 优先加载项目根目录 .env
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


@dataclass(frozen=True)
class LongportCredentials:
    app_key: str
    app_secret: str
    access_token: str
    region: str  # "cn" | "hk"


@dataclass(frozen=True)
class StrategyParams:
    boll_period: int
    boll_std_dev: float
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    trend_ema_period: int


@dataclass(frozen=True)
class RiskParams:
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    max_drawdown_pct: Decimal
    trade_cooldown_seconds: int
    max_position_value: Decimal
    max_total_exposure: Decimal
    max_positions: int


@dataclass(frozen=True)
class WatchlistParams:
    use_watchlist: bool
    market_filter: str
    refresh_interval: int

@dataclass(frozen=True)
class TradingConfig:
    paper_trading: bool
    watch_symbols: list[str]
    order_quantity: int
    kline_poll_interval: int
    credentials: LongportCredentials
    strategy: StrategyParams
    risk: RiskParams
    watchlist: WatchlistParams


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val


def load_config() -> TradingConfig:
    """从环境变量构建完整的交易配置。"""
    credentials = LongportCredentials(
        app_key=_env("LONGPORT_APP_KEY"),
        app_secret=_env("LONGPORT_APP_SECRET"),
        access_token=_env("LONGPORT_ACCESS_TOKEN"),
        region=_env("LONGPORT_REGION", "hk"),
    )

    strategy = StrategyParams(
        boll_period=int(_env("BOLL_PERIOD", "20")),
        boll_std_dev=float(_env("BOLL_STD_DEV", "2.0")),
        rsi_period=int(_env("RSI_PERIOD", "14")),
        rsi_oversold=float(_env("RSI_OVERSOLD", "30")),
        rsi_overbought=float(_env("RSI_OVERBOUGHT", "70")),
        trend_ema_period=int(_env("TREND_EMA_PERIOD", "60")),
    )

    risk = RiskParams(
        stop_loss_pct=Decimal(_env("STOP_LOSS_PCT", "1.5")),
        take_profit_pct=Decimal(_env("TAKE_PROFIT_PCT", "1.5")),
        max_drawdown_pct=Decimal(_env("MAX_DRAWDOWN_PCT", "3.0")),
        trade_cooldown_seconds=int(_env("TRADE_COOLDOWN_SECONDS", "60")),
        max_position_value=Decimal(_env("MAX_POSITION_VALUE", "1000")),
        max_total_exposure=Decimal(_env("MAX_TOTAL_EXPOSURE", "10000")),
        max_positions=int(_env("MAX_POSITIONS", "10")),
    )

    symbols_raw = _env("WATCH_SYMBOLS", "700.HK,TSLA.US")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    watchlist = WatchlistParams(
        use_watchlist=_env("USE_WATCHLIST", "false").lower() in ("true", "1", "yes"),
        market_filter=_env("WATCHLIST_MARKET_FILTER", "US"),
        refresh_interval=int(_env("WATCHLIST_REFRESH_INTERVAL", "300")),
    )

    return TradingConfig(
        paper_trading=_env("PAPER_TRADING", "true").lower() in ("true", "1", "yes"),
        watch_symbols=symbols,
        order_quantity=int(_env("ORDER_QUANTITY", "100")),
        kline_poll_interval=int(_env("KLINE_POLL_INTERVAL", "10")),
        credentials=credentials,
        strategy=strategy,
        risk=risk,
        watchlist=watchlist,
    )
