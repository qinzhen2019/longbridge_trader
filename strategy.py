"""
策略引擎
买入信号: 布林带触碰下轨 且 RSI < 30 (超卖)
卖出信号: 价格回升至布林带中轨 或 盈利达到 take_profit_pct
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from config import StrategyParams
from indicators import IndicatorSnapshot
from logger import get_logger
from strategy_base import BaseStrategy

log = get_logger("strategy")


class Signal(Enum):
    HOLD = auto()
    BUY = auto()
    SELL = auto()


@dataclass
class SignalResult:
    signal: Signal
    reason: str
    price: float
    indicators: IndicatorSnapshot


class BollingerRsiStrategy(BaseStrategy):
    def __init__(self, params: StrategyParams):
        self._params = params

    def evaluate(
        self,
        symbol: str,
        current_price: float,
        indicators: IndicatorSnapshot,
        has_position: bool,
    ) -> SignalResult:
        boll = indicators.bollinger
        rsi = indicators.rsi

        if boll is None or rsi is None:
            return SignalResult(Signal.HOLD, "Insufficient data for indicators", current_price, indicators)

        band_width = boll.upper - boll.lower
        price_position = (current_price - boll.lower) / band_width if band_width > 0 else 0.5

        if not has_position:
            near_lower = current_price <= boll.lower * 1.01
            rsi_oversold = rsi < self._params.rsi_oversold
            rsi_low = rsi < 40
            in_lower_half = price_position < 0.4

            if current_price < (indicators.ema or 0):
                # Filter out buy signals if we are in a downtrend (price < EMA)
                if near_lower and rsi_oversold:
                    log.info("[%s] BUY signal suppressed by Trend Filter (Price %.2f < EMA %.2f)", symbol, current_price, indicators.ema or 0)
                return SignalResult(Signal.HOLD, "Downtrend Filtered", current_price, indicators)

            if near_lower and rsi_oversold:
                reason = f"BUY[强]: price {current_price:.2f} near lower {boll.lower:.2f}, RSI {rsi:.1f} oversold"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.BUY, reason, current_price, indicators)

            if in_lower_half and rsi_low:
                reason = f"BUY[中]: price_pos {price_position:.1%} in lower half, RSI {rsi:.1f} < 40"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.BUY, reason, current_price, indicators)

            if rsi < 35 and current_price < boll.middle:
                reason = f"BUY[弱]: RSI {rsi:.1f} < 35, price {current_price:.2f} below middle {boll.middle:.2f}"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.BUY, reason, current_price, indicators)

        if has_position:
            if current_price >= boll.middle and rsi > 50:
                reason = f"SELL: price {current_price:.2f} >= middle {boll.middle:.2f}, RSI {rsi:.1f}"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.SELL, reason, current_price, indicators)

            if rsi > self._params.rsi_overbought:
                reason = f"SELL: RSI {rsi:.1f} > {self._params.rsi_overbought} (overbought)"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.SELL, reason, current_price, indicators)

            if price_position > 0.8:
                reason = f"SELL: price_pos {price_position:.1%} near upper band"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.SELL, reason, current_price, indicators)

        return SignalResult(Signal.HOLD, "No signal", current_price, indicators)
