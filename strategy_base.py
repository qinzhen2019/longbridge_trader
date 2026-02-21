"""
统一策略接口
所有交易策略（规则型、ML 型）均继承此基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from indicators import IndicatorSnapshot


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


class BaseStrategy(ABC):
    """Strategy interface that all strategies must implement."""

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        current_price: float,
        indicators: IndicatorSnapshot,
        has_position: bool,
    ) -> SignalResult:
        """
        Evaluate current market state and return a trading signal.

        Args:
            symbol: Ticker symbol (e.g. "TSLA.US")
            current_price: Current real-time price
            indicators: Pre-computed technical indicators (BB, RSI, EMA)
            has_position: Whether we currently hold a position in this symbol

        Returns:
            SignalResult containing HOLD / BUY / SELL and reasoning
        """
        ...

    def on_new_candles(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        volumes: list[float] | None = None,
    ) -> None:
        """
        Optional hook — called when fresh candle data arrives.
        ML strategies can use this to update rolling feature buffers.
        Rule-based strategies can ignore this.
        """
        pass
