"""
技术指标计算: 布林带 (Bollinger Bands) 和 RSI
仅依赖 numpy，无需额外 TA 库。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np


@dataclass
class BollingerBands:
    upper: float
    middle: float
    lower: float


@dataclass
class IndicatorSnapshot:
    bollinger: BollingerBands | None
    rsi: float | None
    ema: float | None


def compute_bollinger(closes: list[float], period: int = 20, num_std: float = 2.0) -> BollingerBands | None:
    if len(closes) < period:
        return None
    window = np.array(closes[-period:])
    middle = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    return BollingerBands(
        upper=middle + num_std * std,
        middle=middle,
        lower=middle - num_std * std,
    )


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's smoothing RSI."""
    if len(closes) < period + 1:
        return None

    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_ema(closes: list[float], period: int = 60) -> float | None:
    if len(closes) < period:
        return None
    
    # We use a simple numpy implementation of EMA
    alpha = 2 / (period + 1.0)
    ema = float(closes[0])
    for price in closes[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema


def compute_indicators(
    closes: list[float],
    boll_period: int = 20,
    boll_std: float = 2.0,
    rsi_period: int = 14,
    ema_period: int = 60,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        bollinger=compute_bollinger(closes, boll_period, boll_std),
        rsi=compute_rsi(closes, rsi_period),
        ema=compute_ema(closes, ema_period),
    )


def compute_ma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))
