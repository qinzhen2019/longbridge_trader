"""
特征工程模块
从原始 K 线 closes / highs / lows / volumes 中提取可供 ML 模型使用的特征向量。
"""

from __future__ import annotations

import numpy as np

from indicators import (
    compute_bollinger,
    compute_rsi,
    compute_ema,
    compute_ma,
)


def _safe(val: float | None, default: float = 0.0) -> float:
    return val if val is not None else default


def build_feature_vector(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    boll_period: int = 20,
    boll_std: float = 2.0,
    rsi_period: int = 14,
    ema_period: int = 60,
) -> np.ndarray | None:
    """
    从 K 线序列构建一个 1-D 特征向量。
    至少需要 max(boll_period, rsi_period+1, ema_period) 根 K 线。

    Returns:
        numpy array of shape (n_features,) or None if data insufficient.
    """
    min_len = max(boll_period, rsi_period + 1, ema_period, 20)
    if len(closes) < min_len:
        return None

    price = closes[-1]
    features: list[float] = []

    # ── 1. 价格变化率 (ROC) ──
    for lag in [1, 3, 5, 10]:
        if len(closes) > lag:
            roc = (closes[-1] - closes[-1 - lag]) / closes[-1 - lag]
            features.append(roc)
        else:
            features.append(0.0)

    # ── 2. 波动率 (近20根的标准差 / 均价) ──
    recent_20 = np.array(closes[-20:])
    volatility = float(np.std(recent_20)) / float(np.mean(recent_20))
    features.append(volatility)

    # ── 3. RSI ──
    rsi = compute_rsi(closes, rsi_period)
    features.append(_safe(rsi, 50.0) / 100.0)  # normalize to [0, 1]

    # ── 4. 布林带位置 ──
    boll = compute_bollinger(closes, boll_period, boll_std)
    if boll and boll.upper != boll.lower:
        boll_pos = (price - boll.lower) / (boll.upper - boll.lower)
        boll_width = (boll.upper - boll.lower) / boll.middle
    else:
        boll_pos = 0.5
        boll_width = 0.0
    features.append(boll_pos)
    features.append(boll_width)

    # ── 5. EMA 偏离度 ──
    ema = compute_ema(closes, ema_period)
    if ema and ema != 0:
        ema_deviation = (price - ema) / ema
    else:
        ema_deviation = 0.0
    features.append(ema_deviation)

    # ── 6. MA 趋势 (MA5 vs MA20) ──
    ma5 = compute_ma(closes, 5)
    ma20 = compute_ma(closes, 20)
    if ma5 is not None and ma20 is not None and ma20 != 0:
        ma_ratio = (ma5 - ma20) / ma20
    else:
        ma_ratio = 0.0
    features.append(ma_ratio)

    # ── 7. 近期高低点位置 ──
    if highs and lows and len(highs) >= 20 and len(lows) >= 20:
        high_20 = max(highs[-20:])
        low_20 = min(lows[-20:])
        span = high_20 - low_20
        hl_pos = (price - low_20) / span if span > 0 else 0.5
        features.append(hl_pos)
    else:
        features.append(0.5)

    # ── 8. 成交量变化 ──
    if volumes and len(volumes) >= 20:
        vol_ma20 = float(np.mean(volumes[-20:]))
        vol_ratio = volumes[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
        features.append(vol_ratio)
    else:
        features.append(1.0)

    # ── 9. 滞后收益率 (lag returns) ──
    for lag in [1, 2, 3]:
        if len(closes) > lag:
            features.append(closes[-lag] / closes[-lag - 1] - 1.0 if closes[-lag - 1] != 0 else 0.0)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    "roc_1", "roc_3", "roc_5", "roc_10",
    "volatility_20",
    "rsi_norm",
    "boll_position", "boll_width",
    "ema_deviation",
    "ma5_ma20_ratio",
    "hl_position_20",
    "volume_ratio_20",
    "lag_return_1", "lag_return_2", "lag_return_3",
]
