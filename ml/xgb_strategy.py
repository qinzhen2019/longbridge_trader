"""
XGBoost 监督学习策略
当模型预测"上涨概率"超过阈值时才允许发出 BUY 信号。
SELL 逻辑沿用布林带/RSI 规则。
"""

from __future__ import annotations

import numpy as np

from config import StrategyParams
from indicators import IndicatorSnapshot
from logger import get_logger
from strategy import Signal, SignalResult
from strategy_base import BaseStrategy
from ml.feature_engineer import build_feature_vector
from ml.xgb_model import XGBModelManager

log = get_logger("xgb_strategy")


class XGBoostStrategy(BaseStrategy):
    def __init__(
        self,
        params: StrategyParams,
        model_name: str = "xgb_model",
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
    ):
        self._params = params
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

        # 缓存每个 symbol 的 K 线数据
        self._closes_buffer: dict[str, list[float]] = {}
        self._highs_buffer: dict[str, list[float]] = {}
        self._lows_buffer: dict[str, list[float]] = {}
        self._volumes_buffer: dict[str, list[float]] = {}

        # 加载模型
        self._model = XGBModelManager()
        if not self._model.load(model_name):
            log.warning(
                "XGBoost model '%s' not found. Strategy will output HOLD "
                "until a model is trained and saved. Run `python ml/train_xgb.py` first.",
                model_name,
            )

    def on_new_candles(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        volumes: list[float] | None = None,
    ) -> None:
        """缓存最新 K 线，供特征工程使用。"""
        self._closes_buffer[symbol] = closes
        if highs:
            self._highs_buffer[symbol] = highs
        if lows:
            self._lows_buffer[symbol] = lows
        if volumes:
            self._volumes_buffer[symbol] = volumes

    def evaluate(
        self,
        symbol: str,
        current_price: float,
        indicators: IndicatorSnapshot,
        has_position: bool,
    ) -> SignalResult:
        # 如果模型没加载，保持观望
        if not self._model.is_loaded:
            return SignalResult(Signal.HOLD, "Model not loaded", current_price, indicators)

        closes = self._closes_buffer.get(symbol)
        if not closes or len(closes) < 60:
            return SignalResult(Signal.HOLD, "Insufficient candle data", current_price, indicators)

        features = build_feature_vector(
            closes,
            highs=self._highs_buffer.get(symbol),
            lows=self._lows_buffer.get(symbol),
            volumes=self._volumes_buffer.get(symbol),
            boll_period=self._params.boll_period,
            boll_std=self._params.boll_std_dev,
            rsi_period=self._params.rsi_period,
            ema_period=self._params.trend_ema_period,
        )

        if features is None:
            return SignalResult(Signal.HOLD, "Feature extraction failed", current_price, indicators)

        prob_up = self._model.predict_proba(features)

        log.info(
            "[%s] XGBoost prob_up=%.3f (buy_thresh=%.2f, sell_thresh=%.2f) pos=%s",
            symbol, prob_up, self._buy_threshold, self._sell_threshold, has_position,
        )

        # ── BUY logic ──
        if not has_position and prob_up >= self._buy_threshold:
            reason = (
                f"XGB_BUY: prob_up={prob_up:.3f} >= {self._buy_threshold:.2f}"
            )
            log.info("[%s] %s", symbol, reason)
            return SignalResult(Signal.BUY, reason, current_price, indicators)

        # ── SELL logic ──
        if has_position:
            # 模型看跌 → 卖出
            if prob_up <= self._sell_threshold:
                reason = f"XGB_SELL: prob_up={prob_up:.3f} <= {self._sell_threshold:.2f}"
                log.info("[%s] %s", symbol, reason)
                return SignalResult(Signal.SELL, reason, current_price, indicators)

            # 传统指标辅助卖出 (布林带上轨 + RSI 超买)
            boll = indicators.bollinger
            rsi = indicators.rsi
            if boll and rsi:
                if rsi > self._params.rsi_overbought:
                    reason = f"XGB_SELL(RSI): RSI={rsi:.1f} overbought"
                    return SignalResult(Signal.SELL, reason, current_price, indicators)

        return SignalResult(Signal.HOLD, f"XGB_HOLD: prob_up={prob_up:.3f}", current_price, indicators)
