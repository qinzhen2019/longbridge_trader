"""
强化学习策略
使用训练好的 RL Agent 直接输出 Buy / Sell / Hold 动作。
"""

from __future__ import annotations

import numpy as np

from config import StrategyParams
from indicators import IndicatorSnapshot
from logger import get_logger
from strategy_base import BaseStrategy, Signal, SignalResult
from ml.feature_engineer import build_feature_vector
from ml.rl_agent import RLAgentManager

log = get_logger("rl_strategy")

# RL action → Signal mapping
ACTION_MAP = {
    0: Signal.HOLD,
    1: Signal.BUY,
    2: Signal.SELL,
}


class RLStrategy(BaseStrategy):
    def __init__(
        self,
        params: StrategyParams,
        algo: str = "PPO",
        model_name: str = "rl_model",
    ):
        self._params = params

        # 缓存 K 线
        self._closes_buffer: dict[str, list[float]] = {}
        self._highs_buffer: dict[str, list[float]] = {}
        self._lows_buffer: dict[str, list[float]] = {}
        self._volumes_buffer: dict[str, list[float]] = {}

        # 加载 Agent
        self._agent = RLAgentManager(algo=algo)
        if not self._agent.load(model_name):
            log.warning(
                "RL model '%s' (%s) not found. Strategy will output HOLD. "
                "Run `python ml/train_rl.py` first.",
                model_name, algo,
            )

    def on_new_candles(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        volumes: list[float] | None = None,
    ) -> None:
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
        if not self._agent.is_loaded:
            return SignalResult(Signal.HOLD, "RL model not loaded", current_price, indicators)

        closes = self._closes_buffer.get(symbol)
        if not closes or len(closes) < 60:
            return SignalResult(Signal.HOLD, "Insufficient candle data", current_price, indicators)

        # 构建基础特征
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

        # 附加持仓状态 (与 trading_env 一致)
        has_pos_f = 1.0 if has_position else 0.0
        unrealized_pnl = 0.0  # 实盘中可从 risk_control 获取
        obs = np.concatenate([features, [has_pos_f, unrealized_pnl]]).astype(np.float32)

        # Agent 决策
        action = self._agent.predict(obs)
        signal = ACTION_MAP.get(action, Signal.HOLD)

        # 安全检查: 不允许无仓位时卖出, 有仓位时买入
        if signal == Signal.BUY and has_position:
            signal = Signal.HOLD
        if signal == Signal.SELL and not has_position:
            signal = Signal.HOLD

        reason = f"RL_{signal.name}: action={action}"
        log.info("[%s] %s", symbol, reason)

        return SignalResult(signal, reason, current_price, indicators)
