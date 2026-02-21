"""
自定义 Gymnasium 交易环境
用于强化学习 Agent 的训练和评估。
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required for RL strategy. "
        "Install it with: pip install gymnasium"
    )

from ml.feature_engineer import build_feature_vector


class TradingEnv(gym.Env):
    """
    股票交易 Gymnasium 环境。

    Observation:
        技术指标特征向量 + [持仓状态, 未实现收益率]

    Actions:
        0 = Hold, 1 = Buy, 2 = Sell

    Reward:
        基于持仓收益变化的即时奖励
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        closes: list[float],
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        volumes: list[float] | None = None,
        initial_balance: float = 100_000.0,
        commission_rate: float = 0.001,
        max_position_pct: float = 0.3,
        lookback: int = 60,
    ):
        super().__init__()

        self._closes = np.array(closes, dtype=np.float64)
        self._highs = np.array(highs, dtype=np.float64) if highs else None
        self._lows = np.array(lows, dtype=np.float64) if lows else None
        self._volumes = np.array(volumes, dtype=np.float64) if volumes else None

        self._initial_balance = initial_balance
        self._commission_rate = commission_rate
        self._max_position_pct = max_position_pct
        self._lookback = lookback

        # 特征维度 = feature_engineer 输出(15) + 持仓状态(1) + 未实现PnL(1)
        self._n_features = 17

        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._n_features,),
            dtype=np.float32,
        )

        self._start_idx = self._lookback
        self._end_idx = len(self._closes) - 1

        # State variables
        self._current_idx = self._start_idx
        self._balance = initial_balance
        self._shares = 0
        self._entry_price = 0.0
        self._total_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_idx = self._start_idx
        self._balance = self._initial_balance
        self._shares = 0
        self._entry_price = 0.0
        self._total_reward = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        price = self._closes[self._current_idx]
        reward = 0.0

        # ── Execute action ──
        if action == 1 and self._shares == 0:  # Buy
            max_invest = self._balance * self._max_position_pct
            shares_to_buy = int(max_invest / price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price * (1 + self._commission_rate)
                self._balance -= cost
                self._shares = shares_to_buy
                self._entry_price = price

        elif action == 2 and self._shares > 0:  # Sell
            proceeds = self._shares * price * (1 - self._commission_rate)
            pnl = proceeds - self._shares * self._entry_price
            reward = pnl / self._initial_balance * 100  # PnL as % of initial
            self._balance += proceeds
            self._shares = 0
            self._entry_price = 0.0

        # 持仓时的浮动收益作为小额奖励
        if self._shares > 0:
            unrealized = (price - self._entry_price) / self._entry_price
            reward += unrealized * 0.1  # small incentive for winners

        self._total_reward += reward
        self._current_idx += 1

        terminated = self._current_idx >= self._end_idx
        truncated = False

        # 爆仓检查
        portfolio_value = self._balance + self._shares * price
        if portfolio_value < self._initial_balance * 0.9:
            terminated = True
            reward -= 10.0  # heavy penalty for drawdown

        obs = self._get_obs()
        info = {
            "balance": self._balance,
            "shares": self._shares,
            "portfolio_value": portfolio_value,
            "total_reward": self._total_reward,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        idx = self._current_idx
        closes_slice = self._closes[max(0, idx - self._lookback): idx + 1].tolist()

        highs_slice = self._highs[max(0, idx - self._lookback): idx + 1].tolist() if self._highs is not None else None
        lows_slice = self._lows[max(0, idx - self._lookback): idx + 1].tolist() if self._lows is not None else None
        vols_slice = self._volumes[max(0, idx - self._lookback): idx + 1].tolist() if self._volumes is not None else None

        features = build_feature_vector(closes_slice, highs_slice, lows_slice, vols_slice)

        if features is None:
            features = np.zeros(15, dtype=np.float32)

        # 附加状态
        has_position = 1.0 if self._shares > 0 else 0.0
        if self._shares > 0 and self._entry_price > 0:
            unrealized_pnl = (self._closes[idx] - self._entry_price) / self._entry_price
        else:
            unrealized_pnl = 0.0

        obs = np.concatenate([features, [has_position, unrealized_pnl]])
        return obs.astype(np.float32)

    def render(self):
        idx = self._current_idx
        price = self._closes[idx]
        pv = self._balance + self._shares * price
        print(
            f"Step {idx}: Price={price:.2f} | Balance={self._balance:.2f} | "
            f"Shares={self._shares} | Portfolio={pv:.2f}"
        )
