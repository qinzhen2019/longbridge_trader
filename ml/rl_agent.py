"""
RL Agent 管理
封装 Stable-Baselines3 的训练、预测、持久化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from logger import get_logger

log = get_logger("rl_agent")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class RLAgentManager:
    """管理 Stable-Baselines3 RL 模型."""

    SUPPORTED_ALGOS = {"PPO", "DQN", "A2C"}

    def __init__(self, algo: str = "PPO", model_dir: str | Path | None = None):
        if algo.upper() not in self.SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm: {algo}. Choose from {self.SUPPORTED_ALGOS}")

        try:
            import stable_baselines3  # noqa: F401
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for RL strategy. "
                "Install it with: pip install stable-baselines3"
            )

        self._algo_name = algo.upper()
        self._model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _get_algo_class(self):
        """动态获取 SB3 算法类。"""
        from stable_baselines3 import PPO, DQN, A2C
        return {"PPO": PPO, "DQN": DQN, "A2C": A2C}[self._algo_name]

    def train(
        self,
        env,
        total_timesteps: int = 50_000,
        learning_rate: float = 3e-4,
        verbose: int = 1,
    ) -> None:
        """
        训练 RL agent。

        Args:
            env: Gymnasium-compatible trading environment
            total_timesteps: 总训练步数
            learning_rate: 学习率
            verbose: 日志详细程度
        """
        AlgoClass = self._get_algo_class()

        log.info("Training %s for %d timesteps...", self._algo_name, total_timesteps)

        self._model = AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        self._model.learn(total_timesteps=total_timesteps)
        log.info("Training complete")

    def predict(self, obs: np.ndarray) -> int:
        """
        给定 observation 返回动作。

        Args:
            obs: shape (n_features,)

        Returns:
            action: 0=Hold, 1=Buy, 2=Sell
        """
        if self._model is None:
            log.warning("Model not loaded, returning Hold (0)")
            return 0

        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def save(self, name: str = "rl_model") -> Path:
        """保存模型。"""
        if self._model is None:
            raise RuntimeError("No model to save")

        path = self._model_dir / f"{name}_{self._algo_name}"
        self._model.save(str(path))
        log.info("RL model saved to %s", path)
        return path

    def load(self, name: str = "rl_model") -> bool:
        """加载模型。"""
        AlgoClass = self._get_algo_class()
        path = self._model_dir / f"{name}_{self._algo_name}"

        # SB3 保存时会自动加扩展名
        zip_path = Path(str(path) + ".zip")
        if not zip_path.exists():
            log.warning("RL model file not found: %s", zip_path)
            return False

        self._model = AlgoClass.load(str(path))
        log.info("RL model loaded from %s", path)
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
