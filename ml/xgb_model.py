"""
XGBoost 模型管理
负责训练、预测、保存和加载 XGBoost 分类器。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from logger import get_logger

log = get_logger("xgb_model")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class XGBModelManager:
    """管理 XGBoost 二分类模型 (预测未来涨 / 跌)."""

    def __init__(self, model_dir: str | Path | None = None):
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError:
            raise ImportError(
                "xgboost is required for XGBoost strategy. "
                "Install it with: pip install xgboost"
            )

        self._model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model: "xgb.XGBClassifier | None" = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        eval_split: float = 0.2,
    ) -> dict:
        """
        训练 XGBoost 分类器。

        Args:
            X: 特征矩阵, shape (n_samples, n_features)
            y: 标签数组, shape (n_samples,)  0 = 跌, 1 = 涨
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            eval_split: 验证集比例

        Returns:
            dict with training metrics
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=eval_split, shuffle=False  # 时间序列不能 shuffle
        )

        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = self._model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        log.info("Training complete — val accuracy: %.4f", acc)
        log.info("  Precision(1): %.4f  Recall(1): %.4f",
                 report.get("1", {}).get("precision", 0),
                 report.get("1", {}).get("recall", 0))

        return {
            "accuracy": acc,
            "report": report,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }

    def predict_proba(self, features: np.ndarray) -> float:
        """
        预测单条样本的"上涨概率"。

        Args:
            features: shape (n_features,) or (1, n_features)

        Returns:
            上涨概率 [0.0, 1.0]
        """
        if self._model is None:
            log.warning("Model not loaded, returning 0.5")
            return 0.5

        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self._model.predict_proba(features)
        # proba shape: (1, 2) → [prob_down, prob_up]
        return float(proba[0][1])

    def save(self, name: str = "xgb_model") -> Path:
        """保存模型到磁盘。"""
        if self._model is None:
            raise RuntimeError("No model to save")

        path = self._model_dir / f"{name}.json"
        self._model.save_model(str(path))
        log.info("Model saved to %s", path)
        return path

    def load(self, name: str = "xgb_model") -> bool:
        """从磁盘加载模型。"""
        import xgboost as xgb

        path = self._model_dir / f"{name}.json"
        if not path.exists():
            log.warning("Model file not found: %s", path)
            return False

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))
        log.info("Model loaded from %s", path)
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
