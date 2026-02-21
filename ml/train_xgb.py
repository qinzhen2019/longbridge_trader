#!/usr/bin/env python3
"""
XGBoost 离线训练脚本
使用长桥 API 拉取历史 K 线数据，生成标签，训练分类器。

Usage:
    python ml/train_xgb.py --symbols TSLA.US,AAPL.US --days 365

默认保存模型到 models/xgb_model.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import load_config
from logger import setup_logging, get_logger
from ml.feature_engineer import build_feature_vector, FEATURE_NAMES
from ml.xgb_model import XGBModelManager

log = get_logger("train_xgb")


def fetch_historical_data(symbols: list[str], kline_count: int = 500):
    """从长桥 API 拉取日线数据。"""
    from longport.openapi import Config, QuoteContext, Period, AdjustType

    cfg = load_config()
    lb_config = Config(
        app_key=cfg.credentials.app_key,
        app_secret=cfg.credentials.app_secret,
        access_token=cfg.credentials.access_token,
    )
    quote_ctx = QuoteContext(lb_config)

    all_data = {}
    for symbol in symbols:
        try:
            candles = quote_ctx.candlesticks(
                symbol, Period.Day, kline_count, AdjustType.NoAdjust
            )
            if candles:
                all_data[symbol] = {
                    "closes": [float(c.close) for c in candles],
                    "highs": [float(c.high) for c in candles],
                    "lows": [float(c.low) for c in candles],
                    "volumes": [float(c.volume) for c in candles],
                }
                log.info("Fetched %d daily candles for %s", len(candles), symbol)
            else:
                log.warning("No data for %s", symbol)
        except Exception:
            log.exception("Failed to fetch data for %s", symbol)

    return all_data


def generate_labels(closes: list[float], horizon: int = 5) -> np.ndarray:
    """
    生成标签: 未来 horizon 根 K 线的收益率 > 0 则为 1 (涨), 否则为 0 (跌)。
    最后 horizon 根 K 线没有标签, 设为 -1。
    """
    labels = np.full(len(closes), -1, dtype=np.int32)
    for i in range(len(closes) - horizon):
        future_return = (closes[i + horizon] - closes[i]) / closes[i]
        labels[i] = 1 if future_return > 0 else 0
    return labels


def build_dataset(
    all_data: dict,
    horizon: int = 5,
    boll_period: int = 20,
    boll_std: float = 2.0,
    rsi_period: int = 14,
    ema_period: int = 60,
):
    """从多支股票数据生成训练集。"""
    X_list, y_list = [], []

    for symbol, data in all_data.items():
        closes = data["closes"]
        highs = data["highs"]
        lows = data["lows"]
        volumes = data["volumes"]

        labels = generate_labels(closes, horizon)

        min_lookback = max(boll_period, rsi_period + 1, ema_period, 20)

        for i in range(min_lookback, len(closes)):
            if labels[i] == -1:
                continue  # no valid label

            features = build_feature_vector(
                closes[: i + 1],
                highs=highs[: i + 1],
                lows=lows[: i + 1],
                volumes=volumes[: i + 1],
                boll_period=boll_period,
                boll_std=boll_std,
                rsi_period=rsi_period,
                ema_period=ema_period,
            )

            if features is not None:
                X_list.append(features)
                y_list.append(labels[i])

        log.info(
            "%s: generated %d samples (label dist: %d up / %d down)",
            symbol,
            sum(1 for l in labels if l != -1),
            sum(1 for l in labels if l == 1),
            sum(1 for l in labels if l == 0),
        )

    if not X_list:
        return None, None

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost trading model")
    parser.add_argument(
        "--symbols", type=str, default="TSLA.US,AAPL.US,NVDA.US",
        help="Comma-separated symbols to train on",
    )
    parser.add_argument("--klines", type=int, default=500, help="Number of daily candles to fetch")
    parser.add_argument("--horizon", type=int, default=5, help="Future return horizon (days)")
    parser.add_argument("--estimators", type=int, default=200, help="XGBoost n_estimators")
    parser.add_argument("--depth", type=int, default=6, help="XGBoost max_depth")
    parser.add_argument("--lr", type=float, default=0.05, help="XGBoost learning_rate")
    parser.add_argument("--model-name", type=str, default="xgb_model", help="Model name for save")

    args = parser.parse_args()
    setup_logging()

    symbols = [s.strip() for s in args.symbols.split(",")]
    log.info("=" * 60)
    log.info("XGBoost Training Pipeline")
    log.info("Symbols: %s", symbols)
    log.info("K-lines: %d, Horizon: %d days", args.klines, args.horizon)
    log.info("=" * 60)

    # Step 1: Fetch data
    log.info("Step 1/3: Fetching historical data...")
    all_data = fetch_historical_data(symbols, args.klines)
    if not all_data:
        log.error("No data fetched, exiting")
        return

    # Step 2: Build dataset
    log.info("Step 2/3: Building feature dataset...")
    X, y = build_dataset(all_data, args.horizon)
    if X is None:
        log.error("No valid samples generated, exiting")
        return

    log.info("Dataset: %d samples, %d features", X.shape[0], X.shape[1])
    log.info("Feature names: %s", FEATURE_NAMES)
    log.info("Label distribution: %d up (%.1f%%) / %d down (%.1f%%)",
             np.sum(y == 1), np.mean(y == 1) * 100,
             np.sum(y == 0), np.mean(y == 0) * 100)

    # Step 3: Train model
    log.info("Step 3/3: Training XGBoost...")
    mgr = XGBModelManager()
    metrics = mgr.train(
        X, y,
        n_estimators=args.estimators,
        max_depth=args.depth,
        learning_rate=args.lr,
    )

    log.info("Validation accuracy: %.4f", metrics["accuracy"])

    # Save
    save_path = mgr.save(args.model_name)
    log.info("Model saved to: %s", save_path)
    log.info("Done! You can now set STRATEGY_TYPE=xgboost in .env")


if __name__ == "__main__":
    main()
