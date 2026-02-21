#!/usr/bin/env python3
"""
RL 离线训练脚本
使用长桥 API 拉取历史 K 线 → 构建 TradingEnv → 训练 PPO Agent → 保存模型。

Usage:
    python ml/train_rl.py --symbols TSLA.US --timesteps 50000 --algo PPO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import load_config
from logger import setup_logging, get_logger
from ml.trading_env import TradingEnv
from ml.rl_agent import RLAgentManager

log = get_logger("train_rl")


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
            if candles and len(candles) > 100:
                all_data[symbol] = {
                    "closes": [float(c.close) for c in candles],
                    "highs": [float(c.high) for c in candles],
                    "lows": [float(c.low) for c in candles],
                    "volumes": [float(c.volume) for c in candles],
                }
                log.info("Fetched %d daily candles for %s", len(candles), symbol)
            else:
                log.warning("Insufficient data for %s (%d candles)", symbol, len(candles) if candles else 0)
        except Exception:
            log.exception("Failed to fetch data for %s", symbol)

    return all_data


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--symbols", type=str, default="TSLA.US", help="Training symbol")
    parser.add_argument("--klines", type=int, default=500, help="Number of daily candles")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training timesteps")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN", "A2C"], help="RL algorithm")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="rl_model", help="Model name for save")

    args = parser.parse_args()
    setup_logging()

    symbols = [s.strip() for s in args.symbols.split(",")]
    log.info("=" * 60)
    log.info("RL Training Pipeline (%s)", args.algo)
    log.info("Symbols: %s", symbols)
    log.info("Timesteps: %d", args.timesteps)
    log.info("=" * 60)

    # Step 1: Fetch data
    log.info("Step 1/3: Fetching historical data...")
    all_data = fetch_historical_data(symbols, args.klines)
    if not all_data:
        log.error("No data fetched, exiting")
        return

    # 使用第一个符号的数据训练
    symbol = list(all_data.keys())[0]
    data = all_data[symbol]

    log.info("Training on %s (%d candles)", symbol, len(data["closes"]))

    # Step 2: Create environment
    log.info("Step 2/3: Creating TradingEnv...")
    env = TradingEnv(
        closes=data["closes"],
        highs=data["highs"],
        lows=data["lows"],
        volumes=data["volumes"],
        initial_balance=100_000.0,
        commission_rate=0.001,
        max_position_pct=0.3,
    )

    # Step 3: Train agent
    log.info("Step 3/3: Training %s agent for %d timesteps...", args.algo, args.timesteps)
    agent = RLAgentManager(algo=args.algo)
    agent.train(
        env=env,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        verbose=1,
    )

    # Save
    save_path = agent.save(args.model_name)
    log.info("Model saved to: %s", save_path)

    # Quick evaluation
    log.info("Running evaluation episode...")
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    log.info(
        "Evaluation: %d steps, total_reward=%.2f, final_portfolio=%.2f",
        steps, total_reward, info.get("portfolio_value", 0),
    )
    log.info("Done! You can now set STRATEGY_TYPE=rl in .env")


if __name__ == "__main__":
    main()
