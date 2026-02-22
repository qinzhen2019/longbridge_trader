#!/usr/bin/env python3
"""
回测评估引擎
支持 XGBoost 和 RL 模型在历史数据上的模拟交易表现评估。

Usage (standalone):
    python ml/backtest.py --symbol TSLA.US --model-type xgboost --model-name xgb_model
    python ml/backtest.py --symbol TSLA.US --model-type rl --algo PPO --model-name rl_model
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from logger import get_logger
from ml.feature_engineer import build_feature_vector

log = get_logger("backtest")


# ─── Data Structures ────────────────────────────────────────

@dataclass
class Trade:
    """A single completed round-trip trade."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_bars: int


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    symbol: str
    model_type: str
    total_bars: int
    train_bars: int
    test_bars: int

    # Performance
    initial_balance: float
    final_balance: float
    total_return_pct: float
    buy_hold_return_pct: float
    excess_return_pct: float

    # Risk
    max_drawdown_pct: float
    sharpe_ratio: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float

    # Trade list
    trades: list[Trade] = field(default_factory=list)

    # Equity curve for display
    equity_curve: list[float] = field(default_factory=list)
    price_curve: list[float] = field(default_factory=list)

    # Action log
    actions: list[int] = field(default_factory=list)


# ─── Backtest Engine ────────────────────────────────────────

def run_backtest(
    closes: list[float],
    highs: list[float] | None,
    lows: list[float] | None,
    volumes: list[float] | None,
    model_type: str,
    model,
    symbol: str = "UNKNOWN",
    test_ratio: float = 0.2,
    initial_balance: float = 100_000.0,
    commission_rate: float = 0.001,
    max_position_pct: float = 0.3,
    boll_period: int = 20,
    boll_std: float = 2.0,
    rsi_period: int = 14,
    ema_period: int = 60,
    xgb_buy_threshold: float = 0.6,
    xgb_sell_threshold: float = 0.4,
) -> BacktestResult:
    """
    Run a walk-forward backtest: train on first (1-test_ratio) of data,
    evaluate model on the last test_ratio portion.

    Args:
        closes: Full price history
        model_type: 'xgboost' or 'rl'
        model: Loaded model manager (XGBModelManager or RLAgentManager)
        test_ratio: Fraction of data reserved for testing

    Returns:
        BacktestResult with detailed metrics
    """
    n = len(closes)
    split_idx = int(n * (1 - test_ratio))
    lookback = max(boll_period, rsi_period + 1, ema_period, 60)

    # Ensure test set starts with enough lookback
    test_start = max(split_idx, lookback)
    test_closes = closes[test_start:]
    test_n = len(test_closes)

    if test_n < 10:
        raise ValueError(f"Test set too small: {test_n} bars (need ≥ 10)")

    # ── Simulate trading on test set ──
    balance = initial_balance
    shares = 0
    entry_price = 0.0

    trades: list[Trade] = []
    equity_curve: list[float] = [initial_balance]
    price_curve: list[float] = [test_closes[0]]
    actions_log: list[int] = []
    peak_equity = initial_balance
    max_dd = 0.0
    daily_returns: list[float] = []

    for i in range(test_n):
        global_idx = test_start + i
        current_price = closes[global_idx]

        # Build feature vector using all data up to this point
        hist_closes = closes[:global_idx + 1]
        hist_highs = highs[:global_idx + 1] if highs else None
        hist_lows = lows[:global_idx + 1] if lows else None
        hist_vols = volumes[:global_idx + 1] if volumes else None

        features = build_feature_vector(
            hist_closes, hist_highs, hist_lows, hist_vols,
            boll_period=boll_period, boll_std=boll_std,
            rsi_period=rsi_period, ema_period=ema_period,
        )

        if features is None:
            actions_log.append(0)
            pv = balance + shares * current_price
            equity_curve.append(pv)
            price_curve.append(current_price)
            continue

        has_position = shares > 0

        # ── Get model signal ──
        action = 0  # default: hold
        if model_type == "xgboost":
            prob_up = model.predict_proba(features)
            if not has_position and prob_up >= xgb_buy_threshold:
                action = 1  # buy
            elif has_position and prob_up <= xgb_sell_threshold:
                action = 2  # sell
        elif model_type == "rl":
            # RL needs position state appended
            has_pos_f = 1.0 if has_position else 0.0
            unrealized = (current_price - entry_price) / entry_price if has_position and entry_price > 0 else 0.0
            obs = np.concatenate([features, [has_pos_f, unrealized]]).astype(np.float32)
            action = model.predict(obs)
            # Safety: no buy if already holding, no sell if no position
            if action == 1 and has_position:
                action = 0
            if action == 2 and not has_position:
                action = 0

        actions_log.append(action)

        # ── Execute trade ──
        if action == 1 and not has_position:  # Buy
            max_invest = balance * max_position_pct
            shares_to_buy = int(max_invest / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + commission_rate)
                balance -= cost
                shares = shares_to_buy
                entry_price = current_price

        elif action == 2 and has_position:  # Sell
            proceeds = shares * current_price * (1 - commission_rate)
            pnl = proceeds - shares * entry_price * (1 + commission_rate)
            pnl_pct = (current_price - entry_price) / entry_price * 100

            trades.append(Trade(
                entry_idx=0, exit_idx=i,
                entry_price=entry_price, exit_price=current_price,
                shares=shares, pnl=pnl, pnl_pct=pnl_pct,
                hold_bars=0,
            ))

            balance += proceeds
            shares = 0
            entry_price = 0.0

        # Track equity
        portfolio_value = balance + shares * current_price
        equity_curve.append(portfolio_value)
        price_curve.append(current_price)

        # Track max drawdown
        if portfolio_value > peak_equity:
            peak_equity = portfolio_value
        dd = (peak_equity - portfolio_value) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        # Track daily returns
        if len(equity_curve) >= 2:
            prev = equity_curve[-2]
            if prev > 0:
                daily_returns.append((portfolio_value - prev) / prev)

    # ── Close any open position at end ──
    if shares > 0:
        final_price = closes[-1]
        proceeds = shares * final_price * (1 - commission_rate)
        pnl = proceeds - shares * entry_price * (1 + commission_rate)
        pnl_pct = (final_price - entry_price) / entry_price * 100
        trades.append(Trade(
            entry_idx=0, exit_idx=test_n - 1,
            entry_price=entry_price, exit_price=final_price,
            shares=shares, pnl=pnl, pnl_pct=pnl_pct,
            hold_bars=0,
        ))
        balance += proceeds
        shares = 0

    # ── Compute metrics ──
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100
    buy_hold_return = (test_closes[-1] - test_closes[0]) / test_closes[0] * 100
    excess = total_return - buy_hold_return

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning) / len(trades) * 100 if trades else 0

    avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0.0

    total_profit = sum(t.pnl for t in winning) if winning else 0
    total_loss = abs(sum(t.pnl for t in losing)) if losing else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0

    # Sharpe ratio (annualized, assuming daily bars)
    if daily_returns and len(daily_returns) > 1:
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return BacktestResult(
        symbol=symbol,
        model_type=model_type,
        total_bars=n,
        train_bars=split_idx,
        test_bars=test_n,
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_return_pct=total_return,
        buy_hold_return_pct=buy_hold_return,
        excess_return_pct=excess,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        avg_win_pct=float(avg_win),
        avg_loss_pct=float(avg_loss),
        profit_factor=profit_factor,
        trades=trades,
        equity_curve=equity_curve,
        price_curve=price_curve,
        actions=actions_log,
    )


def print_backtest_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to console."""
    r = result

    print(f"\n{'═' * 72}")
    print(f"  📊 回测评估报告 — {r.symbol} ({r.model_type.upper()})")
    print(f"{'═' * 72}")

    # Data split
    print(f"\n  ── 数据划分 ──")
    print(f"  总数据:   {r.total_bars} 根日线")
    print(f"  训练集:   {r.train_bars} 根 ({r.train_bars/r.total_bars:.0%})")
    print(f"  测试集:   {r.test_bars} 根 ({r.test_bars/r.total_bars:.0%})")

    # Performance
    print(f"\n  ── 收益表现 ──")
    ret_icon = "📈" if r.total_return_pct >= 0 else "📉"
    bh_icon = "📈" if r.buy_hold_return_pct >= 0 else "📉"
    ex_icon = "✅" if r.excess_return_pct > 0 else "❌"

    print(f"  初始资金:     ${r.initial_balance:>12,.2f}")
    print(f"  最终资金:     ${r.final_balance:>12,.2f}")
    print(f"  策略收益:     {ret_icon} {r.total_return_pct:>+.2f}%")
    print(f"  买入持有:     {bh_icon} {r.buy_hold_return_pct:>+.2f}%")
    print(f"  超额收益:     {ex_icon} {r.excess_return_pct:>+.2f}%")

    # Risk
    print(f"\n  ── 风险指标 ──")
    print(f"  最大回撤:     {r.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio: {r.sharpe_ratio:.2f}")

    # Trade stats
    print(f"\n  ── 交易统计 ──")
    print(f"  总交易次数:   {r.total_trades}")
    print(f"  盈利交易:     {r.winning_trades} ({r.win_rate:.1f}%)")
    print(f"  亏损交易:     {r.losing_trades} ({100 - r.win_rate:.1f}%)")
    if r.total_trades > 0:
        print(f"  平均盈利:     +{r.avg_win_pct:.2f}%")
        print(f"  平均亏损:     {r.avg_loss_pct:.2f}%")
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "∞"
        print(f"  盈亏比:       {pf_str}")

    # Action distribution
    if r.actions:
        holds = r.actions.count(0)
        buys = r.actions.count(1)
        sells = r.actions.count(2)
        total_a = len(r.actions)
        print(f"\n  ── 动作分布 ──")
        print(f"  Hold:  {holds:>5} ({holds/total_a:.1%})")
        print(f"  Buy:   {buys:>5} ({buys/total_a:.1%})")
        print(f"  Sell:  {sells:>5} ({sells/total_a:.1%})")

    # Trade details
    if r.trades:
        print(f"\n  ── 交易明细 (最近 10 笔) ──")
        print(f"  {'#':<4} {'买入价':>10} {'卖出价':>10} {'收益率':>8} {'盈亏金额':>12}")
        print(f"  {'─' * 48}")
        for i, t in enumerate(r.trades[-10:], 1):
            icon = "🟢" if t.pnl > 0 else "🔴"
            print(
                f"  {icon}{i:<3} {t.entry_price:>10,.2f} {t.exit_price:>10,.2f} "
                f"{t.pnl_pct:>+7.2f}% ${t.pnl:>11,.2f}"
            )

    # Visual equity curve (ASCII mini chart)
    if len(r.equity_curve) > 10:
        print(f"\n  ── 权益曲线 (ASCII) ──")
        _print_ascii_chart(r.equity_curve, "策略", width=50)
        # Normalize price curve to same scale for comparison
        if r.price_curve:
            bh_curve = [initial * (p / r.price_curve[0]) for initial, p in
                        zip([r.initial_balance] * len(r.price_curve), r.price_curve)]
            _print_ascii_chart(bh_curve, "持有", width=50)

    # Verdict
    print(f"\n  ── 综合评价 ──")
    if r.total_trades == 0:
        print("  ⚠️  测试期间无任何交易，模型可能过于保守或数据不足")
    elif r.excess_return_pct > 0 and r.sharpe_ratio > 0.5:
        print("  ✅ 模型在测试集上跑赢买入持有，且 Sharpe 合理，有一定有效性")
    elif r.excess_return_pct > 0:
        print("  🟡 模型跑赢买入持有，但 Sharpe 偏低，收益波动较大")
    elif r.win_rate > 50:
        print("  🟡 模型胜率尚可，但总收益未跑赢买入持有")
    else:
        print("  ❌ 模型未跑赢买入持有策略，建议重新训练或调整参数")

    print(f"\n  ⚠️  回测结果不代表未来表现，请谨慎参考")
    print(f"{'═' * 72}\n")


def _print_ascii_chart(values: list[float], label: str, width: int = 50) -> None:
    """Print a simple ASCII sparkline chart."""
    if not values or len(values) < 2:
        return

    # Downsample to width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo if hi != lo else 1

    height = 6
    chart = [[" " for _ in range(len(sampled))] for _ in range(height)]

    for col, val in enumerate(sampled):
        row = int((val - lo) / span * (height - 1))
        row = min(max(row, 0), height - 1)
        chart[row][col] = "█"

    for row in reversed(range(height)):
        line = "".join(chart[row])
        if row == height - 1:
            print(f"  {label} ↑ {line} ${hi:,.0f}")
        elif row == 0:
            print(f"       ↓ {line} ${lo:,.0f}")
        else:
            print(f"         {line}")


# ─── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest ML trading model")
    parser.add_argument("--symbol", type=str, default="TSLA.US", help="Symbol to backtest")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "rl"])
    parser.add_argument("--model-name", type=str, default="xgb_model", help="Model name")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN", "A2C"])
    parser.add_argument("--klines", type=int, default=500, help="Number of daily candles")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio")

    args = parser.parse_args()

    from logger import setup_logging
    setup_logging()

    # Normalize symbol
    symbol = args.symbol.upper()
    if "." not in symbol:
        symbol += ".US"

    # Fetch data
    log.info("Fetching %d daily candles for %s ...", args.klines, symbol)
    from longport.openapi import Config, QuoteContext, Period, AdjustType
    from config import load_config

    cfg = load_config()
    lb_config = Config(
        app_key=cfg.credentials.app_key,
        app_secret=cfg.credentials.app_secret,
        access_token=cfg.credentials.access_token,
    )
    quote_ctx = QuoteContext(lb_config)
    candles = quote_ctx.candlesticks(symbol, Period.Day, args.klines, AdjustType.NoAdjust)

    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    volumes = [float(c.volume) for c in candles]

    log.info("Got %d candles", len(closes))

    # Load model
    if args.model_type == "xgboost":
        from ml.xgb_model import XGBModelManager
        model = XGBModelManager()
        if not model.load(args.model_name):
            log.error("Failed to load XGBoost model: %s", args.model_name)
            return
    else:
        from ml.rl_agent import RLAgentManager
        model = RLAgentManager(algo=args.algo)
        if not model.load(args.model_name):
            log.error("Failed to load RL model: %s (%s)", args.model_name, args.algo)
            return

    # Run backtest
    result = run_backtest(
        closes, highs, lows, volumes,
        model_type=args.model_type,
        model=model,
        symbol=symbol,
        test_ratio=args.test_ratio,
        boll_period=cfg.strategy.boll_period,
        boll_std=cfg.strategy.boll_std_dev,
        rsi_period=cfg.strategy.rsi_period,
        ema_period=cfg.strategy.trend_ema_period,
        xgb_buy_threshold=cfg.ml.xgb_buy_threshold,
        xgb_sell_threshold=cfg.ml.xgb_sell_threshold,
    )

    print_backtest_report(result)


if __name__ == "__main__":
    main()
