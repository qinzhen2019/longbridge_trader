from __future__ import annotations

import os
import sys
import subprocess
import platform
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from longport.openapi import (
    Config,
    QuoteContext,
    TradeContext,
    Period,
    AdjustType,
    TopicType,
)

from config import load_config, TradingConfig
from indicators import compute_bollinger, compute_rsi, compute_ma, BollingerBands
from logger import setup_logging, get_logger
from order_executor import OrderExecutor

log = get_logger("dashboard")

KLINE_COUNT = 100
MODEL_DIR = Path(__file__).resolve().parent / "models"

TIMEFRAMES = [
    ("日线", Period.Day),
    ("5分钟", Period.Min_5),
    ("1分钟", Period.Min_1),
]

# ─── ML dependency names for display ──────────────────────────
_ML_PACKAGES = {
    "xgboost":           "XGBoost (梯度提升树)",
    "stable_baselines3": "Stable-Baselines3 (强化学习)",
    "sklearn":           "scikit-learn (机器学习工具库)",
    "numpy":             "NumPy (数值计算)",
}


# ╭───────────────────────────────────────────────────────────╮
# │  Utility helpers                                          │
# ╰───────────────────────────────────────────────────────────╯

def _build_connections(cfg: TradingConfig):
    lb_config = Config(
        app_key=cfg.credentials.app_key,
        app_secret=cfg.credentials.app_secret,
        access_token=cfg.credentials.access_token,
    )
    quote_ctx = QuoteContext(lb_config)
    trade_ctx = TradeContext(lb_config)
    trade_ctx.subscribe([TopicType.Private])
    return quote_ctx, trade_ctx


def _trend_label(closes: list[float]) -> str:
    if len(closes) < 5:
        return "数据不足"
    recent = closes[-5:]
    diff = recent[-1] - recent[0]
    pct = diff / recent[0] * 100
    if pct > 0.3:
        return f"↑ 偏多 (+{pct:.2f}%)"
    elif pct < -0.3:
        return f"↓ 偏空 ({pct:.2f}%)"
    else:
        return f"→ 震荡 ({pct:+.2f}%)"


def _format_price(val: float) -> str:
    return f"{val:,.2f}"


def _strategy_label(cfg: TradingConfig) -> str:
    """Return a human-friendly label for the active strategy."""
    st = cfg.ml.strategy_type.lower()
    if st == "xgboost":
        return f"🤖 XGBoost (buy≥{cfg.ml.xgb_buy_threshold:.2f}, sell≤{cfg.ml.xgb_sell_threshold:.2f})"
    elif st == "rl":
        return f"🧠 强化学习 {cfg.ml.rl_algo} (RL)"
    else:
        return "📊 布林带+RSI (规则型)"


def _check_ml_deps() -> dict[str, bool]:
    """Probe availability of ML-related packages."""
    result = {}
    for pkg in _ML_PACKAGES:
        try:
            __import__(pkg)
            result[pkg] = True
        except ImportError:
            result[pkg] = False
    return result


def _get_model_info(cfg: TradingConfig) -> dict:
    """Return metadata about the ML model file on disk."""
    st = cfg.ml.strategy_type.lower()
    if st == "xgboost":
        path = MODEL_DIR / f"{cfg.ml.model_name}.json"
    elif st == "rl":
        path = MODEL_DIR / f"{cfg.ml.model_name}_{cfg.ml.rl_algo}.zip"
    else:
        return {"exists": False, "path": None, "strategy": st}

    info: dict = {"path": str(path), "strategy": st}
    if path.exists():
        stat = path.stat()
        info["exists"] = True
        info["size_kb"] = stat.st_size / 1024
        info["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    else:
        info["exists"] = False
    return info


def _model_status_line(info: dict) -> str:
    """One-liner model status for menus."""
    if info.get("path") is None:
        return "规则型策略，无需模型文件"
    if info["exists"]:
        return f"✅ {Path(info['path']).name} ({info['size_kb']:.0f}KB, {info['modified']})"
    return f"❌ 模型不存在: {Path(info['path']).name}"


# ╭───────────────────────────────────────────────────────────╮
# │  1. 分析股票标的                                          │
# ╰───────────────────────────────────────────────────────────╯

def analyze_symbol(quote_ctx: QuoteContext, symbol: str, cfg: TradingConfig) -> None:
    print(f"\n{'=' * 60}")
    print(f"  股票分析: {symbol}")
    print(f"{'=' * 60}")

    day_closes: list[float] = []
    day_highs: list[float] = []
    day_lows: list[float] = []
    day_volumes: list[float] = []

    for tf_name, period_enum in TIMEFRAMES:
        try:
            candles = quote_ctx.candlesticks(symbol, period_enum, KLINE_COUNT, AdjustType.NoAdjust)
            closes = [float(c.close) for c in candles]
        except Exception as e:
            print(f"\n  [{tf_name}] 获取K线失败: {e}")
            continue

        if not closes:
            print(f"\n  [{tf_name}] 无数据")
            continue

        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        volumes = [float(c.volume) for c in candles]
        latest = closes[-1]
        high_max = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        low_min = min(lows[-20:]) if len(lows) >= 20 else min(lows)

        if period_enum == Period.Day:
            day_closes, day_highs, day_lows, day_volumes = closes, highs, lows, volumes

        boll = compute_bollinger(closes, cfg.strategy.boll_period, cfg.strategy.boll_std_dev)
        rsi = compute_rsi(closes, cfg.strategy.rsi_period)
        trend = _trend_label(closes)

        print(f"\n  ┌─ {tf_name} (最近 {len(closes)} 根K线)")
        print(f"  │  最新价:  {_format_price(latest)}")
        print(f"  │  趋势:    {trend}")
        print(f"  │  近20根:  最高 {_format_price(high_max)} / 最低 {_format_price(low_min)}")

        if boll:
            print(f"  │")
            print(f"  │  布林带:")
            print(f"  │    上轨:  {_format_price(boll.upper)}")
            print(f"  │    中轨:  {_format_price(boll.middle)}")
            print(f"  │    下轨:  {_format_price(boll.lower)}")

        if rsi is not None:
            rsi_status = ""
            if rsi < cfg.strategy.rsi_oversold:
                rsi_status = " ← 超卖区"
            elif rsi > cfg.strategy.rsi_overbought:
                rsi_status = " ← 超买区"
            print(f"  │  RSI:     {rsi:.1f}{rsi_status}")

        print(f"  └{'─' * 40}")

    # ── ML 特征向量预览 ──
    _show_ml_features(cfg, day_closes, day_highs, day_lows, day_volumes)

    # ── 建议点位 ──
    _show_price_levels(quote_ctx, symbol, cfg, day_closes, day_highs, day_lows)


def _show_ml_features(
    cfg: TradingConfig,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
) -> None:
    """Show ML feature vector preview if using ML strategy."""
    st = cfg.ml.strategy_type.lower()
    if st not in ("xgboost", "rl"):
        return

    if not closes or len(closes) < 60:
        print(f"\n  ┌─ ML 特征向量 (数据不足，需至少 60 根K线)")
        print(f"  └{'─' * 40}")
        return

    try:
        from ml.feature_engineer import build_feature_vector, FEATURE_NAMES
    except ImportError:
        return

    features = build_feature_vector(
        closes,
        highs=highs or None,
        lows=lows or None,
        volumes=volumes or None,
        boll_period=cfg.strategy.boll_period,
        boll_std=cfg.strategy.boll_std_dev,
        rsi_period=cfg.strategy.rsi_period,
        ema_period=cfg.strategy.trend_ema_period,
    )

    if features is None:
        return

    label = "XGBoost" if st == "xgboost" else f"RL ({cfg.ml.rl_algo})"
    print(f"\n  ┌─ ML 特征向量 ({label})")

    # Format features in 2-column layout
    names = FEATURE_NAMES if len(FEATURE_NAMES) <= len(features) else FEATURE_NAMES[:len(features)]
    for i in range(0, len(names), 2):
        left = f"{names[i]}: {features[i]:>+.4f}"
        if i + 1 < len(names):
            right = f"{names[i+1]}: {features[i+1]:>+.4f}"
            print(f"  │  {left:<28} {right}")
        else:
            print(f"  │  {left}")

    print(f"  └{'─' * 40}")


def _show_price_levels(
    quote_ctx: QuoteContext,
    symbol: str,
    cfg: TradingConfig,
    day_closes: list[float],
    day_highs: list[float],
    day_lows: list[float],
) -> None:
    """Show suggested entry/exit levels based on daily Bollinger Bands."""
    print(f"\n{'─' * 60}")
    print("  建议点位 (基于日线布林带)")
    print(f"{'─' * 60}")

    # Use cached data if available, otherwise fetch
    if not day_closes:
        try:
            day_candles = quote_ctx.candlesticks(symbol, Period.Day, KLINE_COUNT, AdjustType.NoAdjust)
            day_closes = [float(c.close) for c in day_candles]
            day_lows = [float(c.low) for c in day_candles]
            day_highs = [float(c.high) for c in day_candles]
        except Exception:
            print("  无法获取日线数据，跳过点位建议")
            return

    if not day_closes:
        return

    day_boll = compute_bollinger(day_closes, cfg.strategy.boll_period, cfg.strategy.boll_std_dev)
    day_rsi = compute_rsi(day_closes, cfg.strategy.rsi_period)
    latest_price = day_closes[-1]

    support_1 = day_boll.lower if day_boll else None
    support_2 = min(day_lows[-10:]) if len(day_lows) >= 10 else min(day_lows) if day_lows else None

    resist_1 = day_boll.middle if day_boll else None
    resist_2 = day_boll.upper if day_boll else None
    resist_3 = max(day_highs[-10:]) if len(day_highs) >= 10 else max(day_highs) if day_highs else None

    print()
    print(f"  当前价格:  {_format_price(latest_price)}")
    if day_rsi is not None:
        print(f"  日线 RSI:  {day_rsi:.1f}")
    print()

    print("  ▼ 建议买入区间 (逢低分批)")
    if support_1 is not None:
        distance_1 = (latest_price - support_1) / latest_price * 100
        print(f"    一档:  {_format_price(support_1)}  (布林下轨, 距当前 {distance_1:.1f}%)")
    if support_2 is not None:
        distance_2 = (latest_price - support_2) / latest_price * 100
        print(f"    二档:  {_format_price(support_2)}  (近10日最低, 距当前 {distance_2:.1f}%)")

    print()
    print("  ▲ 建议卖出区间 (逢高分批)")
    if resist_1 is not None:
        distance_r1 = (resist_1 - latest_price) / latest_price * 100
        print(f"    一档:  {_format_price(resist_1)}  (布林中轨, 距当前 +{distance_r1:.1f}%)")
    if resist_2 is not None:
        distance_r2 = (resist_2 - latest_price) / latest_price * 100
        print(f"    二档:  {_format_price(resist_2)}  (布林上轨, 距当前 +{distance_r2:.1f}%)")
    if resist_3 is not None:
        distance_r3 = (resist_3 - latest_price) / latest_price * 100
        print(f"    三档:  {_format_price(resist_3)}  (近10日最高, 距当前 +{distance_r3:.1f}%)")

    print()
    if day_boll and day_rsi is not None:
        if latest_price <= day_boll.lower and day_rsi < cfg.strategy.rsi_oversold:
            print("  ★ 综合研判: 当前处于超卖区域，布林下轨附近，可考虑轻仓试探买入")
        elif latest_price >= day_boll.upper and day_rsi > cfg.strategy.rsi_overbought:
            print("  ★ 综合研判: 当前处于超买区域，布林上轨附近，注意风险，可考虑减仓")
        elif latest_price >= day_boll.middle:
            print("  ★ 综合研判: 价格在布林中轨上方，偏强运行，持仓为主")
        else:
            print("  ★ 综合研判: 价格在布林中轨下方，偏弱运行，观望或等待买入信号")


# ╭───────────────────────────────────────────────────────────╮
# │  2. 查看持仓                                              │
# ╰───────────────────────────────────────────────────────────╯

def show_positions(trade_ctx: TradeContext) -> None:
    print(f"\n{'=' * 60}")
    print("  股票持仓")
    print(f"{'=' * 60}")

    try:
        resp = trade_ctx.stock_positions()
        all_positions = []
        for channel in resp.channels:
            all_positions.extend(channel.positions)
    except Exception as e:
        print(f"  获取持仓失败: {e}")
        return

    if not all_positions:
        print("  (空仓，暂无持仓)")
        return

    print(f"\n  {'代码':<12} {'名称':<16} {'数量':>8} {'可用':>8} {'成本价':>10} {'币种':>6}")
    print(f"  {'─' * 62}")

    for pos in all_positions:
        print(
            f"  {pos.symbol:<12} {pos.symbol_name:<16} {pos.quantity:>8} "
            f"{pos.available_quantity:>8} {str(pos.cost_price):>10} {pos.currency:>6}"
        )

    print()


# ╭───────────────────────────────────────────────────────────╮
# │  3. 查看现金                                              │
# ╰───────────────────────────────────────────────────────────╯

def show_cash_balance(trade_ctx: TradeContext) -> None:
    print(f"\n{'=' * 60}")
    print("  现金余额")
    print(f"{'=' * 60}")

    try:
        balances = trade_ctx.account_balance()
    except Exception as e:
        print(f"  获取余额失败: {e}")
        return

    for bal in balances:
        print(f"\n  账户币种: {bal.currency}")
        print(f"  总现金:         {bal.total_cash}")
        print(f"  净资产:         {bal.net_assets}")
        print(f"  融资上限:       {bal.max_finance_amount}")
        print(f"  剩余融资额度:   {bal.remaining_finance_amount}")
        print(f"  风险等级:       {bal.risk_level}")
        print(f"  购买力:         {bal.buy_power}")

        if bal.cash_infos:
            print(f"\n  {'币种':>6}  {'可用':>14}  {'可取':>14}  {'冻结':>14}  {'待交收':>14}")
            print(f"  {'─' * 66}")
            for ci in bal.cash_infos:
                print(
                    f"  {ci.currency:>6}  {ci.available_cash:>14}  {ci.withdraw_cash:>14}  "
                    f"{ci.frozen_cash:>14}  {ci.settling_cash:>14}"
                )

    print()


# ╭───────────────────────────────────────────────────────────╮
# │  4. 扫描关注清单                                          │
# ╰───────────────────────────────────────────────────────────╯

def _compute_buy_score(closes: list[float], cfg: TradingConfig) -> tuple[float, dict]:
    """
    综合评分 0~100，越高越值得买入。
    RSI 权重 40% — 越低越超卖，得分越高
    布林带位置 30% — 越接近/低于下轨，得分越高
    MA 趋势 20% — MA5 上穿 MA20 趋势向好加分
    近期回撤 10% — 从近期高点回撤越大，反弹空间越大
    """
    details: dict = {}
    total = 0.0

    rsi = compute_rsi(closes, cfg.strategy.rsi_period)
    if rsi is not None:
        rsi_score = max(0.0, min(40.0, (70.0 - rsi) / 70.0 * 40.0))
        total += rsi_score
        details["rsi"] = rsi
        details["rsi_score"] = rsi_score
    else:
        details["rsi"] = None
        details["rsi_score"] = 0.0

    boll = compute_bollinger(closes, cfg.strategy.boll_period, cfg.strategy.boll_std_dev)
    if boll and boll.middle != boll.lower:
        price = closes[-1]
        band_width = boll.middle - boll.lower
        distance_below_mid = boll.middle - price
        boll_score = max(0.0, min(30.0, distance_below_mid / band_width * 30.0))
        total += boll_score
        details["boll_score"] = boll_score
        details["boll_pos"] = "下轨下方" if price <= boll.lower else ("下轨~中轨" if price < boll.middle else "中轨上方")
    else:
        details["boll_score"] = 0.0
        details["boll_pos"] = "N/A"

    ma5 = compute_ma(closes, 5)
    ma20 = compute_ma(closes, 20)
    if ma5 is not None and ma20 is not None:
        if ma5 >= ma20:
            ma_score = 20.0
        else:
            gap_pct = (ma20 - ma5) / ma20 * 100
            ma_score = max(0.0, 20.0 - gap_pct * 4)
        total += ma_score
        details["ma5"] = ma5
        details["ma20"] = ma20
        details["ma_score"] = ma_score
    else:
        details["ma5"] = None
        details["ma20"] = None
        details["ma_score"] = 0.0

    if len(closes) >= 20:
        recent_high = max(closes[-20:])
        drawdown = (recent_high - closes[-1]) / recent_high * 100
        dd_score = min(10.0, drawdown)
        total += dd_score
        details["drawdown_pct"] = drawdown
        details["dd_score"] = dd_score
    else:
        details["drawdown_pct"] = 0.0
        details["dd_score"] = 0.0

    details["total"] = total
    return total, details


def scan_watchlist(quote_ctx: QuoteContext, cfg: TradingConfig) -> None:
    print(f"\n{'=' * 60}")
    print("  美股关注清单扫描")
    print(f"{'=' * 60}")
    print("  正在获取关注清单 ...")

    try:
        groups = quote_ctx.watchlist()
    except Exception as e:
        print(f"  获取关注清单失败: {e}")
        return

    us_symbols: list[str] = []
    for group in groups:
        for sec in group.securities:
            if sec.symbol.endswith(".US") and sec.symbol not in us_symbols:
                us_symbols.append(sec.symbol)

    if not us_symbols:
        print("  关注清单中没有美股标的")
        return

    print(f"  找到 {len(us_symbols)} 只美股: {', '.join(us_symbols[:10])}{'...' if len(us_symbols) > 10 else ''}")
    print("  正在逐个分析 ...\n")

    results: list[tuple[str, float, dict, float]] = []

    for i, symbol in enumerate(us_symbols):
        try:
            candles = quote_ctx.candlesticks(symbol, Period.Day, KLINE_COUNT, AdjustType.NoAdjust)
            closes = [float(c.close) for c in candles]
        except Exception as e:
            print(f"  [{i+1}/{len(us_symbols)}] {symbol} - 获取失败: {e}")
            continue

        if len(closes) < 20:
            print(f"  [{i+1}/{len(us_symbols)}] {symbol} - 数据不足 ({len(closes)} 根)")
            continue

        score, details = _compute_buy_score(closes, cfg)
        latest_price = closes[-1]
        results.append((symbol, score, details, latest_price))
        print(f"  [{i+1}/{len(us_symbols)}] {symbol:<10} 价格={latest_price:>10,.2f}  评分={score:>5.1f}")

    if not results:
        print("\n  无有效结果")
        return

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'═' * 70}")
    print("  排名  代码         价格        评分   RSI    布林位置     MA趋势")
    print(f"{'─' * 70}")

    for rank, (symbol, score, details, price) in enumerate(results, 1):
        rsi_str = f"{details['rsi']:.1f}" if details['rsi'] is not None else "N/A"
        boll_pos = details['boll_pos']
        ma_str = ""
        if details["ma5"] is not None and details["ma20"] is not None:
            if details["ma5"] >= details["ma20"]:
                ma_str = "多头↑"
            else:
                ma_str = "空头↓"

        marker = ""
        if rank == 1:
            marker = " ★ TOP"
        elif rank <= 3:
            marker = " ☆"

        print(
            f"  {rank:>3}.  {symbol:<12} {price:>10,.2f}  {score:>5.1f}  "
            f"{rsi_str:>5}  {boll_pos:<10}  {ma_str:<6}{marker}"
        )

    print(f"{'═' * 70}")

    top = results[0]
    print(f"\n  ★ 最推荐买入: {top[0]}")
    print(f"    综合评分: {top[1]:.1f} / 100")
    d = top[2]
    print(f"    RSI:      {d['rsi']:.1f}  (得分 {d['rsi_score']:.1f}/40)" if d['rsi'] is not None else "    RSI:      N/A")
    print(f"    布林位置: {d['boll_pos']}  (得分 {d['boll_score']:.1f}/30)")
    print(f"    MA趋势:   MA5={'%.2f' % d['ma5'] if d['ma5'] else 'N/A'} / MA20={'%.2f' % d['ma20'] if d['ma20'] else 'N/A'}  (得分 {d['ma_score']:.1f}/20)")
    print(f"    近期回撤: {d['drawdown_pct']:.1f}%  (得分 {d['dd_score']:.1f}/10)")

    print(f"\n  是否查看该股票的详细分析?")
    yn = input("  (y/n): ").strip().lower()
    if yn == "y":
        analyze_symbol(quote_ctx, top[0], cfg)


# ╭───────────────────────────────────────────────────────────╮
# │  5. 手动交易                                              │
# ╰───────────────────────────────────────────────────────────╯

def manual_trade_menu(executor: OrderExecutor) -> None:
    print(f"\n{'=' * 60}")
    print("  手动交易")
    print(f"{'=' * 60}")
    
    symbol = input("  请输入股票代码 (如 TSLA.US / 700.HK): ").strip().upper()
    if not symbol:
        print("  代码不能为空")
        return
    if "." not in symbol:
        symbol += ".US"

    side_input = input("  买入(B) 还是 卖出(S)?: ").strip().upper()
    if side_input not in ("B", "S"):
        print("  无效输入，已取消")
        return
        
    try:
        qty_input = input("  请输入交易数量 (整数): ").strip()
        quantity = int(qty_input)
        if quantity <= 0:
            print("  数量必须大于 0")
            return
            
        price_input = input("  请输入限价价格 (留空为市价单): ").strip()
        price = Decimal(price_input) if price_input else None
    except ValueError:
        print("  输入格式错误，已取消")
        return

    side_str = "买入" if side_input == "B" else "卖出"
    price_str = f"限价 {price}" if price is not None else "市价"
    print(f"\n  请确认: {side_str} {symbol} {quantity} 股 ({price_str})")
    
    confirm = input("  确认下单? (y/n): ").strip().lower()
    if confirm == "y":
        if side_input == "B":
            order_id = executor.submit_buy(symbol, quantity, price)
        else:
            order_id = executor.submit_sell(symbol, quantity, price)
            
        if order_id:
            print(f"  ✅ 订单提交成功! 订单号: {order_id}")
        else:
            print("  ❌ 订单提交失败，请检查日志")
    else:
        print("  已取消下单")


# ╭───────────────────────────────────────────────────────────╮
# │  6. 查看并撤销订单                                        │
# ╰───────────────────────────────────────────────────────────╯

def cancel_orders_menu(executor: OrderExecutor) -> None:
    print(f"\n{'=' * 60}")
    print("  撤销未成交订单")
    print(f"{'=' * 60}")
    
    orders = executor.get_today_orders()
    active_orders = [o for o in orders if o.status in ("NewStatus", "WaitToNew", "PartialFilled", "PendingReplace")]
    
    if not active_orders:
        print("  当前没有待成交的订单")
        return
        
    print(f"  {'序号':<4} {'代码':<10} {'方向':<6} {'类型':<6} {'状态':<15} {'价格':<10} {'数量(已成交/总)'}")
    print(f"  {'─' * 70}")
    
    for i, o in enumerate(active_orders, 1):
        side_str = "买入" if "Buy" in str(o.side) else "卖出"
        print(f"  [{i:<2}] {o.symbol:<10} {side_str:<6} {str(o.order_type):<6} {str(o.status):<15} {str(o.price):<10} {o.executed_quantity}/{o.quantity}")
        
    print("\n  请选择:")
    print("  0. 返回上一级")
    print("  A. 一键撤销所有订单")
    print("  输入序号撤销单笔订单")
    
    choice = input("\n  您的选择: ").strip().upper()
    if choice == "0" or choice == "":
        return
    elif choice == "A":
        print("  正在撤销所有待成交订单...")
        cancelled = executor.cancel_all_pending_orders()
        print(f"  ✅ 成功提交了 {cancelled} 笔撤单请求")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(active_orders):
                target_order = active_orders[idx]
                if executor._trade_ctx:
                    executor._trade_ctx.cancel_order(target_order.order_id)
                    print(f"  ✅ 已提交撤单请求: {target_order.order_id}")
                else:
                    print("  ❌ TradeContext未连接")
            else:
                print("  无效的序号")
        except ValueError:
            print("  无效输入")


# ╭───────────────────────────────────────────────────────────╮
# │  7. 启动自动交易引擎                                      │
# ╰───────────────────────────────────────────────────────────╯

def _launch_trading_engine(cfg: TradingConfig) -> None:
    """Confirm and launch the auto-trading engine."""
    model_info = _get_model_info(cfg)

    print("\n  即将启动自动交易引擎...")
    print(f"  模式:     {'模拟盘' if cfg.paper_trading else '实盘'}")
    print(f"  标的:     {cfg.watch_symbols}")
    print(f"  策略引擎: {_strategy_label(cfg)}")

    st = cfg.ml.strategy_type.lower()
    if st in ("xgboost", "rl"):
        print(f"  模型状态: {_model_status_line(model_info)}")
        if st == "xgboost":
            print(f"  买入阈值: {cfg.ml.xgb_buy_threshold:.2f} | 卖出阈值: {cfg.ml.xgb_sell_threshold:.2f}")
        elif st == "rl":
            print(f"  RL 算法:  {cfg.ml.rl_algo}")

        if not model_info.get("exists", True):
            print("\n  ⚠️  模型文件不存在！建议先在 [8. ML 策略管理] 中训练模型。")
            print("  ⚠️  引擎将以 HOLD 模式运行，不会发出任何交易信号。")

    confirm = input("\n  确认启动? (y/n): ").strip().lower()
    if confirm != "y":
        print("  已取消")
        return

    from main import TradingEngine
    engine = TradingEngine(cfg)

    print("  =======================================================")
    print("  💡 提示: 自动交易引擎正在前台运行，想退回菜单请按 Ctrl + C")
    print("  =======================================================\n")

    caffeinate_proc = None
    if platform.system() == "Darwin":
        print("  ⚡ 已启动 macOS 防休眠 (caffeinate)")
        caffeinate_proc = subprocess.Popen(
            ["caffeinate", "-i", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n  捕捉到退出信号，已停止自动交易，返回主面板。")
    finally:
        if caffeinate_proc is not None:
            caffeinate_proc.terminate()
            caffeinate_proc.wait()
            print("  🛑 已关闭防休眠进程")


# ╭───────────────────────────────────────────────────────────╮
# │  8. ML 策略管理                                           │
# ╰───────────────────────────────────────────────────────────╯

def ml_management_menu(cfg: TradingConfig) -> None:
    while True:
        print(f"\n{'=' * 60}")
        print(f"  ML 策略管理  |  当前: {_strategy_label(cfg)}")
        print(f"{'=' * 60}")
        print(f"  1. 查看当前策略配置")
        print(f"  2. 查看模型状态")
        print(f"  3. 检测 ML 依赖库")
        print(f"  4. 训练 XGBoost 模型")
        print(f"  5. 训练 RL 模型")
        print(f"  0. 返回主菜单")

        choice = input("\n  请选择 [0-5]: ").strip()

        if choice == "1":
            _show_strategy_config(cfg)
        elif choice == "2":
            _show_model_status(cfg)
        elif choice == "3":
            _show_ml_deps()
        elif choice == "4":
            _train_xgb_interactive(cfg)
        elif choice == "5":
            _train_rl_interactive(cfg)
        elif choice == "0":
            break
        else:
            print("  无效选择")


def _show_strategy_config(cfg: TradingConfig) -> None:
    print(f"\n{'─' * 50}")
    print("  当前策略配置")
    print(f"{'─' * 50}")
    print(f"  策略类型:     {cfg.ml.strategy_type}")
    print(f"  策略显示:     {_strategy_label(cfg)}")
    print()

    st = cfg.ml.strategy_type.lower()
    if st == "xgboost":
        print("  ── XGBoost 参数 ──")
        print(f"  模型名称:     {cfg.ml.model_name}")
        print(f"  买入阈值:     {cfg.ml.xgb_buy_threshold:.2f}  (prob_up ≥ 此值才买入)")
        print(f"  卖出阈值:     {cfg.ml.xgb_sell_threshold:.2f}  (prob_up ≤ 此值才卖出)")
    elif st == "rl":
        print("  ── RL 参数 ──")
        print(f"  RL 算法:      {cfg.ml.rl_algo}")
        print(f"  模型名称:     {cfg.ml.model_name}")
    else:
        print("  使用规则型策略，无 ML 参数")

    print()
    print("  ── 底层技术指标参数 ──")
    print(f"  布林带:       周期={cfg.strategy.boll_period}, 标准差={cfg.strategy.boll_std_dev}")
    print(f"  RSI:          周期={cfg.strategy.rsi_period}, 超卖={cfg.strategy.rsi_oversold}, 超买={cfg.strategy.rsi_overbought}")
    print(f"  EMA 趋势:     周期={cfg.strategy.trend_ema_period}")
    print()
    print("  ── ML 特征清单 (15维) ──")
    try:
        from ml.feature_engineer import FEATURE_NAMES
        for i, name in enumerate(FEATURE_NAMES, 1):
            print(f"    {i:>2}. {name}")
    except ImportError:
        print("    (无法加载特征工程模块)")

    print(f"\n  💡 修改策略请编辑 .env 文件中的 STRATEGY_TYPE 变量")
    print(f"     可选值: bollinger_rsi | xgboost | rl")


def _show_model_status(cfg: TradingConfig) -> None:
    print(f"\n{'─' * 50}")
    print("  模型状态")
    print(f"{'─' * 50}")

    model_info = _get_model_info(cfg)

    # XGBoost model
    xgb_path = MODEL_DIR / f"{cfg.ml.model_name}.json"
    if xgb_path.exists():
        stat = xgb_path.stat()
        size_kb = stat.st_size / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  XGBoost:  ✅ {xgb_path.name}  ({size_kb:.0f}KB, {mtime})")
    else:
        print(f"  XGBoost:  ❌ {xgb_path.name} 不存在")

    # RL model (check common algo variants)
    for algo in ["PPO", "DQN", "A2C"]:
        rl_path = MODEL_DIR / f"{cfg.ml.model_name}_{algo}.zip"
        if rl_path.exists():
            stat = rl_path.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  RL ({algo}): ✅ {rl_path.name}  ({size_kb:.0f}KB, {mtime})")
        else:
            print(f"  RL ({algo}): ❌ {rl_path.name} 不存在")

    st = cfg.ml.strategy_type.lower()
    if st in ("xgboost", "rl"):
        active_status = _model_status_line(model_info)
        print(f"\n  当前激活: {active_status}")
    else:
        print(f"\n  当前使用规则型策略，无需模型文件")

    print(f"\n  模型目录: {MODEL_DIR}")


def _show_ml_deps() -> None:
    print(f"\n{'─' * 50}")
    print("  ML 依赖库检测")
    print(f"{'─' * 50}")

    deps = _check_ml_deps()
    all_ok = True
    for pkg, desc in _ML_PACKAGES.items():
        status = "✅" if deps[pkg] else "❌"
        if not deps[pkg]:
            all_ok = False
        # Try to get version if available
        ver = ""
        if deps[pkg]:
            try:
                mod = __import__(pkg)
                ver = f" v{mod.__version__}" if hasattr(mod, "__version__") else ""
            except Exception:
                pass
        print(f"  {status}  {desc:<36} ({pkg}{ver})")

    if all_ok:
        print("\n  🎉 所有 ML 依赖已就绪!")
    else:
        print("\n  ⚠️  部分依赖缺失，请安装:")
        if not deps.get("xgboost"):
            print("      pip install xgboost")
        if not deps.get("stable_baselines3"):
            print("      pip install stable-baselines3")
        if not deps.get("sklearn"):
            print("      pip install scikit-learn")

    # Check libomp on macOS
    if platform.system() == "Darwin":
        libomp_path = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
        if libomp_path.exists():
            print(f"\n  ✅  macOS OpenMP runtime (libomp) 已安装")
        else:
            print(f"\n  ❌  macOS OpenMP runtime (libomp) 未安装")
            print(f"      brew install libomp")


def _train_xgb_interactive(cfg: TradingConfig) -> None:
    deps = _check_ml_deps()
    if not deps.get("xgboost") or not deps.get("sklearn"):
        print("\n  ❌ 缺少依赖: 请先安装 xgboost 和 scikit-learn")
        print("     pip install xgboost scikit-learn")
        return

    print(f"\n{'─' * 50}")
    print("  训练 XGBoost 模型")
    print(f"{'─' * 50}")

    symbols_default = ",".join(cfg.watch_symbols) if cfg.watch_symbols else "TSLA.US,AAPL.US"
    symbols_input = input(f"  训练标的 [{symbols_default}]: ").strip()
    symbols = symbols_input if symbols_input else symbols_default

    klines_input = input("  日线数量 [500]: ").strip()
    klines = klines_input if klines_input else "500"

    model_name = input(f"  模型名称 [{cfg.ml.model_name}]: ").strip()
    model_name = model_name if model_name else cfg.ml.model_name

    cmd = [
        sys.executable, "ml/train_xgb.py",
        "--symbols", symbols,
        "--klines", klines,
        "--model-name", model_name,
    ]

    print(f"\n  即将执行: {' '.join(cmd)}")
    confirm = input("  确认开始训练? (y/n): ").strip().lower()
    if confirm != "y":
        print("  已取消")
        return

    print("\n  ⏳ 训练中... (请稍候)\n")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=False,
        )
        if result.returncode == 0:
            print("\n  ✅ 训练完成!")
        else:
            print(f"\n  ❌ 训练失败 (exit code: {result.returncode})")
    except Exception as e:
        print(f"\n  ❌ 训练失败: {e}")


def _train_rl_interactive(cfg: TradingConfig) -> None:
    deps = _check_ml_deps()
    if not deps.get("stable_baselines3"):
        print("\n  ❌ 缺少依赖: 请先安装 stable-baselines3")
        print("     pip install stable-baselines3")
        return

    print(f"\n{'─' * 50}")
    print("  训练 RL 模型")
    print(f"{'─' * 50}")

    symbols_default = ",".join(cfg.watch_symbols) if cfg.watch_symbols else "TSLA.US,AAPL.US"
    symbols_input = input(f"  训练标的 [{symbols_default}]: ").strip()
    symbols = symbols_input if symbols_input else symbols_default

    algo_input = input(f"  RL 算法 (PPO/DQN/A2C) [{cfg.ml.rl_algo}]: ").strip().upper()
    algo = algo_input if algo_input in ("PPO", "DQN", "A2C") else cfg.ml.rl_algo

    steps_input = input("  训练步数 [50000]: ").strip()
    steps = steps_input if steps_input else "50000"

    model_name = input(f"  模型名称 [{cfg.ml.model_name}]: ").strip()
    model_name = model_name if model_name else cfg.ml.model_name

    cmd = [
        sys.executable, "ml/train_rl.py",
        "--symbols", symbols,
        "--algo", algo,
        "--timesteps", steps,
        "--model-name", model_name,
    ]

    print(f"\n  即将执行: {' '.join(cmd)}")
    confirm = input("  确认开始训练? (y/n): ").strip().lower()
    if confirm != "y":
        print("  已取消")
        return

    print("\n  ⏳ 训练中... (请稍候, RL 训练可能需要较长时间)\n")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=False,
        )
        if result.returncode == 0:
            print("\n  ✅ 训练完成!")
        else:
            print(f"\n  ❌ 训练失败 (exit code: {result.returncode})")
    except Exception as e:
        print(f"\n  ❌ 训练失败: {e}")


# ╭───────────────────────────────────────────────────────────╮
# │  9. ML 预测关注清单                                       │
# ╰───────────────────────────────────────────────────────────╯

def ml_predict_watchlist(quote_ctx: QuoteContext, cfg: TradingConfig) -> None:
    """Use the trained ML model to predict 5-day up/down probability for watchlist stocks."""
    print(f"\n{'=' * 72}")
    print("  🤖 ML 模型预测 — 关注清单未来5日涨跌概率")
    print(f"{'=' * 72}")

    # ── Check model availability ──
    model_info = _get_model_info(cfg)
    st = cfg.ml.strategy_type.lower()

    if st not in ("xgboost",):
        print(f"\n  ⚠️  当前仅支持 XGBoost 模型预测 (当前策略: {cfg.ml.strategy_type})")
        print("     请在 .env 中设置 STRATEGY_TYPE=xgboost")
        return

    if not model_info.get("exists"):
        print(f"\n  ❌ 模型文件不存在: {model_info.get('path', 'N/A')}")
        print("     请先在 [8. ML 策略管理] 中训练模型")
        return

    # ── Load model ──
    print("  正在加载模型 ...")
    try:
        from ml.xgb_model import XGBModelManager
        from ml.feature_engineer import build_feature_vector, FEATURE_NAMES
        mgr = XGBModelManager()
        if not mgr.load(cfg.ml.model_name):
            print("  ❌ 模型加载失败")
            return
    except ImportError as e:
        print(f"  ❌ 缺少依赖: {e}")
        return
    except Exception as e:
        print(f"  ❌ 模型加载异常: {e}")
        return

    print(f"  模型: {_model_status_line(model_info)}")

    # ── Fetch watchlist symbols ──
    print("  正在获取关注清单 ...")
    try:
        groups = quote_ctx.watchlist()
    except Exception as e:
        print(f"  获取关注清单失败: {e}")
        return

    us_symbols: list[str] = []
    for group in groups:
        for sec in group.securities:
            if sec.symbol.endswith(".US") and sec.symbol not in us_symbols:
                us_symbols.append(sec.symbol)

    if not us_symbols:
        print("  关注清单中没有美股标的")
        return

    print(f"  找到 {len(us_symbols)} 只美股，正在逐个预测 ...\n")

    # ── Predict each symbol ──
    predictions: list[tuple[str, float, float, float]] = []  # (symbol, price, prob_up, prob_down)

    for i, symbol in enumerate(us_symbols):
        try:
            candles = quote_ctx.candlesticks(symbol, Period.Day, KLINE_COUNT, AdjustType.NoAdjust)
            closes = [float(c.close) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]
            volumes = [float(c.volume) for c in candles]
        except Exception as e:
            print(f"  [{i+1}/{len(us_symbols)}] {symbol:<10} ❌ 获取数据失败: {e}")
            continue

        if len(closes) < 60:
            print(f"  [{i+1}/{len(us_symbols)}] {symbol:<10} ⚠️  数据不足 ({len(closes)} 根, 需至少60)")
            continue

        features = build_feature_vector(
            closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            boll_period=cfg.strategy.boll_period,
            boll_std=cfg.strategy.boll_std_dev,
            rsi_period=cfg.strategy.rsi_period,
            ema_period=cfg.strategy.trend_ema_period,
        )

        if features is None:
            print(f"  [{i+1}/{len(us_symbols)}] {symbol:<10} ⚠️  特征提取失败")
            continue

        prob_up = mgr.predict_proba(features)
        prob_down = 1.0 - prob_up
        latest_price = closes[-1]
        predictions.append((symbol, latest_price, prob_up, prob_down))

        # Progress indicator
        bar = "█" * int(prob_up * 20) + "░" * (20 - int(prob_up * 20))
        print(f"  [{i+1}/{len(us_symbols)}] {symbol:<10} 价格={latest_price:>10,.2f}  上涨={prob_up:.1%}  {bar}")

    if not predictions:
        print("\n  无有效预测结果")
        return

    # ── Sort by probability and display ──
    predictions.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'═' * 72}")
    print("  🏆 ML 预测排行榜 — 未来5个交易日涨跌概率 (XGBoost)")
    print(f"{'═' * 72}")
    print(f"  {'排名':<4}  {'代码':<10}  {'现价':>10}  {'上涨概率':>8}  {'下跌概率':>8}  {'信号':>6}  预测柱")
    print(f"  {'─' * 68}")

    for rank, (symbol, price, p_up, p_down) in enumerate(predictions, 1):
        # Signal based on thresholds
        if p_up >= cfg.ml.xgb_buy_threshold:
            signal = "🟢 买入"
        elif p_up <= cfg.ml.xgb_sell_threshold:
            signal = "🔴 卖出"
        else:
            signal = "⚪ 观望"

        # Visual bar
        bar_len = 20
        up_blocks = int(p_up * bar_len)
        bar = f"\033[32m{'█' * up_blocks}\033[31m{'█' * (bar_len - up_blocks)}\033[0m"

        # Rank markers
        marker = ""
        if rank == 1:
            marker = " ★"
        elif rank <= 3:
            marker = " ☆"

        print(
            f"  {rank:>3}.  {symbol:<10}  {price:>10,.2f}  "
            f"{p_up:>7.1%}   {p_down:>7.1%}   {signal}  {bar}{marker}"
        )

    print(f"  {'─' * 68}")

    # ── Summary ──
    buy_candidates = [(s, p, pu) for s, p, pu, _ in predictions if pu >= cfg.ml.xgb_buy_threshold]
    sell_candidates = [(s, p, pu) for s, p, pu, _ in predictions if pu <= cfg.ml.xgb_sell_threshold]
    hold_count = len(predictions) - len(buy_candidates) - len(sell_candidates)

    print(f"\n  📊 汇总: 共 {len(predictions)} 只股票")
    print(f"     🟢 买入信号: {len(buy_candidates)} 只 (prob_up ≥ {cfg.ml.xgb_buy_threshold:.0%})")
    print(f"     🔴 卖出信号: {len(sell_candidates)} 只 (prob_up ≤ {cfg.ml.xgb_sell_threshold:.0%})")
    print(f"     ⚪ 观望:     {hold_count} 只")

    if buy_candidates:
        print(f"\n  🟢 推荐买入:")
        for s, p, pu in buy_candidates:
            print(f"     {s:<10}  价格 {p:>10,.2f}  上涨概率 {pu:.1%}")

    if sell_candidates:
        print(f"\n  🔴 建议回避/卖出:")
        for s, p, pu in sell_candidates:
            print(f"     {s:<10}  价格 {p:>10,.2f}  上涨概率 {pu:.1%}")

    print(f"\n  ⚠️  提示: 预测基于历史数据训练的 XGBoost 模型，仅供参考，不构成投资建议")
    print(f"     模型训练 horizon=5 (未来5个交易日)")
    print(f"     买入阈值: {cfg.ml.xgb_buy_threshold:.0%} | 卖出阈值: {cfg.ml.xgb_sell_threshold:.0%}")


# ╭───────────────────────────────────────────────────────────╮
# │  Main Menu & Entry point                                  │
# ╰───────────────────────────────────────────────────────────╯

def main_menu(cfg: TradingConfig) -> str:
    strategy = _strategy_label(cfg)
    mode_str = "模拟盘" if cfg.paper_trading else "实盘"

    print(f"\n{'╔' + '═' * 52 + '╗'}")
    print(f"{'║'}{'长桥交易助手 - 交互式面板':^46}{'║'}")
    print(f"{'╠' + '═' * 52 + '╣'}")
    print(f"{'║'}  策略: {strategy:<42}{'║'}")
    print(f"{'║'}  模式: {mode_str:<42}{'║'}")
    print(f"{'╠' + '═' * 52 + '╣'}")
    print(f"{'║'}  1. 分析股票标的                                  {'║'}")
    print(f"{'║'}  2. 查看账户持仓                                  {'║'}")
    print(f"{'║'}  3. 查看现金余额                                  {'║'}")
    print(f"{'║'}  4. 扫描美股关注清单 (智能推荐)                   {'║'}")
    print(f"{'║'}  5. 手动下单交易                                  {'║'}")
    print(f"{'║'}  6. 查看并撤销订单                                {'║'}")
    print(f"{'║'}  7. 启动自动交易引擎 (按 Ctrl+C 可停止并返回)     {'║'}")
    print(f"{'║'}  8. ML 策略管理                                   {'║'}")
    print(f"{'║'}  9. ML 预测关注清单 (未来5日涨跌概率)             {'║'}")
    print(f"{'║'}  0. 退出                                          {'║'}")
    print(f"{'╚' + '═' * 52 + '╝'}")
    return input("\n  请选择 [0-9]: ").strip()


def main() -> None:
    setup_logging()

    print("\n  正在连接长桥 API ...")
    cfg = load_config()
    quote_ctx, trade_ctx = _build_connections(cfg)
    print("  连接成功!\n")

    while True:
        choice = main_menu(cfg)

        if choice == "1":
            symbol = input("  请输入股票代码 (如 TSLA.US / 700.HK): ").strip().upper()
            if not symbol:
                print("  代码不能为空")
                continue
            if "." not in symbol:
                symbol += ".US"
                print(f"  自动补全为: {symbol}")
            analyze_symbol(quote_ctx, symbol, cfg)

        elif choice == "2":
            show_positions(trade_ctx)

        elif choice == "3":
            show_cash_balance(trade_ctx)

        elif choice == "4":
            scan_watchlist(quote_ctx, cfg)

        elif choice == "5":
            executor = OrderExecutor(cfg)
            executor._trade_ctx = trade_ctx
            manual_trade_menu(executor)
            
        elif choice == "6":
            executor = OrderExecutor(cfg)
            executor._trade_ctx = trade_ctx
            cancel_orders_menu(executor)

        elif choice == "7":
            _launch_trading_engine(cfg)

        elif choice == "8":
            ml_management_menu(cfg)

        elif choice == "9":
            ml_predict_watchlist(quote_ctx, cfg)

        elif choice == "0":
            print("\n  再见!")
            sys.exit(0)

        else:
            print("  无效选择，请输入 0-9")


if __name__ == "__main__":
    main()
