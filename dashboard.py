from __future__ import annotations

import sys
import subprocess
import platform
from decimal import Decimal

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

TIMEFRAMES = [
    ("日线", Period.Day),
    ("5分钟", Period.Min_5),
    ("1分钟", Period.Min_1),
]


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


def analyze_symbol(quote_ctx: QuoteContext, symbol: str, cfg: TradingConfig) -> None:
    print(f"\n{'=' * 60}")
    print(f"  股票分析: {symbol}")
    print(f"{'=' * 60}")

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
        latest = closes[-1]
        high_max = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        low_min = min(lows[-20:]) if len(lows) >= 20 else min(lows)

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

    print(f"\n{'─' * 60}")
    print("  建议点位 (基于日线布林带)")
    print(f"{'─' * 60}")

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

    # 支撑位: 布林下轨, 近期最低价
    support_1 = day_boll.lower if day_boll else None
    support_2 = min(day_lows[-10:]) if len(day_lows) >= 10 else min(day_lows) if day_lows else None

    # 阻力位: 布林上轨, 布林中轨, 近期最高价
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
        # RSI 70→0 映射到 0→40
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
        # price 在下轨或以下 → 满分30; 在中轨 → 0; 在上轨 → 负分截断到0
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
        # MA5 > MA20 → 多头排列加分; MA5 < MA20 但差距在收窄 → 部分加分
        if ma5 >= ma20:
            ma_score = 20.0
        else:
            gap_pct = (ma20 - ma5) / ma20 * 100
            ma_score = max(0.0, 20.0 - gap_pct * 4)  # 差距5%以上→0分
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
        # 回撤 0%→0分, 10%+→满分10
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


def main_menu() -> str:
    print(f"\n{'╔' + '═' * 52 + '╗'}")
    print(f"{'║'}{'长桥交易助手 - 交互式面板':^46}{'║'}")
    print(f"{'╠' + '═' * 52 + '╣'}")
    print(f"{'║'}  1. 分析股票标的                                  {'║'}")
    print(f"{'║'}  2. 查看账户持仓                                  {'║'}")
    print(f"{'║'}  3. 查看现金余额                                  {'║'}")
    print(f"{'║'}  4. 扫描美股关注清单 (智能推荐)                   {'║'}")
    print(f"{'║'}  5. 手动下单交易                                  {'║'}")
    print(f"{'║'}  6. 查看并撤销订单                                {'║'}")
    print(f"{'║'}  7. 启动自动交易引擎 (按 Ctrl+C 可停止并返回)     {'║'}")
    print(f"{'║'}  8. 退出                                          {'║'}")
    print(f"{'╚' + '═' * 52 + '╝'}")
    return input("\n  请选择 [1-8]: ").strip()


def main() -> None:
    setup_logging()

    print("\n  正在连接长桥 API ...")
    cfg = load_config()
    quote_ctx, trade_ctx = _build_connections(cfg)
    print("  连接成功!\n")

    while True:
        choice = main_menu()

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
            print("\n  即将启动自动交易引擎...")
            print(f"  模式: {'模拟盘' if cfg.paper_trading else '实盘'}")
            print(f"  标的: {cfg.watch_symbols}")
            confirm = input("  确认启动? (y/n): ").strip().lower()
            if confirm == "y":
                from main import TradingEngine
                engine = TradingEngine(cfg)
                # Override trade ctx to avoid reconnecting or double connections if possible, 
                # but TradingEngine initializes its own. For simplicity, we just let it run.
                print("  =======================================================")
                print("  💡 提示: 自动交易引擎正在前台运行，想退回菜单请按 Ctrl + C")
                print("  =======================================================\n")
                
                caffeinate_proc = None
                if platform.system() == "Darwin":
                    print("  ⚡ 已启动 macOS 防休眠 (caffeinate)")
                    # -i: prevent idle sleep, -s: prevent system sleep
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

        elif choice == "8":
            print("\n  再见!")
            sys.exit(0)

        else:
            print("  无效选择，请输入 1-8")


if __name__ == "__main__":
    main()
