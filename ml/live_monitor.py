"""
实盘信号监控器 (Live Signal Monitor)
整合数据拉取、特征工程、XGBoost 推理、Telegram 通知的主循环。

本模块不进行自动下单, 定位为用户的"交易副手" (Trade Copilot)。
当 XGBoost 模型触发买卖阈值时, 通过 Telegram 向用户推送结构化信号。
"""

from __future__ import annotations

import asyncio
import signal
import time
from collections import defaultdict

from longport.openapi import (
    Config,
    QuoteContext,
    Period,
    AdjustType,
    SubType,
    PushQuote,
)

from config import TradingConfig, load_config
from logger import get_logger, setup_logging
from ml.feature_engineer import build_feature_vector, FEATURE_NAMES
from ml.market_hours import (
    is_us_market_open,
    now_et,
    format_et_time,
    format_local_time,
    get_next_open_time,
)
from ml.telegram_notifier import TelegramNotifier
from ml.xgb_model import XGBModelManager

log = get_logger("live_monitor")

# K 线历史数量 (至少 100 根以确保 EMA60 等长周期指标可计算)
KLINE_HISTORY_COUNT = 100
API_REQUEST_INTERVAL = 0.15
API_MAX_RETRIES = 3

# 默认信号冷却时间 (秒): 同一股票同方向信号在此时间内不重复触发
DEFAULT_SIGNAL_COOLDOWN = 300


class LiveSignalMonitor:
    """
    实盘信号监控器。

    流程:
      1. 启动时单次加载 XGBoost 模型
      2. 接入长桥 API 拉取 5 分钟 K 线 (轮询)
      3. 对每只股票保留 100 根 K 线的滚动窗口
      4. 每轮检查是否在美股交易时段
      5. 计算 15 维特征 → 模型推理 → 信号判定
      6. 触发信号时通过 Telegram 推送
    """

    def __init__(self, cfg: TradingConfig):
        self._cfg = cfg

        # ── 模型 ──
        self._model = XGBModelManager()
        self._model_loaded = False

        # ── 数据 ──
        self._quote_ctx: QuoteContext | None = None
        self._active_symbols: list[str] = []
        self._candle_cache: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"closes": [], "highs": [], "lows": [], "volumes": []}
        )
        self._latest_price: dict[str, float] = {}

        # ── 信号冷却: {f"{symbol}:{direction}": last_trigger_timestamp} ──
        self._signal_cooldown: dict[str, float] = {}
        self._cooldown_seconds = getattr(
            cfg, "telegram", None
        )
        if self._cooldown_seconds and hasattr(self._cooldown_seconds, "signal_cooldown_seconds"):
            self._cooldown_seconds = self._cooldown_seconds.signal_cooldown_seconds
        else:
            self._cooldown_seconds = DEFAULT_SIGNAL_COOLDOWN

        # ── Telegram ──
        tg_cfg = getattr(cfg, "telegram", None)
        if tg_cfg and tg_cfg.enabled:
            self._notifier = TelegramNotifier(
                bot_token=tg_cfg.bot_token,
                chat_id=tg_cfg.chat_id,
            )
        else:
            self._notifier = None
            log.warning("Telegram notifier disabled (TELEGRAM_ENABLED=false or missing config)")

        # ── 控制 ──
        self._running = False

    # ── 初始化 ────────────────────────────────────────────────

    def _load_model(self) -> bool:
        """加载 XGBoost 模型 (单次)。"""
        model_name = self._cfg.ml.model_name
        if self._model.load(model_name):
            log.info("✅ XGBoost model '%s' loaded successfully", model_name)
            self._model_loaded = True
            return True
        else:
            log.error("❌ XGBoost model '%s' not found. Run 'python ml/train_xgb.py' first.", model_name)
            return False

    def _init_connections(self) -> None:
        """初始化长桥行情连接。"""
        cred = self._cfg.credentials
        lb_config = Config(
            app_key=cred.app_key,
            app_secret=cred.app_secret,
            access_token=cred.access_token,
        )
        self._quote_ctx = QuoteContext(lb_config)
        self._quote_ctx.set_on_quote(self._on_quote_push)
        log.info("QuoteContext connected")

    def _resolve_symbols(self) -> None:
        """确定监控标的列表。"""
        if self._cfg.watchlist.use_watchlist:
            from watchlist import WatchlistManager, parse_market_filter
            mgr = WatchlistManager(self._quote_ctx)
            market_filter = parse_market_filter(self._cfg.watchlist.market_filter)
            self._active_symbols = mgr.get_symbols_by_market(market_filter)
        else:
            self._active_symbols = self._cfg.watch_symbols.copy()

        log.info("Monitoring %d symbols: %s", len(self._active_symbols), self._active_symbols)

    def _on_quote_push(self, symbol: str, quote: PushQuote) -> None:
        """实时报价回调。"""
        self._latest_price[symbol] = float(quote.last_done)

    # ── K 线数据 ──────────────────────────────────────────────

    async def _fetch_candles_with_retry(self, symbol: str) -> list | None:
        """带重试的 K 线拉取。"""
        for attempt in range(API_MAX_RETRIES):
            try:
                return await asyncio.to_thread(
                    self._quote_ctx.candlesticks,
                    symbol, Period.Min_5, KLINE_HISTORY_COUNT, AdjustType.NoAdjust,
                )
            except Exception as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 0.5
                    log.warning(
                        "Retry %d/%d for %s: %s", attempt + 1, API_MAX_RETRIES, symbol, str(e)[:80],
                    )
                    await asyncio.sleep(wait_time)
                else:
                    log.error("Failed to fetch candles for %s after %d retries", symbol, API_MAX_RETRIES)
        return None

    async def _load_all_candles(self) -> None:
        """批量加载所有标的的历史 K 线。"""
        log.info("Loading historical 5m candles for %d symbols ...", len(self._active_symbols))

        async def fetch_and_store(symbol: str) -> None:
            candles = await self._fetch_candles_with_retry(symbol)
            if candles:
                self._candle_cache[symbol]["closes"] = [float(c.close) for c in candles]
                self._candle_cache[symbol]["highs"] = [float(c.high) for c in candles]
                self._candle_cache[symbol]["lows"] = [float(c.low) for c in candles]
                self._candle_cache[symbol]["volumes"] = [float(c.volume) for c in candles]
                log.info(
                    "Loaded %d candles for %s, latest close=%.2f",
                    len(candles), symbol,
                    self._candle_cache[symbol]["closes"][-1] if candles else 0,
                )

        chunk: list = []
        for symbol in self._active_symbols:
            chunk.append(fetch_and_store(symbol))
            if len(chunk) >= 5:
                await asyncio.gather(*chunk)
                await asyncio.sleep(API_REQUEST_INTERVAL)
                chunk = []
        if chunk:
            await asyncio.gather(*chunk)

    async def _refresh_all_candles(self) -> None:
        """刷新所有标的的最新 K 线。"""
        async def fetch_and_store(symbol: str) -> None:
            candles = await self._fetch_candles_with_retry(symbol)
            if candles:
                self._candle_cache[symbol]["closes"] = [float(c.close) for c in candles]
                self._candle_cache[symbol]["highs"] = [float(c.high) for c in candles]
                self._candle_cache[symbol]["lows"] = [float(c.low) for c in candles]
                self._candle_cache[symbol]["volumes"] = [float(c.volume) for c in candles]

        chunk: list = []
        for symbol in self._active_symbols:
            chunk.append(fetch_and_store(symbol))
            if len(chunk) >= 5:
                await asyncio.gather(*chunk)
                await asyncio.sleep(API_REQUEST_INTERVAL)
                chunk = []
        if chunk:
            await asyncio.gather(*chunk)

    # ── 信号处理 ──────────────────────────────────────────────

    def _is_cooled_down(self, symbol: str, direction: str) -> bool:
        """检查信号冷却: 同一股票同方向信号在冷却期内不重复触发。"""
        key = f"{symbol}:{direction}"
        last = self._signal_cooldown.get(key, 0)
        return (time.time() - last) >= self._cooldown_seconds

    def _record_signal(self, symbol: str, direction: str) -> None:
        """记录信号触发时间。"""
        self._signal_cooldown[f"{symbol}:{direction}"] = time.time()

    def _process_symbol(self, symbol: str) -> None:
        """对单只股票进行推理和信号判定。"""
        cache = self._candle_cache.get(symbol)
        if not cache:
            return

        closes = cache["closes"]
        highs = cache["highs"]
        lows = cache["lows"]
        volumes = cache["volumes"]

        if len(closes) < 60:
            return

        # ── 特征工程 ──
        features = build_feature_vector(
            closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            boll_period=self._cfg.strategy.boll_period,
            boll_std=self._cfg.strategy.boll_std_dev,
            rsi_period=self._cfg.strategy.rsi_period,
            ema_period=self._cfg.strategy.trend_ema_period,
        )

        if features is None:
            return

        # ── 模型推理 ──
        prob_up = self._model.predict_proba(features)
        current_price = self._latest_price.get(symbol) or closes[-1]

        buy_threshold = self._cfg.ml.xgb_buy_threshold
        sell_threshold = self._cfg.ml.xgb_sell_threshold

        log.info(
            "[%s] price=%.2f prob_up=%.3f (buy>=%.2f, sell<=%.2f)",
            symbol, current_price, prob_up, buy_threshold, sell_threshold,
        )

        # ── 信号判定 ──
        signal_type = None
        if prob_up >= buy_threshold:
            signal_type = "BUY"
        elif prob_up <= sell_threshold:
            signal_type = "SELL"

        if signal_type is None:
            return

        # ── 冷却检查 ──
        if not self._is_cooled_down(symbol, signal_type):
            log.debug("[%s] %s signal suppressed (cooldown)", symbol, signal_type)
            return

        self._record_signal(symbol, signal_type)

        # ── 构建特征快照 ──
        feature_snapshot = {}
        # 从 15 维特征中选取关键指标
        feature_dict = dict(zip(FEATURE_NAMES, features.tolist()))
        rsi_norm = feature_dict.get("rsi_norm", 0)
        feature_snapshot["RSI"] = f"{rsi_norm * 100:.1f}"
        feature_snapshot["EMA偏离度"] = f"{feature_dict.get('ema_deviation', 0) * 100:.2f}%"
        feature_snapshot["布林位置"] = f"{feature_dict.get('boll_position', 0):.3f}"
        feature_snapshot["波动率"] = f"{feature_dict.get('volatility_20', 0) * 100:.2f}%"
        feature_snapshot["MA5/MA20"] = f"{feature_dict.get('ma5_ma20_ratio', 0) * 100:.3f}%"

        # ── 日志输出 ──
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        log.info(
            "%s [%s] %s @ $%.2f | prob_up=%.3f | RSI=%.1f | EMA_dev=%.3f%%",
            emoji, symbol, signal_type, current_price, prob_up,
            rsi_norm * 100, feature_dict.get("ema_deviation", 0) * 100,
        )

        # ── Telegram 通知 ──
        if self._notifier:
            self._notifier.send_signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                prob_up=prob_up,
                features_snapshot=feature_snapshot,
                est_time_str=format_et_time(),
                local_time_str=format_local_time(),
            )

    # ── 主循环 ────────────────────────────────────────────────

    async def run(self) -> None:
        """异步主循环入口。"""
        log.info("=" * 60)
        log.info("Live Signal Monitor starting")
        log.info("  Mode:     Trade Copilot (通知模式, 不自动下单)")
        log.info("  Model:    %s", self._cfg.ml.model_name)
        log.info("  Buy>=:    %.2f", self._cfg.ml.xgb_buy_threshold)
        log.info("  Sell<=:   %.2f", self._cfg.ml.xgb_sell_threshold)
        log.info("  Cooldown: %ds", self._cooldown_seconds)
        log.info("  Telegram: %s", "enabled" if self._notifier else "disabled")
        log.info("=" * 60)

        # 1. 加载模型
        if not self._load_model():
            return

        # 2. 连接行情
        self._init_connections()
        self._resolve_symbols()

        if not self._active_symbols:
            log.error("No symbols to monitor, exiting")
            return

        # 3. 加载历史 K 线
        await self._load_all_candles()

        # 4. 订阅实时报价
        if self._quote_ctx and self._active_symbols:
            self._quote_ctx.subscribe(self._active_symbols, [SubType.Quote])
            log.info("Subscribed to real-time quotes")

        # 5. 启动通知
        if self._notifier:
            self._notifier.send_text(
                f"🚀 实盘信号监控已启动\n"
                f"📊 标的: {len(self._active_symbols)} 只\n"
                f"🤖 模型: {self._cfg.ml.model_name}\n"
                f"⏰ {format_et_time()}"
            )

        self._running = True

        def handle_sigint(_sig, _frame):
            log.info("Received signal, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        poll_interval = self._cfg.kline_poll_interval
        log.info("Entering main loop (poll every %ds) ...", poll_interval)

        try:
            while self._running:
                try:
                    # ── 交易时段检查 ──
                    if not is_us_market_open():
                        next_open = get_next_open_time()
                        log.info(
                            "Market closed. Next open: %s. Sleeping %ds ...",
                            format_et_time(next_open), poll_interval * 6,
                        )
                        await asyncio.sleep(poll_interval * 6)  # 非交易时段放慢轮询
                        continue

                    # ── 刷新 K 线 ──
                    await self._refresh_all_candles()

                    # ── 逐股推理 ──
                    for symbol in self._active_symbols:
                        self._process_symbol(symbol)

                except asyncio.CancelledError:
                    break
                except Exception:
                    log.exception("Error in monitor loop")

                await asyncio.sleep(poll_interval)

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            if self._notifier:
                self._notifier.send_text("🛑 实盘信号监控已停止")
                self._notifier.shutdown()
            log.info("Live Signal Monitor stopped")

    def run_sync(self) -> None:
        """同步入口。"""
        asyncio.run(self.run())


# ── CLI 入口 ──────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    cfg = load_config()
    monitor = LiveSignalMonitor(cfg)
    monitor.run_sync()


if __name__ == "__main__":
    main()
