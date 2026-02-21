from __future__ import annotations

import asyncio
import signal
import sys
import time
from collections import defaultdict
from decimal import Decimal

from longport.openapi import (
    Config,
    QuoteContext,
    Period,
    AdjustType,
    SubType,
    PushQuote,
)

from config import TradingConfig, load_config
from indicators import compute_indicators
from logger import get_logger, setup_logging
from order_executor import OrderExecutor
from risk_control import RiskController, OrderFill
from strategy import BollingerRsiStrategy, Signal
from strategy_base import BaseStrategy
from watchlist import WatchlistManager, parse_market_filter

log = get_logger("main")

KLINE_HISTORY_COUNT = 100
API_REQUEST_INTERVAL = 0.15
API_MAX_RETRIES = 3


class TradingEngine:
    def __init__(self, cfg: TradingConfig):
        self._cfg = cfg

        self._risk = RiskController(
            stop_loss_pct=cfg.risk.stop_loss_pct,
            take_profit_pct=cfg.risk.take_profit_pct,
            max_drawdown_pct=cfg.risk.max_drawdown_pct,
            cooldown_seconds=cfg.risk.trade_cooldown_seconds,
            max_position_value=cfg.risk.max_position_value,
            max_total_exposure=cfg.risk.max_total_exposure,
            max_positions=cfg.risk.max_positions,
        )

        self._strategy = self._create_strategy(cfg)
        self._executor = OrderExecutor(cfg)

        self._quote_ctx: QuoteContext | None = None
        self._watchlist_mgr: WatchlistManager | None = None
        self._active_symbols: list[str] = []
        self._candle_cache: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"1m": [], "5m": []}
        )

        self._latest_price: dict[str, float] = {}
        self._running = False
        self._last_watchlist_refresh: float = 0
        self._order_fill_queue: asyncio.Queue[OrderFill] | None = None

    @staticmethod
    def _create_strategy(cfg: TradingConfig) -> BaseStrategy:
        """根据配置动态选择策略引擎。"""
        strategy_type = cfg.ml.strategy_type.lower()

        if strategy_type == "xgboost":
            from ml.xgb_strategy import XGBoostStrategy
            log.info("Using XGBoost ML strategy (buy_threshold=%.2f)", cfg.ml.xgb_buy_threshold)
            return XGBoostStrategy(
                params=cfg.strategy,
                model_name=cfg.ml.model_name,
                buy_threshold=cfg.ml.xgb_buy_threshold,
                sell_threshold=cfg.ml.xgb_sell_threshold,
            )
        elif strategy_type == "rl":
            from ml.rl_strategy import RLStrategy
            log.info("Using RL strategy (algo=%s)", cfg.ml.rl_algo)
            return RLStrategy(
                params=cfg.strategy,
                algo=cfg.ml.rl_algo,
                model_name=cfg.ml.model_name,
            )
        else:
            log.info("Using default BollingerRsi strategy")
            return BollingerRsiStrategy(cfg.strategy)

    def _init_connections(self) -> None:
        lb_config = Config(
            app_key=self._cfg.credentials.app_key,
            app_secret=self._cfg.credentials.app_secret,
            access_token=self._cfg.credentials.access_token,
        )
        self._quote_ctx = QuoteContext(lb_config)
        self._quote_ctx.set_on_quote(self._on_quote_push)

        if self._cfg.watchlist.use_watchlist:
            self._watchlist_mgr = WatchlistManager(self._quote_ctx)
            self._refresh_watchlist_symbols()
        else:
            self._active_symbols = self._cfg.watch_symbols.copy()

        self._order_fill_queue = asyncio.Queue()
        self._executor.set_fill_callback(self._on_fill_callback)
        self._executor.connect()
        log.info("All connections established")

        self._sync_existing_positions()

    def _sync_existing_positions(self) -> None:
        log.info("Syncing existing positions from broker...")
        broker_positions = self._executor.get_stock_positions()

        if not broker_positions:
            log.info("No existing positions found in broker account")
            return

        synced = self._risk.sync_positions_from_broker(broker_positions)

        for pos in broker_positions:
            symbol = pos.symbol
            if symbol not in self._active_symbols:
                self._active_symbols.append(symbol)
                log.info("Added existing position %s to active symbols", symbol)

    def _refresh_watchlist_symbols(self) -> None:
        if self._watchlist_mgr is None:
            return

        market_filter = parse_market_filter(self._cfg.watchlist.market_filter)
        new_symbols = self._watchlist_mgr.get_symbols_by_market(market_filter)

        if not new_symbols:
            log.warning("No symbols found in watchlist for filter %s", self._cfg.watchlist.market_filter)
            return

        added = set(new_symbols) - set(self._active_symbols)
        removed = set(self._active_symbols) - set(new_symbols)

        if added:
            log.info("New symbols added to watch: %s", list(added))
        if removed:
            log.info("Symbols removed from watch: %s", list(removed))

        self._active_symbols = new_symbols
        self._last_watchlist_refresh = time.time()

    def _on_quote_push(self, symbol: str, quote: PushQuote) -> None:
        self._latest_price[symbol] = float(quote.last_done)



    def _on_fill_callback(self, event) -> None:
        """Called by OrderExecutor thread when an order fills."""
        queue = self._order_fill_queue
        if queue is None:
            return
        
        try:
            side = "Buy" if str(event.side).endswith("Buy") else "Sell"
            fill = OrderFill(
                symbol=event.symbol,
                quantity=int(event.executed_quantity),
                price=Decimal(str(event.executed_price)),
                side=side
            )
            # Use put_nowait because this might be called from a different thread
            # and we just want to enqueue it for the main async loop.
            queue.put_nowait(fill)
        except Exception:
            log.exception("Error enqueuing order fill")

    async def _fetch_candles_with_retry(self, symbol: str, period_enum: Period) -> list | None:
        for attempt in range(API_MAX_RETRIES):
            try:
                return await asyncio.to_thread(
                    self._quote_ctx.candlesticks,
                    symbol, period_enum, KLINE_HISTORY_COUNT, AdjustType.NoAdjust
                )
            except Exception as e:
                if attempt < API_MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 0.5
                    log.warning("Retry %d/%d for %s: %s, waiting %.1fs",
                                attempt + 1, API_MAX_RETRIES, symbol, str(e)[:50], wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    log.error("Failed to fetch candles for %s after %d retries", symbol, API_MAX_RETRIES)
        return None

    async def _load_historical_candles_async(self) -> None:
        if self._quote_ctx is None:
            return

        log.info("Loading historical candles for %d symbols concurrently...", len(self._active_symbols))
        
        async def fetch_and_store(symbol: str, period: str, period_enum: Period) -> None:
            candles = await self._fetch_candles_with_retry(symbol, period_enum)
            if candles:
                closes = [float(c.close) for c in candles]
                self._candle_cache[symbol][period] = closes
                log.info(
                    "Loaded %d %s candles for %s, latest close=%.2f",
                    len(closes), period, symbol, closes[-1] if closes else 0,
                )

        chunk_tasks = []
        for symbol in self._active_symbols:
            chunk_tasks.append(fetch_and_store(symbol, "1m", Period.Min_1))
            chunk_tasks.append(fetch_and_store(symbol, "5m", Period.Min_5))
            if len(chunk_tasks) >= 10:
                await asyncio.gather(*chunk_tasks)
                await asyncio.sleep(API_REQUEST_INTERVAL)
                chunk_tasks = []
            
        if chunk_tasks:
            await asyncio.gather(*chunk_tasks)

    async def _refresh_candles_async(self) -> None:
        if self._quote_ctx is None:
            return

        async def fetch_and_store(symbol: str, period: str, period_enum: Period) -> None:
            candles = await self._fetch_candles_with_retry(symbol, period_enum)
            if candles:
                closes = [float(c.close) for c in candles]
                self._candle_cache[symbol][period] = closes

        chunk_tasks = []
        for symbol in self._active_symbols:
            chunk_tasks.append(fetch_and_store(symbol, "1m", Period.Min_1))
            chunk_tasks.append(fetch_and_store(symbol, "5m", Period.Min_5))
            if len(chunk_tasks) >= 10:
                await asyncio.gather(*chunk_tasks)
                await asyncio.sleep(API_REQUEST_INTERVAL)
                chunk_tasks = []

        if chunk_tasks:
            await asyncio.gather(*chunk_tasks)

    def _evaluate_and_trade(self) -> None:
        self._maybe_refresh_watchlist()

        total_assets = self._executor.get_total_assets()
        if self._risk.check_max_drawdown(total_assets):
            cancelled = self._executor.cancel_all_pending_orders()
            log.warning("Max drawdown breached — cancelled %d pending orders", cancelled)
            return

        for symbol in self._active_symbols:
            self._process_symbol(symbol)

    def _maybe_refresh_watchlist(self) -> None:
        if not self._cfg.watchlist.use_watchlist or self._watchlist_mgr is None:
            return

        elapsed = time.time() - self._last_watchlist_refresh
        if elapsed < self._cfg.watchlist.refresh_interval:
            return

        log.info("Refreshing watchlist (interval=%ds)", self._cfg.watchlist.refresh_interval)
        old_symbols = set(self._active_symbols)
        self._refresh_watchlist_symbols()
        new_symbols = set(self._active_symbols)

        newly_added = new_symbols - old_symbols
        if newly_added and self._quote_ctx is not None:
            self._quote_ctx.subscribe(list(newly_added), [SubType.Quote])
            log.info("Subscribed to newly added symbols: %s", list(newly_added))

    def _calculate_quantity(self, price: float, max_value: Decimal) -> int:
        if price <= 0:
            return 0
        qty = int(float(max_value) / price)
        return max(1, qty)

    def _process_symbol(self, symbol: str) -> None:
        closes_1m = self._candle_cache[symbol]["1m"]
        closes_5m = self._candle_cache[symbol]["5m"]

        if not closes_1m and not closes_5m:
            return

        primary_closes = closes_5m if len(closes_5m) >= self._cfg.strategy.boll_period else closes_1m
        if not primary_closes:
            return

        current_price = self._latest_price.get(symbol) or (primary_closes[-1] if primary_closes else None)
        if current_price is None:
            return

        current_price_decimal = Decimal(str(current_price))
        position = self._risk.get_position(symbol)

        if position is not None:
            if self._risk.check_stop_loss(symbol, current_price_decimal):
                log.warning("STOP LOSS — force selling %s at market", symbol)
                self._executor.submit_sell(symbol, position.quantity)
                # We do NOT clear_position here, we wait for the Fill event
                return

            if self._risk.check_take_profit(symbol, current_price_decimal):
                log.info("TAKE PROFIT — selling %s at market", symbol)
                self._executor.submit_sell(symbol, position.quantity)
                # We do NOT clear_position here, we wait for the Fill event
                return

        indicators = compute_indicators(
            primary_closes,
            boll_period=self._cfg.strategy.boll_period,
            boll_std=self._cfg.strategy.boll_std_dev,
            rsi_period=self._cfg.strategy.rsi_period,
        )

        has_position = position is not None
        result = self._strategy.evaluate(symbol, current_price, indicators, has_position)

        boll = indicators.bollinger
        rsi = indicators.rsi
        log.info(
            "[%s] price=%.2f | BB(%.2f / %.2f / %.2f) | RSI=%.1f | pos=%s | signal=%s",
            symbol,
            current_price,
            boll.lower if boll else 0,
            boll.middle if boll else 0,
            boll.upper if boll else 0,
            rsi if rsi is not None else -1,
            has_position,
            result.signal.name,
        )

        if result.signal == Signal.BUY:
            if not self._risk.can_open_position(symbol, current_price_decimal):
                return
            quantity = self._calculate_quantity(current_price, self._cfg.risk.max_position_value)
            if quantity <= 0:
                log.warning("Calculated quantity is 0 for %s at price %.2f", symbol, current_price)
                return
            log.info("BUY signal accepted: submitting market buy for %s x %d", symbol, quantity)
            self._executor.submit_buy(symbol, quantity)
            # We do NOT record_position here, we wait for the Fill event

        elif result.signal == Signal.SELL and position is not None:
            log.info("SELL signal accepted: submitting market sell for %s x %d", symbol, position.quantity)
            self._executor.submit_sell(symbol, position.quantity)
            # We do NOT clear_position here, we wait for the Fill event

    async def _process_fill_events(self) -> None:
        """Background task to process completed order fills."""
        while self._running:
            queue = self._order_fill_queue
            if queue is None:
                await asyncio.sleep(1)
                continue
            
            try:
                # Wait for fill events from the queue
                fill = await asyncio.wait_for(queue.get(), timeout=1.0)
                self._risk.apply_order_fill(fill)
            except asyncio.TimeoutError:
                pass
            except Exception:
                log.exception("Error processing fill event")

    async def run_async(self) -> None:
        log.info("=" * 60)
        log.info("Trading Engine starting (Async)")
        log.info("Paper trading: %s", self._cfg.paper_trading)
        log.info("Use watchlist: %s", self._cfg.watchlist.use_watchlist)
        if self._cfg.watchlist.use_watchlist:
            log.info("Watchlist market filter: %s", self._cfg.watchlist.market_filter)
            log.info("Watchlist refresh interval: %ds", self._cfg.watchlist.refresh_interval)
        else:
            log.info("Static symbols: %s", self._cfg.watch_symbols)
        log.info("Order quantity: %d", self._cfg.order_quantity)
        log.info(
            "Strategy: Bollinger(%d, %.1f) + RSI(%d, oversold=%.0f, overbought=%.0f)",
            self._cfg.strategy.boll_period,
            self._cfg.strategy.boll_std_dev,
            self._cfg.strategy.rsi_period,
            self._cfg.strategy.rsi_oversold,
            self._cfg.strategy.rsi_overbought,
        )
        log.info(
            "Risk: stop_loss=%.1f%%, take_profit=%.1f%%, max_drawdown=%.1f%%, cooldown=%ds",
            float(self._cfg.risk.stop_loss_pct),
            float(self._cfg.risk.take_profit_pct),
            float(self._cfg.risk.max_drawdown_pct),
            self._cfg.risk.trade_cooldown_seconds,
        )
        log.info("=" * 60)

        # Connect synchronously before the event loop starts its heavy work
        self._init_connections()

        if not self._active_symbols:
            log.error("No symbols to watch, exiting")
            return

        log.info("Active symbols (%d): %s", len(self._active_symbols), self._active_symbols)

        await self._load_historical_candles_async()
        
        # Subscribe to realtime quotes
        if self._quote_ctx is not None and self._active_symbols:
            self._quote_ctx.subscribe(self._active_symbols, [SubType.Quote])
            log.info("Subscribed to real-time quotes: %s", self._active_symbols)

        self._running = True

        def handle_sigint(_sig, _frame):
            log.info("Received SIGINT, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        poll_interval = self._cfg.kline_poll_interval
        log.info("Entering async main loop (poll every %ds)...", poll_interval)

        # Start the background tasks
        fill_task = asyncio.create_task(self._process_fill_events())

        try:
            while self._running:
                try:
                    await self._refresh_candles_async()
                    
                    # Evaluate trading logic based on fresh candles and realtime prices
                    self._evaluate_and_trade()
                except asyncio.CancelledError:
                    break
                except Exception:
                    log.exception("Error in main loop iteration")

                await asyncio.sleep(poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            fill_task.cancel()
            log.info("Trading Engine stopped")

    def run(self) -> None:
        """Synchronous wrapper to start the asyncio loop."""
        asyncio.run(self.run_async())


def main() -> None:
    setup_logging()
    cfg = load_config()
    engine = TradingEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
