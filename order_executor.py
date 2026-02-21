from __future__ import annotations

import asyncio
from decimal import Decimal

from longport.openapi import (
    Config,
    OrderSide,
    OrderType,
    TimeInForceType,
    TradeContext,
    TopicType,
    PushOrderChanged,
)

from config import TradingConfig
from logger import get_logger

log = get_logger("order_executor")


class OrderExecutor:
    def __init__(self, cfg: TradingConfig):
        self._cfg = cfg
        self._trade_ctx: TradeContext | None = None
        self._fill_callback = None

    def set_fill_callback(self, callback):
        self._fill_callback = callback

    def connect(self) -> None:
        lb_config = Config(
            app_key=self._cfg.credentials.app_key,
            app_secret=self._cfg.credentials.app_secret,
            access_token=self._cfg.credentials.access_token,
        )
        self._trade_ctx = TradeContext(lb_config)
        self._trade_ctx.set_on_order_changed(self._on_order_changed)
        self._trade_ctx.subscribe([TopicType.Private])
        log.info("TradeContext connected, order push subscribed")

    def _on_order_changed(self, event: PushOrderChanged) -> None:
        log.info(
            "Order update: %s %s %s status=%s qty=%s price=%s",
            event.side, event.symbol, event.order_type,
            event.status, event.executed_quantity, event.executed_price,
        )
        if event.status in ("Filled", "PartialFilled") and self._fill_callback:
            # We assume executed_quantity is the cumulative filled quantity.
            # For a proper production system, we'd calculate the delta, 
            # but for simplicity, the exact order fills are emitted.
            self._fill_callback(event)

    def submit_buy(self, symbol: str, quantity: int, price: Decimal | None = None) -> str | None:
        if self._cfg.paper_trading:
            log.info("[PAPER] BUY %s qty=%d price=%s", symbol, quantity, price)
            return f"PAPER-{symbol}-BUY"

        if self._trade_ctx is None:
            log.error("TradeContext not connected")
            return None

        try:
            if price is not None:
                resp = self._trade_ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.LO,
                    side=OrderSide.Buy,
                    submitted_quantity=Decimal(quantity),
                    time_in_force=TimeInForceType.Day,
                    submitted_price=price,
                    remark="auto-trader-buy",
                )
            else:
                resp = self._trade_ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.MO,
                    side=OrderSide.Buy,
                    submitted_quantity=Decimal(quantity),
                    time_in_force=TimeInForceType.Day,
                    remark="auto-trader-buy-market",
                )
            order_id = resp.order_id
            log.info("BUY order submitted: %s order_id=%s", symbol, order_id)
            return order_id
        except Exception:
            log.exception("Failed to submit BUY order for %s", symbol)
            return None

    def submit_sell(self, symbol: str, quantity: int, price: Decimal | None = None) -> str | None:
        if self._cfg.paper_trading:
            log.info("[PAPER] SELL %s qty=%d price=%s", symbol, quantity, price)
            return f"PAPER-{symbol}-SELL"

        if self._trade_ctx is None:
            log.error("TradeContext not connected")
            return None

        try:
            if price is not None:
                resp = self._trade_ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.LO,
                    side=OrderSide.Sell,
                    submitted_quantity=Decimal(quantity),
                    time_in_force=TimeInForceType.Day,
                    submitted_price=price,
                    remark="auto-trader-sell",
                )
            else:
                resp = self._trade_ctx.submit_order(
                    symbol=symbol,
                    order_type=OrderType.MO,
                    side=OrderSide.Sell,
                    submitted_quantity=Decimal(quantity),
                    time_in_force=TimeInForceType.Day,
                    remark="auto-trader-sell-market",
                )
            order_id = resp.order_id
            log.info("SELL order submitted: %s order_id=%s", symbol, order_id)
            return order_id
        except Exception:
            log.exception("Failed to submit SELL order for %s", symbol)
            return None

    def cancel_all_pending_orders(self) -> int:
        if self._cfg.paper_trading:
            log.info("[PAPER] Cancel all pending orders")
            return 0

        if self._trade_ctx is None:
            log.error("TradeContext not connected")
            return 0

        cancelled = 0
        try:
            orders = self._trade_ctx.today_orders()
            for order in orders:
                if order.status in ("NewStatus", "WaitToNew", "PartialFilled", "PendingReplace"):
                    try:
                        self._trade_ctx.cancel_order(order.order_id)
                        cancelled += 1
                        log.info("Cancelled order %s (%s)", order.order_id, order.symbol)
                    except Exception:
                        log.exception("Failed to cancel order %s", order.order_id)
        except Exception:
            log.exception("Failed to fetch today orders for cancellation")

        return cancelled

    def get_stock_positions(self) -> list:
        if self._cfg.paper_trading:
            return []

        if self._trade_ctx is None:
            return []

        try:
            resp = self._trade_ctx.stock_positions()
            positions = []
            for channel in resp.channels:
                positions.extend(channel.positions)
            return positions
        except Exception:
            log.exception("Failed to get stock positions")
            return []

    def get_total_assets(self) -> Decimal:
        if self._cfg.paper_trading:
            return Decimal("1000000")

        if self._trade_ctx is None:
            return Decimal("0")

        try:
            balances = self._trade_ctx.account_balance()
            total = Decimal("0")
            for bal in balances:
                total += Decimal(str(bal.total_cash))
            return total
        except Exception:
            log.exception("Failed to get account balance")
            return Decimal("0")

    def get_account_balance_detail(self) -> list:
        if self._trade_ctx is None:
            return []
        try:
            return list(self._trade_ctx.account_balance())
        except Exception:
            log.exception("Failed to get account balance detail")
            return []
