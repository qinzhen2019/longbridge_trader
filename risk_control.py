"""
风险控制模块
- 硬止损: 单笔亏损超阈值则市价清仓
- 最大回撤保护: 当日账户总资产回撤超阈值则停止交易
- 频率控制: 同一标的冷却期内不重复交易
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal

from logger import get_logger

log = get_logger("risk_control")


@dataclass
class PositionRecord:
    symbol: str
    entry_price: Decimal
    quantity: int
    side: str  # "Buy" | "Sell"
    entry_time: float = field(default_factory=time.time)
    
@dataclass
class OrderFill:
    symbol: str
    quantity: int
    price: Decimal
    side: str



class RiskController:
    def __init__(
        self,
        stop_loss_pct: Decimal,
        take_profit_pct: Decimal,
        max_drawdown_pct: Decimal,
        cooldown_seconds: int,
        max_position_value: Decimal = Decimal("1000"),
        max_total_exposure: Decimal = Decimal("10000"),
        max_positions: int = 10,
    ):
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._cooldown_seconds = cooldown_seconds
        self._max_position_value = max_position_value
        self._max_total_exposure = max_total_exposure
        self._max_positions = max_positions

        self._positions: dict[str, PositionRecord] = {}
        self._last_trade_time: dict[str, float] = {}
        self._trading_halted = False
        self._daily_peak_assets: Decimal | None = None

    @property
    def trading_halted(self) -> bool:
        return self._trading_halted

    def halt_trading(self, reason: str) -> None:
        self._trading_halted = True
        log.critical("TRADING HALTED: %s", reason)

    def resume_trading(self) -> None:
        self._trading_halted = False
        log.info("Trading resumed")

    def reset_daily(self) -> None:
        self._daily_peak_assets = None
        self._trading_halted = False
        log.info("Daily risk counters reset")

    def record_position(self, symbol: str, entry_price: Decimal, quantity: int, side: str) -> None:
        self._positions[symbol] = PositionRecord(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            side=side,
        )
        self._last_trade_time[symbol] = time.time()
        log.info("Position recorded: %s %s %d @ %s", side, symbol, quantity, entry_price)

    def sync_positions_from_broker(self, broker_positions: list) -> int:
        synced = 0
        for pos in broker_positions:
            symbol = pos.symbol
            if symbol in self._positions:
                continue

            quantity = int(pos.quantity)
            if quantity <= 0:
                continue

            cost_price = Decimal(str(pos.cost_price)) if pos.cost_price else Decimal("0")

            self._positions[symbol] = PositionRecord(
                symbol=symbol,
                entry_price=cost_price,
                quantity=quantity,
                side="Buy",
            )
            synced += 1
            log.info("Synced position from broker: %s %d @ %s", symbol, quantity, cost_price)

        log.info("Total synced positions: %d, current positions: %d", synced, len(self._positions))
        return synced

    def clear_position(self, symbol: str) -> None:
        if symbol in self._positions:
            self._positions.pop(symbol, None)
            self._last_trade_time[symbol] = time.time()
            log.info("Position cleared: %s", symbol)

    def apply_order_fill(self, fill: OrderFill) -> None:
        """根据真实成交更新仓位信息"""
        if fill.side == "Buy":
            if fill.symbol in self._positions:
                pos = self._positions[fill.symbol]
                new_qty = pos.quantity + fill.quantity
                # 计算平均持仓成本
                new_price = (pos.entry_price * pos.quantity + fill.price * fill.quantity) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_price
                log.info("Position updated (Add): %s %d @ %.2f", fill.symbol, new_qty, new_price)
            else:
                self.record_position(fill.symbol, fill.price, fill.quantity, fill.side)
        elif fill.side == "Sell":
            if fill.symbol in self._positions:
                pos = self._positions[fill.symbol]
                new_qty = pos.quantity - fill.quantity
                if new_qty <= 0:
                    self.clear_position(fill.symbol)
                else:
                    pos.quantity = new_qty
                    log.info("Position updated (Reduce): %s %d remaining", fill.symbol, new_qty)
            else:
                log.warning("Received SELL fill for %s but no position found", fill.symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> PositionRecord | None:
        return self._positions.get(symbol)

    def is_in_cooldown(self, symbol: str) -> bool:
        last = self._last_trade_time.get(symbol)
        if last is None:
            return False
        elapsed = time.time() - last
        in_cooldown = elapsed < self._cooldown_seconds
        if in_cooldown:
            remaining = self._cooldown_seconds - elapsed
            log.debug("%s in cooldown, %.0fs remaining", symbol, remaining)
        return in_cooldown

    def check_stop_loss(self, symbol: str, current_price: Decimal) -> bool:
        pos = self._positions.get(symbol)
        if pos is None:
            return False

        if pos.side == "Buy":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

        if pnl_pct <= -self._stop_loss_pct:
            log.warning(
                "STOP LOSS triggered for %s: pnl=%.2f%%, threshold=-%.2f%%",
                symbol, float(pnl_pct), float(self._stop_loss_pct),
            )
            return True
        return False

    def check_take_profit(self, symbol: str, current_price: Decimal) -> bool:
        pos = self._positions.get(symbol)
        if pos is None:
            return False

        if pos.side == "Buy":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

        if pnl_pct >= self._take_profit_pct:
            log.info(
                "TAKE PROFIT triggered for %s: pnl=%.2f%%, threshold=%.2f%%",
                symbol, float(pnl_pct), float(self._take_profit_pct),
            )
            return True
        return False

    def check_max_drawdown(self, current_total_assets: Decimal) -> bool:
        if self._daily_peak_assets is None:
            self._daily_peak_assets = current_total_assets
            return False

        if current_total_assets > self._daily_peak_assets:
            self._daily_peak_assets = current_total_assets

        drawdown_pct = (
            (self._daily_peak_assets - current_total_assets)
            / self._daily_peak_assets
            * 100
        )

        if drawdown_pct >= self._max_drawdown_pct:
            self.halt_trading(
                f"Max drawdown breached: {float(drawdown_pct):.2f}% >= {float(self._max_drawdown_pct)}%"
            )
            return True
        return False

    def can_trade(self, symbol: str) -> bool:
        if self._trading_halted:
            log.warning("Trade rejected for %s: trading is halted", symbol)
            return False
        if self.is_in_cooldown(symbol):
            log.warning("Trade rejected for %s: in cooldown", symbol)
            return False
        return True

    def get_total_exposure(self) -> Decimal:
        total = Decimal("0")
        for pos in self._positions.values():
            total += pos.entry_price * pos.quantity
        return total

    def get_position_count(self) -> int:
        return len(self._positions)

    def can_open_position(self, symbol: str, price: Decimal) -> bool:
        if self._trading_halted:
            log.warning("Trade rejected for %s: trading halted", symbol)
            return False

        if self.is_in_cooldown(symbol):
            log.warning("Trade rejected for %s: in cooldown", symbol)
            return False

        if self.has_position(symbol):
            log.warning("Trade rejected for %s: already has position", symbol)
            return False

        if self.get_position_count() >= self._max_positions:
            log.warning("Trade rejected for %s: max positions (%d) reached",
                        symbol, self._max_positions)
            return False

        current_exposure = self.get_total_exposure()
        new_exposure = current_exposure + self._max_position_value
        if new_exposure > self._max_total_exposure:
            log.warning("Trade rejected for %s: total exposure $%.2f + $%.2f > max $%.2f",
                        symbol, float(current_exposure), float(self._max_position_value),
                        float(self._max_total_exposure))
            return False

        return True
