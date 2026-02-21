"""关注清单管理模块 - 从长桥 API 获取用户关注清单，支持按市场过滤"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from longport.openapi import QuoteContext

from logger import get_logger

log = get_logger("watchlist")


class Market(Enum):
    US = "US"
    HK = "HK"
    CN = "CN"
    ALL = "ALL"


@dataclass
class WatchlistSymbol:
    symbol: str
    name: str
    market: Market


class WatchlistManager:
    def __init__(self, quote_ctx: QuoteContext):
        self._quote_ctx = quote_ctx
        self._cached_symbols: list[str] = []

    @staticmethod
    def _detect_market(symbol: str) -> Market:
        symbol_upper = symbol.upper()
        if symbol_upper.endswith(".US"):
            return Market.US
        elif symbol_upper.endswith(".HK"):
            return Market.HK
        elif symbol_upper.endswith(".SH") or symbol_upper.endswith(".SZ"):
            return Market.CN
        return Market.ALL

    def fetch_all_watchlist_symbols(self) -> list[WatchlistSymbol]:
        all_symbols: list[WatchlistSymbol] = []
        seen: set[str] = set()

        try:
            watchlist_groups = self._quote_ctx.watchlist()
            log.info("Fetched %d watchlist groups", len(watchlist_groups))

            for group in watchlist_groups:
                group_name = getattr(group, 'name', 'Unknown')
                securities = getattr(group, 'securities', [])
                log.debug("Group '%s' has %d securities", group_name, len(securities))

                for sec in securities:
                    if isinstance(sec, str):
                        symbol = sec
                        name = sec
                    else:
                        symbol = getattr(sec, 'symbol', str(sec))
                        name = getattr(sec, 'name', symbol)

                    if symbol not in seen:
                        seen.add(symbol)
                        market = self._detect_market(symbol)
                        all_symbols.append(WatchlistSymbol(
                            symbol=symbol,
                            name=name,
                            market=market,
                        ))

            log.info("Total unique symbols in watchlist: %d", len(all_symbols))

        except Exception:
            log.exception("Failed to fetch watchlist from Longport API")

        return all_symbols

    def get_symbols_by_market(self, market_filter: Market | list[Market]) -> list[str]:
        all_symbols = self.fetch_all_watchlist_symbols()

        if isinstance(market_filter, Market):
            markets = [market_filter]
        else:
            markets = market_filter

        if Market.ALL in markets:
            result = [s.symbol for s in all_symbols]
        else:
            result = [s.symbol for s in all_symbols if s.market in markets]

        self._cached_symbols = result
        log.info(
            "Filtered symbols by market %s: %d found — %s",
            [m.value for m in markets],
            len(result),
            result[:10] if len(result) > 10 else result,
        )

        return result

    def get_us_symbols(self) -> list[str]:
        return self.get_symbols_by_market(Market.US)

    def get_hk_symbols(self) -> list[str]:
        return self.get_symbols_by_market(Market.HK)

    def get_cached_symbols(self) -> list[str]:
        return self._cached_symbols.copy()

    def refresh(self, market_filter: Market | list[Market] = Market.US) -> list[str]:
        log.info("Refreshing watchlist...")
        return self.get_symbols_by_market(market_filter)


def parse_market_filter(filter_str: str) -> list[Market]:
    if not filter_str or filter_str.upper() == "ALL":
        return [Market.ALL]

    markets = []
    for m in filter_str.upper().split(","):
        m = m.strip()
        try:
            markets.append(Market(m))
        except ValueError:
            log.warning("Unknown market filter: %s, skipping", m)

    return markets if markets else [Market.ALL]
