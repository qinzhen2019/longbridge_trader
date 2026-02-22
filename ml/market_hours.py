"""
美股交易时段判断模块
判断当前是否在美股常规交易时段 (美东 9:30 - 16:00)。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def now_et() -> datetime:
    """返回当前美东时间。"""
    return datetime.now(_ET)


def is_us_market_open() -> bool:
    """
    判断当前是否在美股常规交易时段 (美东 9:30 - 16:00, 周一至周五)。

    注意: 本函数不处理美股公共假日，仅判断工作日 + 时间范围。
    """
    et_now = now_et()
    # 周一=0 … 周五=4, 周六=5, 周日=6
    if et_now.weekday() >= 5:
        return False
    return _MARKET_OPEN <= et_now.time() < _MARKET_CLOSE


def get_next_open_time() -> datetime:
    """
    返回下一个开盘时间 (美东)。

    如果当前在交易时段内, 返回的是 *明天* 的开盘时间。
    """
    et_now = now_et()
    today_open = et_now.replace(
        hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute,
        second=0, microsecond=0,
    )

    # 如果今天还没开盘 (且是工作日)
    if et_now.weekday() < 5 and et_now.time() < _MARKET_OPEN:
        return today_open

    # 否则跳到明天
    next_day = et_now + timedelta(days=1)
    # 跳过周末
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day.replace(
        hour=_MARKET_OPEN.hour, minute=_MARKET_OPEN.minute,
        second=0, microsecond=0,
    )


def format_et_time(dt: datetime | None = None) -> str:
    """格式化为美东时间字符串。"""
    if dt is None:
        dt = now_et()
    else:
        dt = dt.astimezone(_ET)
    return dt.strftime("%Y-%m-%d %H:%M:%S ET")


def format_local_time(dt: datetime | None = None) -> str:
    """格式化为本地时间字符串。"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")
