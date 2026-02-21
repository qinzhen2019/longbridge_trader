"""
日志系统
统一的日志配置，记录信号触发、下单、API 错误等所有关键事件。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "trader.log"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """初始化全局日志配置。"""
    root = logging.getLogger()
    root.setLevel(level)

    # 避免重复添加 handler
    if root.handlers:
        return

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # 文件
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取命名日志记录器。"""
    return logging.getLogger(name)
