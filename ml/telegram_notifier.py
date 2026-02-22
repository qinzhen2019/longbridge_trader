"""
Telegram 交易信号通知模块
高度解耦的 Telegram Bot 通知器, 用于推送 XGBoost 模型产出的买卖信号。

设计要点:
- 使用 requests 发送轻量 REST 请求
- 异步无阻塞 (ThreadPoolExecutor)
- 超时 + 重试机制
- MarkdownV2 格式消息
- InlineKeyboard 快捷按钮
- 股票级别的静音冷却
"""

from __future__ import annotations

import time as _time
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import requests

from logger import get_logger

log = get_logger("telegram")

_API_BASE = "https://api.telegram.org/bot{token}"

# MarkdownV2 需要转义的特殊字符
_MD2_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def _escape_md2(text: str) -> str:
    """转义 MarkdownV2 特殊字符。"""
    result = []
    for ch in str(text):
        if ch in _MD2_ESCAPE_CHARS:
            result.append("\\")
        result.append(ch)
    return "".join(result)


class TelegramNotifier:
    """
    Telegram Bot 信号通知器。

    不阻塞主线程: 所有发送操作通过线程池异步执行。
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: int = 10,
        max_retries: int = 3,
    ):
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._max_retries = max_retries
        self._api_base = _API_BASE.format(token=bot_token)

        # 静音管理: {symbol: mute_until_timestamp}
        self._muted: dict[str, float] = {}

        # 线程池: 保证不阻塞主循环
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tg")

        self._enabled = bool(bot_token and chat_id)
        if not self._enabled:
            log.warning("Telegram notifier disabled: missing bot_token or chat_id")

    # ── 静音管理 ──────────────────────────────────────────────

    def mute_symbol(self, symbol: str, duration_seconds: int = 3600) -> None:
        """静音某只股票的通知 (默认 1 小时)。"""
        self._muted[symbol] = _time.time() + duration_seconds
        log.info("Muted %s for %d seconds", symbol, duration_seconds)

    def unmute_symbol(self, symbol: str) -> None:
        """取消静音。"""
        self._muted.pop(symbol, None)

    def is_muted(self, symbol: str) -> bool:
        """检查某只股票是否处于静音状态。"""
        mute_until = self._muted.get(symbol)
        if mute_until is None:
            return False
        if _time.time() >= mute_until:
            # 静音已过期, 自动清除
            del self._muted[symbol]
            return False
        return True

    # ── 消息发送 ──────────────────────────────────────────────

    def send_signal(
        self,
        signal_type: str,
        symbol: str,
        price: float,
        prob_up: float,
        features_snapshot: dict[str, float] | None = None,
        est_time_str: str = "",
        local_time_str: str = "",
    ) -> None:
        """
        异步发送交易信号到 Telegram (非阻塞)。

        Args:
            signal_type: "BUY" 或 "SELL"
            symbol: 股票代码, e.g. "AAPL.US"
            price: 当前价格
            prob_up: 模型预测上涨概率
            features_snapshot: 核心特征快照 {"RSI": 32.5, "EMA偏离度": -1.2, ...}
            est_time_str: 美东时间字符串
            local_time_str: 本地时间字符串
        """
        if not self._enabled:
            return

        if self.is_muted(symbol):
            log.debug("Signal for %s suppressed (muted)", symbol)
            return

        self._executor.submit(
            self._send_signal_sync,
            signal_type, symbol, price, prob_up,
            features_snapshot, est_time_str, local_time_str,
        )

    def send_text(self, text: str) -> None:
        """异步发送纯文本消息 (非阻塞)。"""
        if not self._enabled:
            return
        self._executor.submit(self._send_message_sync, text, parse_mode=None)

    # ── 内部实现 ──────────────────────────────────────────────

    def _send_signal_sync(
        self,
        signal_type: str,
        symbol: str,
        price: float,
        prob_up: float,
        features_snapshot: dict[str, float] | None,
        est_time_str: str,
        local_time_str: str,
    ) -> None:
        """在工作线程中同步发送信号消息。"""
        try:
            text = self._build_signal_message(
                signal_type, symbol, price, prob_up,
                features_snapshot, est_time_str, local_time_str,
            )
            reply_markup = self._build_inline_keyboard(symbol)
            self._send_message_sync(text, parse_mode="MarkdownV2", reply_markup=reply_markup)
        except Exception:
            log.exception("Failed to send signal to Telegram")

    def _build_signal_message(
        self,
        signal_type: str,
        symbol: str,
        price: float,
        prob_up: float,
        features_snapshot: dict[str, float] | None,
        est_time_str: str,
        local_time_str: str,
    ) -> str:
        """构建 MarkdownV2 格式的信号消息。"""

        if signal_type.upper() == "BUY":
            emoji = "🟢"
            action = "买入信号 BUY"
        else:
            emoji = "🔴"
            action = "卖出信号 SELL"

        # 概率柱状图
        bar_len = 20
        filled = int(prob_up * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        # 主体消息
        lines = [
            f"{emoji} *{_escape_md2(action)}*",
            "",
            f"📌 *{_escape_md2(symbol)}*  `@` *${_escape_md2(f'{price:.2f}')}*",
            "",
            f"🎯 模型置信度: *{_escape_md2(f'{prob_up:.1%}')}*",
            f"`{_escape_md2(bar)}`",
        ]

        # 核心特征快照
        if features_snapshot:
            lines.append("")
            lines.append("📊 *核心指标:*")
            for key, val in features_snapshot.items():
                if isinstance(val, float):
                    lines.append(f"  • {_escape_md2(key)}: `{_escape_md2(f'{val:.4f}')}`")
                else:
                    lines.append(f"  • {_escape_md2(key)}: `{_escape_md2(str(val))}`")

        # 时间
        lines.append("")
        if est_time_str:
            lines.append(f"🕐 美东: `{_escape_md2(est_time_str)}`")
        if local_time_str:
            lines.append(f"🕐 本地: `{_escape_md2(local_time_str)}`")

        return "\n".join(lines)

    def _build_inline_keyboard(self, symbol: str) -> dict:
        """构建 InlineKeyboard 按钮。"""
        # 提取 ticker (去掉 .US / .HK 后缀)
        ticker = symbol.split(".")[0] if "." in symbol else symbol

        buttons = [
            [
                {
                    "text": "📈 打开 TradingView",
                    "url": f"https://www.tradingview.com/chart/?symbol={ticker}",
                },
                {
                    "text": f"🔕 忽略 {ticker} 1小时",
                    "callback_data": f"mute:{symbol}:3600",
                },
            ]
        ]
        return {"inline_keyboard": buttons}

    def _send_message_sync(
        self,
        text: str,
        parse_mode: str | None = "MarkdownV2",
        reply_markup: dict | None = None,
    ) -> bool:
        """
        底层发送方法 (同步, 带重试)。

        Returns:
            True if message was sent successfully.
        """
        url = f"{self._api_base}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(url, data=payload, timeout=self._timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        log.debug("Telegram message sent successfully")
                        return True
                    else:
                        log.warning(
                            "Telegram API error (attempt %d/%d): %s",
                            attempt, self._max_retries, data.get("description", "unknown"),
                        )
                else:
                    log.warning(
                        "Telegram HTTP %d (attempt %d/%d): %s",
                        resp.status_code, attempt, self._max_retries, resp.text[:200],
                    )
            except requests.Timeout:
                log.warning(
                    "Telegram timeout (attempt %d/%d, timeout=%ds)",
                    attempt, self._max_retries, self._timeout,
                )
            except requests.RequestException as e:
                log.warning(
                    "Telegram request error (attempt %d/%d): %s",
                    attempt, self._max_retries, e,
                )

            # 退避重试
            if attempt < self._max_retries:
                _time.sleep(attempt * 1.0)

        log.error("Failed to send Telegram message after %d attempts", self._max_retries)
        return False

    def shutdown(self) -> None:
        """关闭线程池。"""
        self._executor.shutdown(wait=False)
