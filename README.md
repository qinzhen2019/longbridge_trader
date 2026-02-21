# Longbridge Trader 📈

Longbridge Trader 是一款基于 [长桥 OpenAPI (Longport)](https://open.longportapp.com/) 的自动化量化交易机器人。它专为具有特定交易策略（如布林带结合 RSI 的均值回归和趋势跟踪）的用户设计。通过异步架构 (Asyncio) 提供高效的数据拉取与处理，并具有非常严密的风险管理系统，支持实盘与模拟（Paper Trading）交易模式。

## ✨ 核心特性

- **多因子交易策略**: 内置了基于 **Bollinger Bands (布林带)** 和 **RSI (相对强弱指数)** 的交易策略，同时结合 **EMA 趋势过滤**（Trend Filter），防止在明确的下跌趋势中盲目抄底。
- **完备的风控管理 (Risk Control)**:
  - **硬止损 & 止盈 (Stop Loss & Take Profit)**: 按百分比实时监控仓位，触达立即市价清仓。
  - **最大回撤保护 (Max Drawdown)**: 监控每日账户总资产回撤，一旦超出预设阈值（如3%），立即清空当日所有未成交订单并停止当天交易。
  - **时间冷却 (Cooldown)**: 交易频率控制，避免在一只票上短时间内频繁来回摩擦。
  - **仓位管理**: 限制总敞口 (Max Total Exposure)、单次开仓最大金额 (Max Position Value) 及最大持仓数量 (Max Positions)。
- **交互式操作面板 (Dashboard)**: 提供了一个完善的终端 UI，支持：
  - 手动买卖下单（市价单/限价单）
  - 查看并一键撤销账户的所有未成交订单
  - 实时分析特定股票或扫描自选股列表
  - 在面板内一键启动/停止自动交易引擎
- **实时与异步架构**: 基于 Python `asyncio`，并发请求和处理多支股票历史数据，并通过 WebSocket 实时订阅最新的 Tick 级的行情报价以及账户订单变更推送。
- **动态自选股监控**: 项目支持配置监控固定的股票池，也支持基于指定市场（如美股、港股）自动拉取长桥账户的自选股列表进行动态监控。

## 🏗 代码架构介绍

- `dashboard.py`: 推荐的项目入口点，提供集成的终端用户界面。包含行情分析模块和交易引擎的启动。
- `main.py`: 核心运行引擎 (`TradingEngine`)。控制整个 Async 事件循环，并协调各模块的数据流动。
- `config.py`: 配置模块。利用 `dataclass` 从环境与 `.env` 中安全读取及解析各种交易参数。
- `strategy.py`: 信号生成引擎 (`BollingerRsiStrategy`)。负责根据当前指标数据与持仓状态运算并产生发出 买入 (BUY)、卖出 (SELL) 或 观望 (HOLD) 信号。
- `risk_control.py`: 风控模块。在开仓前校验剩余额度、风控状态，并在运行中实时监控是否触发止损 / 止盈及历史回撤。
- `order_executor.py`: 执行器。封装了长桥官方的 `TradeContext`，负责底层订单真正地向交易所提交、取消操作，以及资产和持仓的同步。
- `indicators.py` *(未展示详情)*: 负责依靠获取到的历史 K 线，计算 BB/RSI/EMA 等技术指标。

## 🚀 快速开始

### 1. 环境准备

确保你已安装 Python 3.9 或更高版本。

```bash
# 克隆你自己的代码库
git clone https://github.com/你的用户名/longbridge_trader.git
cd longbridge_trader

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境参数

复制示例的配置文件来创建你自己的环境配置文件：

```bash
cp .env.example .env
```

打开 `.env` 文件，输入你的长桥（Longport）API 凭证，以及所需的交易参数。其中：
* `LONGPORT_APP_KEY`, `LONGPORT_APP_SECRET`, `LONGPORT_ACCESS_TOKEN`: [在长桥开放平台申请](https://open.longbridgeapp.com/)。
* `PAPER_TRADING`: 设为 `true` 则使用模拟交易，即仅打印包含 `[PAPER]` 的日志，不产生真实的买卖。设为 `false` 则为实盘。
* `WATCH_SYMBOLS`: 你想要坚守监听的股票代码，如 `700.HK,TSLA.US`。

### 3. 一键启动

推荐使用交互式仪表盘启动：
```bash
python dashboard.py
```

或者跳过仪表盘，直接启动纯后台自动交易引擎：
```bash
python main.py
```

## ⚠️ 风险提示及免责声明
本项目基于个人编程与量化学术研究目的开源和分享。
**使用本项目连接实盘进行交易可能产生真实的财务损失！** 作者不对使用此代码造成的任何个人投资盈亏负责，请务必在完全阅读并理解项目策略且经过充分 **Paper Trading** 验证后再考虑投入真实资本。

## 🤝 贡献与支持
如果你发现了任何 Bug 或有提升策略的建议，非常欢迎提交 [Pull Request](https://github.com/qinzhen2019/longbridge_trader/pulls) 或提交 [Issue](https://github.com/qinzhen2019/longbridge_trader/issues)。
