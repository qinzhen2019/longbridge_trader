# Longbridge Trader

基于长桥 OpenAPI 的量化交易机器人，支持规则策略与机器学习策略。

## 核心特性

| 特性 | 说明 |
|------|------|
| **多策略引擎** | 布林带+RSI 规则策略 / XGBoost 监督学习 / PPO-DQN 强化学习，一键切换 |
| **完备风控** | 止损止盈、最大回撤保护、仓位限制、交易冷却 |
| **异步架构** | asyncio + WebSocket 实时行情，高效并发 |
| **交互面板** | 终端 Dashboard，支持手动交易、策略管理、模型训练 |
| **模拟盘支持** | Paper Trading 模式安全验证策略 |

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard (交互面板)                  │
├─────────────────────────────────────────────────────────┤
│                   TradingEngine (main.py)               │
│         异步事件循环 + WebSocket 行情订阅                │
├──────────────┬──────────────┬───────────────────────────┤
│   Strategy   │ RiskControl  │      OrderExecutor        │
│   策略引擎    │   风控模块    │        订单执行           │
├──────────────┴──────────────┴───────────────────────────┤
│                     ML 策略层 (可选)                     │
│      XGBoost (监督学习)  │  PPO/DQN (强化学习)          │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装

```bash
git clone https://github.com/你的用户名/longbridge_trader.git
cd longbridge_trader
pip install -r requirements.txt

# 如需使用 ML 策略
pip install -r requirements-ml.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入长桥 API 凭证：

```bash
# 必填 - 在 https://open.longportapp.com 申请
LONGPORT_APP_KEY=your_key
LONGPORT_APP_SECRET=your_secret
LONGPORT_ACCESS_TOKEN=your_token

# 策略选择: bollinger_rsi | xgboost | rl
STRATEGY_TYPE=bollinger_rsi

# 模拟盘 (建议先测试)
PAPER_TRADING=true
```

### 3. 运行

```bash
# 交互面板 (推荐)
python dashboard.py

# 或直接启动交易引擎
python main.py
```

## 策略说明

### 规则策略 (默认)

布林带 + RSI 均值回归策略：

| 信号 | 条件 |
|------|------|
| **买入** | 价格触及布林下轨 + RSI 超卖 (<30) + 价格在 EMA 上方 |
| **卖出** | 价格回到布林中轨 / RSI 超买 (>70) / 触及上轨 |

### ML 策略

详见 [ML_GUIDE.md](ML_GUIDE.md)

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **XGBoost** | 预测未来 N 日涨跌概率 | 有足够历史数据，追求可解释性 |
| **RL (PPO/DQN)** | Agent 自主学习买卖决策 | 复杂市场环境，追求自适应 |

## 风控机制

```
┌─────────────────────────────────────────────┐
│                  风控检查链                  │
├─────────────────────────────────────────────┤
│  1. 交易是否被暂停? (最大回撤触发)           │
│  2. 是否在冷却期? (防止频繁交易)             │
│  3. 是否已有持仓? (单标的不重复开仓)         │
│  4. 持仓数量是否超限? (max_positions)        │
│  5. 总敞口是否超限? (max_total_exposure)     │
├─────────────────────────────────────────────┤
│  实时监控: 止损 / 止盈 / 日内最大回撤        │
└─────────────────────────────────────────────┘
```

关键参数 (`.env` 配置)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `STOP_LOSS_PCT` | 1.5% | 单笔止损阈值 |
| `TAKE_PROFIT_PCT` | 1.5% | 单笔止盈阈值 |
| `MAX_DRAWDOWN_PCT` | 3.0% | 日内最大回撤，触发后停止交易 |
| `MAX_POSITIONS` | 10 | 最大持仓数量 |
| `MAX_POSITION_VALUE` | 1000 | 单笔最大金额 |

## 文件结构

```
longbridge_trader/
├── dashboard.py          # 交互面板入口
├── main.py               # 交易引擎
├── config.py             # 配置管理
├── strategy.py           # 规则策略
├── strategy_base.py      # 策略基类
├── indicators.py         # 技术指标
├── risk_control.py       # 风控模块
├── order_executor.py     # 订单执行
├── ml/                   # ML 策略模块
│   ├── feature_engineer.py
│   ├── xgb_strategy.py
│   ├── rl_strategy.py
│   ├── trading_env.py
│   └── backtest.py
└── models/               # 训练后的模型
```

## Dashboard 功能

| 菜单 | 功能 |
|------|------|
| 1. 分析股票 | 查看技术指标、ML 特征向量、建议买卖点位 |
| 2. 查看持仓 | 显示当前持仓明细 |
| 3. 查看现金 | 账户余额、购买力 |
| 4. 扫描关注清单 | 智能评分推荐买入标的 |
| 5. 手动交易 | 市价/限价买卖 |
| 6. 撤销订单 | 查看并撤销未成交订单 |
| 7. 启动引擎 | 启动自动交易 |
| 8. ML 管理 | 训练模型、查看状态、回测评估 |
| 9. ML 预测 | 预测关注清单涨跌概率 |

## ⚠️ 风险提示

**本项目仅供学习研究，实盘交易可能导致资金损失。**

- 使用前请充分理解策略逻辑
- 建议先在模拟盘 (`PAPER_TRADING=true`) 验证
- 作者不对任何投资损失负责

## License

MIT
