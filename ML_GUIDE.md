# 📖 ML 策略使用指南

本文档介绍如何使用 Longbridge Trader 中的 **机器学习策略引擎** — 包括 **XGBoost 监督学习**和 **PPO/DQN 强化学习**。

---

## 🏗 架构总览

```
┌──────────────────────────────────────────┐
│            TradingEngine (main.py)        │
│              ↓ _create_strategy()        │
├──────────┬──────────────┬────────────────┤
│ 布林带RSI │  XGBoost ML  │  RL (PPO/DQN) │
│ (默认)    │  监督学习     │  强化学习      │
└──────────┴──────────────┴────────────────┘
         ↑                      ↑
    strategy_base.py         strategy_base.py
    (Signal / SignalResult / BaseStrategy)

┌──────────────────────────────────────────┐
│           dashboard.py (交互面板)         │
│  策略配置 │ 模型管理 │ 预测关注清单       │
│  依赖检测 │ 训练模型 │ 特征向量预览       │
└──────────────────────────────────────────┘
```

通过 `.env` 中的 `STRATEGY_TYPE` 变量在三种策略之间一键切换，无需修改代码。

| 配置值 | 策略 | 说明 |
|--------|------|------|
| `bollinger_rsi` | 布林带 + RSI | 规则型策略（默认，向后兼容） |
| `xgboost` | XGBoost | 预测上涨概率 > 阈值时开仓 |
| `rl` | Stable-Baselines3 | AI Agent 自主决策 Buy/Sell/Hold |

---

## ⚡ 快速开始

### 1. 安装 ML 依赖

基础依赖（布林带策略）不需要额外安装。使用 ML 策略前需安装：

```bash
pip install -r requirements-ml.txt
```

这将安装 `scikit-learn`, `xgboost`, `gymnasium`, `stable-baselines3`, `pandas` 等库。

### 2. 配置 `.env`

在你的 `.env` 文件末尾添加以下配置（参考 `.env.example`）：

```bash
# 策略类型: bollinger_rsi | xgboost | rl
STRATEGY_TYPE=xgboost

# XGBoost 参数
XGB_BUY_THRESHOLD=0.6       # 上涨概率超过此值才买入
XGB_SELL_THRESHOLD=0.4       # 下跌概率超过此值才卖出

# RL 参数
RL_ALGO=PPO                  # PPO | DQN | A2C
ML_MODEL_NAME=xgb_model      # 模型文件名 (不含扩展名)
```

---

## 📈 方向一：XGBoost 监督学习

### 原理

XGBoost 策略使用**历史技术指标特征**训练一个二分类模型，预测"未来 N 根日线收盘价是否上涨"。在实盘中，模型输出一个 `prob_up`（上涨概率），仅当概率超过阈值（默认 60%）时才允许买入。

### 特征 (15维)

| # | 特征 | 说明 |
|---|------|------|
| 1-4 | ROC_1, ROC_3, ROC_5, ROC_10 | 多周期价格变化率 |
| 5 | volatility_20 | 近20根K线波动率 |
| 6 | rsi_norm | RSI 归一化值 [0,1] |
| 7-8 | boll_position, boll_width | 布林带相对位置和带宽 |
| 9 | ema_deviation | 价格对 EMA 的偏离度 |
| 10 | ma5_ma20_ratio | 短期均线 vs 长期均线比率 |
| 11 | hl_position_20 | 近20根K线高低点位置 |
| 12 | volume_ratio_20 | 成交量相对均值比率 |
| 13-15 | lag_return_1/2/3 | 滞后收益率 |

### 训练模型

```bash
python ml/train_xgb.py \
  --symbols TSLA.US,AAPL.US,NVDA.US,META.US \
  --klines 500 \
  --horizon 5 \
  --estimators 200 \
  --depth 6 \
  --model-name xgb_model
```

**参数说明：**
- `--symbols`: 用于训练的股票（多支可增加泛化能力）
- `--klines`: 拉取最近多少根日线（默认 500）
- `--horizon`: 预测未来多少天的涨跌（默认 5 天）
- `--estimators`: XGBoost 树的数量
- `--model-name`: 模型保存名称

训练完成后，模型保存在 `models/xgb_model.json`。

### 启动交易

```bash
# .env 设置
STRATEGY_TYPE=xgboost
ML_MODEL_NAME=xgb_model

# 启动
python dashboard.py
```

### 信号逻辑

```
买入条件: prob_up >= XGB_BUY_THRESHOLD (默认 0.6)
卖出条件: prob_up <= XGB_SELL_THRESHOLD (默认 0.4) 或 RSI 超买
```

---

## 🖥 Dashboard 面板集成

`dashboard.py` 已全面集成 ML 功能，无需手动执行命令行操作：

### 主菜单

启动面板后，标题栏自动显示当前策略类型和模式：

```
╔════════════════════════════════════════════════════╗
║            长桥交易助手 - 交互式面板                ║
╠════════════════════════════════════════════════════╣
║  策略: 🤖 XGBoost (buy≥0.60, sell≤0.40)           ║
║  模式: 实盘                                        ║
╠════════════════════════════════════════════════════╣
║  ...                                               ║
║  8. ML 策略管理                                    ║
║  9. ML 预测关注清单 (未来5日涨跌概率)              ║
║  0. 退出                                           ║
╚════════════════════════════════════════════════════╝
```

### 菜单 8: ML 策略管理

| 子菜单 | 功能 |
|--------|------|
| 查看当前策略配置 | 显示策略类型、阈值参数、15维特征清单 |
| 查看模型状态 | 检查 XGBoost/RL 模型文件是否存在、大小、修改时间 |
| 检测 ML 依赖库 | 探测 xgboost / stable-baselines3 / sklearn / numpy / libomp |
| 训练 XGBoost 模型 | 交互式输入训练参数，直接调用 `ml/train_xgb.py` |
| 训练 RL 模型 | 交互式输入训练参数，直接调用 `ml/train_rl.py` |

### 菜单 9: ML 预测关注清单

使用训练好的 XGBoost 模型预测关注清单中所有美股的 **未来5个交易日上涨/下跌概率**：

1. 自动加载 `models/xgb_model.json`
2. 从长桥关注清单获取所有美股标的
3. 逐个拉取日线 K 线，构建 15 维特征向量
4. 模型推理 → 输出 `prob_up` 和 `prob_down`
5. 按上涨概率排序，展示预测排行榜

信号判定规则：

| 条件 | 信号 |
|------|------|
| `prob_up ≥ XGB_BUY_THRESHOLD` (60%) | 🟢 买入 |
| `prob_up ≤ XGB_SELL_THRESHOLD` (40%) | 🔴 卖出 |
| 其他 | ⚪ 观望 |

### 股票分析中的 ML 特征预览 (菜单 1)

分析个股时，如果当前策略为 `xgboost` 或 `rl`，会自动追加 ML 特征向量预览：

```
┌─ ML 特征向量 (XGBoost)
│  roc_1: +0.0032              roc_3: -0.0115
│  rsi_norm: +0.4200           boll_position: +0.3500
│  volatility_20: +0.0210      ema_deviation: -0.0080
│  ...
└──────────────────────────────────
```

### 启动自动交易引擎增强 (菜单 7)

启动引擎前，确认信息中增加：
- 策略引擎类型和参数
- 模型文件状态（存在/缺失/大小/修改时间）
- 模型缺失时给出提醒

---

## 🤖 方向二：强化学习 (Stable-Baselines3)

### 原理

RL 策略将交易过程建模为一个 **马尔可夫决策过程 (MDP)**：
- **状态 (Observation)**: 15维技术指标 + 持仓状态 + 未实现盈亏 = 17维向量
- **动作 (Action)**: `Hold(0)` / `Buy(1)` / `Sell(2)`
- **奖励 (Reward)**: 基于已实现 PnL（作为初始资金的百分比），回撤超10%施加惩罚

Agent 在历史数据"交易环境"中反复试错学习，自动找到最优的买卖策略。

### 训练 Agent

```bash
python ml/train_rl.py \
  --symbols TSLA.US \
  --klines 500 \
  --timesteps 50000 \
  --algo PPO \
  --model-name rl_model
```

**参数说明：**
- `--algo`: 选择算法 `PPO`（推荐）、`DQN` 或 `A2C`
- `--timesteps`: 训练总步数（越大越好，但耗时更长）
- `--lr`: 学习率（默认 3e-4）

训练完成后，模型保存在 `models/rl_model_PPO.zip`。

### 启动交易

```bash
# .env 设置
STRATEGY_TYPE=rl
RL_ALGO=PPO
ML_MODEL_NAME=rl_model

# 启动
python dashboard.py
```

---

## 🔧 进阶用法

### 定期重新训练

建议使用 cron 或定时任务，定期（如每周末）重新训练模型以适应最新的市场状态：

```bash
# crontab -e
# 每周六凌晨 2 点重新训练 XGBoost
0 2 * * 6 cd /path/to/longbridge_trader && python ml/train_xgb.py --symbols TSLA.US,AAPL.US --klines 500
```

### 回到默认策略

如果 ML 策略表现不如预期，随时可以切换回原有策略：

```bash
STRATEGY_TYPE=bollinger_rsi
```

无需卸载任何 ML 依赖，系统完全向后兼容。

---

## 📂 文件结构

```
longbridge_trader/
├── dashboard.py               # 交互式面板 (含 ML 管理/预测)
├── main.py                    # 交易引擎主循环
├── strategy_base.py           # 统一策略接口 (Signal/SignalResult/BaseStrategy)
├── strategy.py                # 布林带+RSI 规则策略
├── ml/
│   ├── __init__.py
│   ├── feature_engineer.py    # 特征工程 (15维)
│   ├── xgb_model.py           # XGBoost 模型管理
│   ├── xgb_strategy.py        # XGBoost 策略
│   ├── train_xgb.py           # XGBoost 训练脚本
│   ├── trading_env.py         # Gymnasium 交易环境
│   ├── rl_agent.py            # RL Agent 管理
│   ├── rl_strategy.py         # RL 策略
│   └── train_rl.py            # RL 训练脚本
├── models/                    # 训练后的模型 (已 gitignore)
├── requirements-ml.txt        # ML 依赖
└── .env.example               # 配置示例 (含 ML 参数)
```

---

## ⚠️ 注意事项

1. **先用模拟盘测试**: 设置 `PAPER_TRADING=true`，确认 ML 策略正常运行后再考虑实盘。
2. **样本量**: XGBoost 建议至少使用 300+ 根日线 K 线训练，信号才有统计意义。
3. **RL 训练步数**: PPO 建议至少 50,000 步，复杂场景可能需要 100,000+ 步才能收敛。
4. **过拟合风险**: 尽量用多支股票联合训练，提高泛化能力。
5. **模型更新**: 金融市场是非平稳的（non-stationary），定期重训练是必要的。
6. **macOS 用户**: XGBoost 需要 OpenMP 运行时: `brew install libomp`。
7. **预测仅供参考**: ML 预测基于历史数据，不构成投资建议。
