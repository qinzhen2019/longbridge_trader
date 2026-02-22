# ML 策略指南

本指南介绍如何使用机器学习策略增强交易决策。

## 策略对比

| 特性 | XGBoost | 强化学习 (PPO/DQN) |
|------|---------|-------------------|
| **类型** | 监督学习 | 强化学习 |
| **原理** | 预测未来涨跌概率 | Agent 自主学习买卖动作 |
| **输出** | 概率值 (0-1) | 直接动作 (Buy/Sell/Hold) |
| **可解释性** | 高 (特征重要性) | 低 (黑盒) |
| **训练数据** | 需要标签 (涨/跌) | 只需价格序列 |
| **适用场景** | 趋势明确的市场 | 复杂多变的市场 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements-ml.txt

# macOS 需要额外安装 OpenMP
brew install libomp
```

### 2. 配置策略

编辑 `.env`：

```bash
# 选择策略
STRATEGY_TYPE=xgboost    # 或 rl

# XGBoost 参数
XGB_BUY_THRESHOLD=0.6    # 上涨概率 ≥ 60% 才买入
XGB_SELL_THRESHOLD=0.4   # 上涨概率 ≤ 40% 才卖出

# RL 参数
RL_ALGO=PPO              # PPO | DQN | A2C
ML_MODEL_NAME=xgb_model  # 模型文件名
```

### 3. 训练模型

**方式一：Dashboard 面板**

```
python dashboard.py
→ 选择 [8. ML 策略管理]
→ 选择 [4. 训练 XGBoost 模型] 或 [5. 训练 RL 模型]
```

**方式二：命令行**

```bash
# XGBoost
python ml/train_xgb.py --symbols TSLA.US,AAPL.US --klines 500

# RL
python ml/train_rl.py --symbols TSLA.US --algo PPO --timesteps 50000
```

### 4. 启动交易

```bash
python dashboard.py
→ 选择 [7. 启动自动交易引擎]
```

---

## XGBoost 策略详解

### 工作原理

```
历史K线 → 特征工程 (15维) → XGBoost 模型 → 上涨概率
                                        ↓
                              概率 ≥ 阈值? → 买入信号
```

### 特征向量 (15维)

| 类别 | 特征 | 说明 |
|------|------|------|
| **动量** | `roc_1`, `roc_3`, `roc_5`, `roc_10` | 多周期价格变化率 |
| **波动** | `volatility_20` | 近20日波动率 |
| **超买超卖** | `rsi_norm` | RSI 归一化 [0,1] |
| **位置** | `boll_position`, `boll_width` | 布林带相对位置和宽度 |
| **趋势** | `ema_deviation`, `ma5_ma20_ratio` | EMA偏离度、均线比率 |
| **区间** | `hl_position_20` | 近20日高低点位置 |
| **量能** | `volume_ratio_20` | 成交量相对均值 |
| **滞后** | `lag_return_1/2/3` | 滞后收益率 |

### 信号逻辑

```python
prob_up = model.predict(features)

if prob_up >= 0.6:    # 买入阈值
    signal = BUY
elif prob_up <= 0.4:  # 卖出阈值
    signal = SELL
else:
    signal = HOLD
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | TSLA.US | 训练标的，多个用逗号分隔 |
| `--klines` | 500 | 日线数量 |
| `--horizon` | 5 | 预测未来几天涨跌 |
| `--estimators` | 200 | 决策树数量 |
| `--depth` | 6 | 树的最大深度 |

---

## 强化学习策略详解

### 工作原理

```
┌─────────────────────────────────────────────────┐
│                   RL 交易环境                    │
├─────────────────────────────────────────────────┤
│  状态 (Observation): 17维向量                   │
│    - 15维技术指标特征                           │
│    - 1维持仓状态 (0/1)                          │
│    - 1维未实现盈亏                              │
├─────────────────────────────────────────────────┤
│  动作 (Action):                                 │
│    0 = Hold (观望)                              │
│    1 = Buy (买入)                               │
│    2 = Sell (卖出)                              │
├─────────────────────────────────────────────────┤
│  奖励 (Reward):                                 │
│    - 卖出时的已实现盈亏 (占初始资金百分比)       │
│    - 持仓时的小额浮动收益激励                   │
│    - 回撤超10%的惩罚                            │
└─────────────────────────────────────────────────┘
```

### 算法选择

| 算法 | 特点 | 推荐场景 |
|------|------|----------|
| **PPO** | 稳定、收敛快 | 默认选择 |
| **DQN** | 离散动作空间 | 简单策略 |
| **A2C** | 并行训练快 | 多环境训练 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | PPO | 算法选择 |
| `--timesteps` | 50000 | 训练步数 |
| `--lr` | 3e-4 | 学习率 |

**建议**：PPO 至少 50,000 步，复杂场景可能需要 100,000+ 步。

---

## 回测评估

### Dashboard 回测

```
python dashboard.py
→ 选择 [8. ML 策略管理]
→ 选择 [6. 回测评估 XGBoost] 或 [7. 回测评估 RL]
```

### 命令行回测

```bash
python ml/backtest.py --symbol TSLA.US --model-type xgboost --test-ratio 0.2
```

### 关键指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **总收益** | 策略收益率 | > 买入持有 |
| **最大回撤** | 峰值到谷底最大跌幅 | < 20% |
| **Sharpe Ratio** | 风险调整后收益 | > 1.0 |
| **胜率** | 盈利交易占比 | > 50% |
| **盈亏比** | 总盈利/总亏损 | > 1.5 |

---

## 最佳实践

### 数据准备

- 至少 300 根日线用于训练
- 使用多支股票联合训练提高泛化能力
- 定期重新训练适应市场变化

### 模型管理

```bash
# 建议每周重新训练
crontab -e
# 每周六凌晨 2 点训练
0 2 * * 6 cd /path/to/longbridge_trader && python ml/train_xgb.py --symbols TSLA.US,AAPL.US
```

### 风险控制

1. **先用模拟盘** - `PAPER_TRADING=true`
2. **小仓位测试** - 降低 `MAX_POSITION_VALUE`
3. **严格止损** - 设置合理的 `STOP_LOSS_PCT`
4. **监控回撤** - 关注 `MAX_DRAWDOWN_PCT` 触发情况

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 模型预测全 HOLD | 检查阈值设置，降低 `XGB_BUY_THRESHOLD` |
| 训练不收敛 | 增加训练步数，检查数据质量 |
| macOS 报错 libomp | `brew install libomp` |
| 回测收益为负 | 调整特征、增加训练数据、尝试不同算法 |

---

## 文件结构

```
ml/
├── feature_engineer.py  # 特征工程 (15维向量)
├── xgb_model.py         # XGBoost 模型管理
├── xgb_strategy.py      # XGBoost 策略实现
├── train_xgb.py         # XGBoost 训练脚本
├── rl_agent.py          # RL Agent 管理
├── rl_strategy.py       # RL 策略实现
├── trading_env.py       # Gymnasium 交易环境
├── train_rl.py          # RL 训练脚本
└── backtest.py          # 回测评估引擎
```

---

## ⚠️ 重要提示

1. **ML 预测基于历史数据，不保证未来表现**
2. **过拟合风险**：避免在单一标的上过度训练
3. **市场非平稳**：定期重训练是必要的
4. **建议组合使用**：ML 信号 + 规则策略 + 风控约束
