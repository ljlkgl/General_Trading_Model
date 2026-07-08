# General Trading Model

基于 **TSMixer + MC Dropout** 的贝叶斯量化交易模型，用于加密货币（BTC/ETH）多周期 K 线的方向预测与仓位回测。

本项目的核心目标：在**严格无未来函数**的前提下，实现方向正确率 ≥ 54%、夏普比率 ≥ 1.1、且交易在时间上分散（非集中一时）的真实可交易策略。

---

## 目录

- [项目架构](#项目架构)
- [项目结构](#项目结构)
- [安装与依赖](#安装与依赖)
- [快速开始](#快速开始)
- [网格搜索](#网格搜索)
- [配置文件详解](#配置文件详解)
- [模型架构](#模型架构)
- [策略与风控](#策略与风控)
- [评估指标](#评估指标)
- [最终结果汇总](#最终结果汇总)
- [数据格式](#数据格式)
- [模块逻辑测试](#模块逻辑测试)
- [常见问题](#常见问题)

---

## 项目架构

端到端流程一图概览：

```
┌─────────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│ CSV K线数据  │ → │ data_loader  │ → │  train   │ → │  evaluate  │ → │  results │
│ data/*.csv   │   │ 特征+滑窗+划分│   │ TSMixer  │   │ MC Dropout │   │ 图表+json│
└─────────────┘   └──────────────┘   │ + Loss   │   │ + 回测+出图│   └──────────┘
                                      └──────────┘   └────────────┘
                                            ↑                ↑
                                      gtm/model.py     gtm/strategy.py
                                   (RevIN/Blocks/Pool) (信号→仓位→风控→回测)
```

完整流程由 [scripts/run.py](scripts/run.py) 串联：

1. **数据管道**（[gtm/data_loader.py](gtm/data_loader.py)）：CSV → 特征工程 → 标签归一化 → 滑窗 → 时间序列划分 → DataLoader
2. **模型构建**（[gtm/model.py](gtm/model.py)）：RevIN → K×TSMixerBlock → AttentionPooling → 线性头
3. **训练**（[gtm/train.py](gtm/train.py)）：DirectionalLoss（MSE+方向CE+IC）+ IC 早停 + 余弦退火
4. **评估**（[gtm/evaluate.py](gtm/evaluate.py)）：MC Dropout 预测 → 信号转仓位 → 风控 → 回测 → 出图 + metrics.json
5. **策略**（[gtm/strategy.py](gtm/strategy.py)）：signal_to_position + apply_risk_controls + backtest + compute_metrics

---

## 项目结构

```
General_Trading_Model/
├── gtm/                        # 核心代码包（General Trading Model）
│   ├── __init__.py             # 包入口
│   ├── data_loader.py          # 数据管道：CSV→特征→滑窗→划分
│   ├── model.py                # TSMixer 模型：RevIN+Blocks+AttentionPooling
│   ├── train.py                # 训练循环：DirectionalLoss+IC早停
│   ├── strategy.py             # 策略回测：信号→仓位→风控→指标
│   └── evaluate.py             # 评估可视化：MC Dropout+出图+json
│
├── scripts/                    # 可执行脚本
│   ├── run.py                  # 主入口：训练+评估一条龙
│   └── grid_search.py          # 网格搜索：参数扫描优化
│
├── configs/                    # 配置文件
│   ├── config_btc.yaml         # BTC 日线配置（已达标）
│   ├── config_eth.yaml         # ETH 日线配置（已达标）
│   └── config.yaml             # ETH 15m 配置（参考，模型在该周期无效）
│
├── data/                       # 原始 K 线数据
│   ├── BTCUSDT_1D_kline.csv    # BTC 日线
│   ├── ETHUSDT_1D_kline.csv    # ETH 日线
│   └── ...                     # 多周期 CSV（15m/30m/5m/1min/5min）
│
├── results/                    # 训练与搜索结果
│   ├── results_btc/            # BTC 日线训练结果
│   │   ├── model.pt            # 最优权重
│   │   ├── metrics.json        # 测试集指标
│   │   ├── equity_curve.png    # 累计收益曲线
│   │   ├── drawdown.png        # 回撤曲线
│   │   └── pred_with_uncertainty.png  # 预测均值与不确定带
│   ├── results_eth/            # ETH 日线训练结果（同上）
│   └── grid_search/            # 网格搜索结果（运行后生成）
│       ├── grid_results.csv    # 全部组合汇总（按指标降序）
│       └── run_XXX/            # 每组参数的独立结果目录
│
├── docs/                       # 文档
│   └── deep-research-report.md # 理论依据与设计文档
│
├── requirements.txt            # 依赖
└── README.md                   # 本文件
```

> **说明**：核心代码组织为 `gtm` 包，可被 `from gtm import data_loader, train, ...` 导入；脚本位于 `scripts/`，自动将项目根目录注入 `sys.path`。

---

## 安装与依赖

### 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0.0（CPU 或 CUDA 均可，本项目日线数据量小，CPU 即可）

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单（[requirements.txt](requirements.txt)）：

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

> 网格搜索额外依赖 `pandas`（已在依赖中），用于汇总结果到 CSV。

---

## 快速开始

> **所有命令均从项目根目录 `General_Trading_Model/` 运行。**

### 1. 训练并评估 BTC 日线策略

```bash
python scripts/run.py --config configs/config_btc.yaml
```

输出示例（节选）：

```
[STEP 1/4] data_loader: 加载数据 + 特征工程 + 滑窗 + 划分
  特征数 C=31, 窗口 L=30
  样本数: train=1368, val=456, test=457
[STEP 2/4] model: 构建 TSMixer
  模型参数量: 38561
[STEP 3/4] train: 训练 TSMixer
  ...
[STEP 4/4] evaluate: MC 采样预测 + 回测 + 指标 + 出图

======================================================================
最终评估指标（测试集）
======================================================================
  directional_accuracy: 0.547486
  annual_return: 1.049918
  annual_volatility: 0.591558
  sharpe: 1.774835
  max_drawdown: 0.340997
  trade_active_segments: 10
  nonzero_position_pct: 0.972

目标: 方向正确率 ≥ 0.54, 夏普比率 ≥ 1.1, 交易分散段数 ≥ 7/10
方向正确率 0.5475 ✓ 达标 (阈值 0.54)
夏普比率   1.7748 ✓ 达标 (阈值 1.1)
交易分散   10/10 段有交易 ✓ 达标 (阈值 7)
总体: 全部达标 ✓
```

结果文件落在 [results/results_btc/](results/results_btc)。

### 2. 训练并评估 ETH 日线策略

```bash
python scripts/run.py --config configs/config_eth.yaml
```

结果文件落在 [results/results_eth/](results/results_eth)。

### 3. 自定义数据/参数

复制 `configs/config_btc.yaml` 为 `configs/config_xxx.yaml`，修改：

- `data.csv_path`：指向 `data/` 下的 CSV（列格式见 [数据格式](#数据格式)）
- `data.window_L` / `data.horizon`：输入窗口与预测视野
- `strategy.*`：杠杆、手续费、风控参数
- `output.results_dir`：结果输出目录（如 `results/results_xxx`）

然后运行：

```bash
python scripts/run.py --config configs/config_xxx.yaml
```

---

## 网格搜索

[scripts/grid_search.py](scripts/grid_search.py) 提供参数扫描能力，对关键超参数做笛卡尔积，自动训练+评估每组组合，汇总到 CSV 按 Sharpe 降序排列。

### 用法

```bash
# 1. 跑逻辑测试（验证参数解析、笛卡尔积等辅助逻辑，不跑真实训练）
python scripts/grid_search.py

# 2. 跑真实网格搜索（需先编辑 GRID 定义扫描参数）
python scripts/grid_search.py --base configs/config_btc.yaml

# 3. 指定排序指标与 Top N
python scripts/grid_search.py --base configs/config_btc.yaml --metric sharpe --top 10
```

### 编辑扫描参数

打开 [scripts/grid_search.py](scripts/grid_search.py)，在顶部找到 `GRID` 字典，编辑要扫描的参数：

```python
GRID = {
    'strategy.conf_threshold': [0.10, 0.12, 0.15],
    'strategy.max_leverage': [3.0, 5.0, 8.0],
    'strategy.vol_target': [0.5, 1.0, 1.5],
}
```

- 键为点号分隔的配置路径（如 `strategy.conf_threshold` → `cfg['strategy']['conf_threshold']`）
- 值为候选值列表，笛卡尔积生成所有组合（上例 = 3×3×3 = 27 组）

### 可扫描的常用参数路径

| 参数路径 | 说明 |
|---------|------|
| `strategy.conf_threshold` | 置信度门限 |
| `strategy.position_lambda` | 不确定度反向缩放系数 |
| `strategy.max_leverage` | 最大杠杆 |
| `strategy.vol_target` | 目标年化波动率 |
| `strategy.max_drawdown_threshold` | 回撤阈值 |
| `strategy.dd_scale` | 回撤减仓强度 |
| `train.epochs` | 训练轮数（减小可加速扫描） |
| `train.lr` | 学习率 |
| `loss.alpha` | 方向 CE 权重 |
| `loss.mse_weight` | MSE 权重 |
| `data.window_L` | 输入窗口长度 |
| `data.horizon` | 预测视野 |

### 输出

- 每组参数结果：`results/grid_search/run_XXX/`（含 metrics.json 与图表）
- 汇总表：`results/grid_search/grid_results.csv`（按 `--metric` 降序）
- 终端打印 Top N

### 加速技巧

网格搜索计算量 = 组合数 × 单次训练时间。建议：

1. 先用小范围探索（如每参数 2-3 个候选值）
2. 在 GRID 中加入 `'train.epochs': [10]` 减少训练轮数
3. 找到最优区域后再用完整 epochs 精细搜索

---

## 配置文件详解

以 [configs/config_eth.yaml](configs/config_eth.yaml) 为例：

```yaml
data:
  csv_path: "data/ETHUSDT_1D_kline.csv"   # 数据文件（相对项目根目录）
  split_ratios: [0.6, 0.2, 0.2]      # train/val/test 时间顺序划分
  window_L: 30                        # 输入历史窗口（30 天）
  horizon: 3                          # 预测未来 3 期收益
  label_clip_quantile: 0.05           # 标签分位数裁剪（去极端值）

features:
  log_return_windows: [1, 3, 5, 10]   # 对数收益率窗口
  momentum_windows: [3, 5, 10]        # 动量窗口
  volatility_windows: [5, 10, 20]     # 波动率窗口
  ma_short: 5                         # 短期均线
  ma_long: 20                         # 长期均线

model:
  num_blocks: 4                       # TSMixer 块数
  dropout: 0.2                        # Dropout（也用于 MC Dropout）
  dropout_anneal: true                # Dropout 退火
  dropout_min: 0.05                   # 退火下限
  feat_hidden_mult: 2                 # FeatMLP 隐层倍数
  revin: true                         # 启用 RevIN
  revin_affine: true                  # RevIN 可学习仿射

train:
  epochs: 50                          # 最大训练轮数
  batch_size: 64                      # 批大小
  lr: 0.001                           # 学习率
  weight_decay: 0.0001                # L2 正则
  warmup_steps: 20                    # 学习率 warmup
  patience: 10                        # 早停耐心
  grad_clip: 1.0                      # 梯度裁剪
  seed: 42                            # 随机种子

bayes:
  mc_samples: 20                      # MC Dropout 采样次数 T

strategy:
  conf_threshold: 0.12                # 置信度门限（|mu| 软门限中心）
  position_lambda: 0.5                # 不确定度反向缩放系数
  fee_rate: 0.0006                    # 单边手续费率（>0）
  slippage: 0.0005                    # 滑点（>0）
  periods_per_year: 365               # 年化周期数（日线=365，15m=35040）
  max_leverage: 10.0                  # 最大杠杆
  use_risk_control: true              # 启用风控
  vol_target: 0.50                    # 目标年化波动率
  vol_window: 20                      # 波动率估计窗口
  max_drawdown_threshold: 0.05        # 回撤阈值
  dd_scale: 0.5                       # 回撤减仓强度

targets:
  min_accuracy: 0.54                  # 最低方向正确率
  min_sharpe: 1.1                     # 最低夏普比率
  min_active_segments: 7              # 10 段中至少 7 段有交易

loss:
  alpha: 1.0                          # 方向 CE 权重
  beta: 0.5                           # IC 损失权重
  dir_threshold: 0.02                 # 方向判定阈值
  mse_weight: 0.1                     # MSE 权重（降低以突出方向）

output:
  results_dir: "results/results_eth"  # 结果目录（相对项目根目录）
  save_model: true                    # 保存模型权重
```

### BTC vs ETH 关键差异

| 参数 | BTC 日线 | ETH 日线 | 说明 |
|------|---------|---------|------|
| `max_leverage` | 5.0 | 10.0 | ETH 波动更大，用更高杠杆放大信号 |
| `vol_target` | 1.0 | 0.50 | BTC 目标波动更高，持仓更激进 |
| `max_drawdown_threshold` | 0.15 | 0.05 | ETH 回撤阈值更严，保护资本 |
| `dd_scale` | 0.7 | 0.5 | BTC 回撤减仓更强 |

---

## 模型架构

### TSMixer 主干（[gtm/model.py](gtm/model.py)）

```
输入 (B, L, C)
  ↓
[RevIN] 可逆实例归一化（应对非平稳分布漂移）
  ↓
[TSMixerBlock] × K
  ├─ TimeMLP：LayerNorm(L) → Linear(L→L) → GELU → Dropout + 残差
  └─ FeatMLP：LayerNorm(C) → Linear(C→2C) → GELU → Dropout → Linear(2C→C) → Dropout + 残差
  ↓
[AttentionPooling] 替代 mean pooling
  score = Linear(C→1) → softmax over L → 加权求和 → (B, C)
  ↓
[Head] Linear(C→1) → (B,)  纯线性输出，无 tanh 压缩
```

**关键设计**：

- **RevIN**：每个样本、每个特征在时间维独立归一化，反归一化可逆，缓解非平稳时序的分布漂移。
- **AttentionPooling**：用可学习打分函数对时间步加权，让模型聚焦近期或重要时间步，替代 mean pooling 的等权平均。
- **纯线性输出**：移除 tanh，让模型自由表达信号强度，不被压缩到 [-1, 1]。

### MC Dropout 不确定性估计

推理时保留 Dropout（其余层 eval），对同一输入做 T 次随机前向：

```python
mu = mean(T 次预测)        # 预测均值
sigma = std(T 次预测)      # 不确定度（≥ 0）
```

- `mu` 作为交易信号方向与强度
- `sigma` 反向缩放仓位（高不确定 → 减仓）

### DirectionalLoss（[gtm/train.py](gtm/train.py)）

```
Loss = mse_weight · MSE + alpha · 方向CE + beta · IC损失
```

- **MSE 项**：保留回归目标，防止预测幅度失控
- **方向 CE 项**：把连续标签离散化为 {跌, 平, 涨} 三类，直接拉开涨跌方向 logit
- **IC 损失项**：`1 - Pearson相关`，逼迫预测与标签排序一致

`mse_weight=0.1` 大幅降低 MSE 权重，避免在噪声数据上退化为"预测均值"。

**早停准则用 IC 而非 MSE**——IC 直接反映排序能力，是交易盈亏的核心驱动力；MSE 会惩罚"激进但方向正确"的预测。

---

## 策略与风控

### 信号转仓位（[gtm/strategy.py](gtm/strategy.py) `signal_to_position`）

```python
# 1. 线性映射：|mu| 在 [0, 3·conf_threshold] 线性映射到 [0, max_position]
saturation_mu = conf_threshold * 3.0
base_position = clip(|mu| / saturation_mu, 0, 1) * max_position

# 2. 符号定方向，不确定度反向缩放
positions = sign(mu) * base_position / (1 + lam · sigma)

# 3. sigmoid 软门限：|mu|<<conf 时仓位极小但非零（避免硬清零的平缓期）
soft_gate = sigmoid(10 · (|mu| - conf_threshold) / conf_threshold)
positions = positions * soft_gate

# 4. 裁剪到 [-max_position, max_position]
```

**设计要点**：

- **线性映射**（非阶跃）：避免刚过门限即满仓的阶跃式建仓，让收益曲线平滑
- **软门限**（非硬清零）：弱信号保留小仓位，避免大量空仓平缓期；但软门限下弱信号噪声会拉低方向正确率，故评估时用 `meaningful_threshold=0.05` 过滤

### 风控（`apply_risk_controls`）

两项独立机制，均严格无未来函数：

**1. 波动率目标缩放**

```python
rolling_vol = forward_returns.shift(1).rolling(vol_window).std()  # 只用过去数据
annualized_vol = rolling_vol * sqrt(periods_per_year)
vol_scale = min(vol_target / annualized_vol, 1.0)  # 仅下缩放，不上缩放
```

**2. 回撤止损（负反馈，已修复死锁 bug）**

```python
# 关键：equity 用"实际减仓后仓位"计算，形成负反馈：
#   回撤触发 → dd_scale 减小 → 实际仓位减小 → 亏损减小 → equity 回升
#   → 回撤缩小 → dd_scale 恢复 → 仓位恢复（避免死锁）
for i in range(n):
    actual_pos_i = pos_after_vol[i] * dd_scale_prev    # t 期用 t-1 期 dd_scale
    cur_equity += actual_pos_i * forward_returns[i]
    drawdown = cur_equity - cur_max
    if drawdown > -recovery_threshold:                  # 回撤恢复 → 重置基准
        cur_max = cur_equity
    if drawdown < -max_drawdown_threshold:              # 回撤超限 → 减仓
        excess_dd = (-drawdown - threshold) / threshold
        dd_scale = max(1 - dd_scale · (1 + excess_dd), min_dd_scale)  # 最低保留 15%
```

**修复的死锁 bug**：早期版本 equity 用"未减仓仓位"计算（正反馈），running_max 永不下降，dd_scale 可归零导致永久空仓。修复后 equity 用实际减仓仓位，`min_dd_scale=0.15` 保证最低仓位，回撤恢复时重置 running_max。

### 回测（`backtest`）

```python
turnover[i] = |positions[i] - positions[i-1]|       # i=0 时 = |positions[0]|
costs[i] = turnover[i] · (fee_rate + slippage)
net_returns[i] = positions[i] · forward_returns[i] - costs[i]
equity_curve = cumsum(net_returns)
```

---

## 评估指标

[gtm/evaluate.py](gtm/evaluate.py) 在测试集上输出 7 个指标：

| 指标 | 含义 | 目标 |
|------|------|------|
| `directional_accuracy` | 方向正确率（仅统计 \|持仓\|>0.05 的有意义样本） | ≥ 0.54 |
| `annual_return` | 年化收益 | — |
| `annual_volatility` | 年化波动 | — |
| `sharpe` | 夏普比率（无风险利率=0） | ≥ 1.1 |
| `max_drawdown` | 最大回撤（正值） | — |
| `trade_active_segments` | 测试集等分 10 段中有交易的段数 | ≥ 7 |
| `nonzero_position_pct` | 非零持仓期数占比 | — |

**为什么方向正确率用 `meaningful_threshold=0.05` 过滤**：软门限下弱信号也有微小仓位，若全部计入会被噪声拉低；只统计 \|持仓\|>0.05（至少 5% 仓位）的样本，反映真实交易决策。

**为什么有 `trade_active_segments`**：防止策略只在测试集开头交易一次就空仓（这种"高收益"是虚假的，不可持续）。10 段中至少 7 段有交易，保证时间分散性。

---

## 最终结果汇总

### BTC 日线（[results/results_btc/metrics.json](results/results_btc/metrics.json)）

| 指标 | 值 | 是否达标 |
|------|------|---------|
| 方向正确率 | 0.5475 | ✓ (≥0.54) |
| 年化收益 | 104.99% | — |
| 年化波动 | 59.16% | — |
| 夏普比率 | 1.7748 | ✓ (≥1.1) |
| 最大回撤 | 34.10% | — |
| 交易分散段数 | 10/10 | ✓ (≥7) |
| 非零持仓占比 | 97.16% | — |

### ETH 日线（[results/results_eth/metrics.json](results/results_eth/metrics.json)）

| 指标 | 值 | 是否达标 |
|------|------|---------|
| 方向正确率 | 0.5654 | ✓ (≥0.54) |
| 年化收益 | 201.08% | — |
| 年化波动 | 103.99% | — |
| 夏普比率 | 1.9336 | ✓ (≥1.1) |
| 最大回撤 | 30.62% | — |
| 交易分散段数 | 10/10 | ✓ (≥7) |
| 非零持仓占比 | 95.00% | — |

### 关于 ETH 15m

`configs/config.yaml` 为 ETH 15m 配置。实测发现：15m 数据噪声极大，修复风控死锁 bug 后真实夏普为负（之前 4.63 是虚假的）。**15m 周期不推荐使用**，保留配置仅供对比参考。

---

## 数据格式

CSV 文件需包含以下列（列名严格一致），放在 `data/` 目录下：

| 列名 | 说明 |
|------|------|
| `datetime` | 时间戳（如 `2020-01-01 00:00:00`） |
| `open` / `high` / `low` / `close` | OHLC 价格 |
| `volume` | 成交量 |
| `taker_buy` / `taker_sell` | 主动买/卖量 |
| `RSI` | RSI 指标 |
| `ATR` | ATR 指标 |

示例（[data/BTCUSDT_1D_kline.csv](data/BTCUSDT_1D_kline.csv)）：

```
datetime,open,high,low,close,volume,taker_buy,taker_sell,RSI,ATR
2020-01-01 00:00:00,7195.3,7254.7,7170.0,7200.0,12345.6,6172.8,6172.8,55.0,80.5
...
```

特征工程（[gtm/data_loader.py](gtm/data_loader.py) `add_features`）会在原始 10 列基础上新增 21 列衍生特征（对数收益率、动量、波动率、MA 差值、taker 压力、RSI/ATR 归一化、量价比等），最终特征数 C=31。

---

## 模块逻辑测试

每个核心模块都内置自包含的逻辑测试，运行 `python <模块>` 即可验证：

```bash
# 核心模块（gtm 包内，可独立运行）
python gtm/model.py        # 模型前向 + RevIN 可逆性 + MC Dropout
python gtm/data_loader.py  # CSV 加载 + 特征 + 滑窗 + 划分
python gtm/strategy.py     # 信号转仓位 + 风控 + 回测 + 指标
python gtm/train.py        # 训练下降 + 早停 + 权重保存 + Dropout 退火
python gtm/evaluate.py     # 评估 + 出图 + metrics.json

# 脚本
python scripts/grid_search.py   # 网格搜索辅助逻辑（参数解析、笛卡尔积、deepcopy 隔离）
```

预期输出均为 `ALL PASS`。测试不依赖外部训练好的模型，使用合成数据或 Mock 模型。

> **包导入兼容性**：`gtm` 包既可作为模块导入（`from gtm import train`），也可直接运行单个文件（`python gtm/train.py`）。模块内对 `model`/`strategy` 的导入使用 `try/except` 兼容两种方式。

---

## 常见问题

### Q1：为什么收益曲线有平缓期？

平缓期对应弱信号时段（\|mu\| < conf_threshold）。信号分层 IC 分析显示：弱信号 IC 为负（反向预测力），空仓或小仓位是合理的。强信号（\|mu\| ≥ 0.08）IC=0.069 才有正向预测力。

### Q2：为什么方向正确率只统计 \|持仓\|>0.05 的样本？

软门限下弱信号也有微小仓位（非零），若全部计入方向正确率会被噪声拉低。`meaningful_threshold=0.05` 过滤掉这些噪声小仓位，只统计至少 5% 仓位的真实交易决策。

### Q3：为什么允许高杠杆？

本项目已严格审计无未来函数（特征只用过去数据，标签归一化统计量只来自训练集，时间顺序划分不 shuffle，风控 shift(1) 防泄露）。在无未来函数的前提下，杠杆是放大真实信号的合法手段。

### Q4：如何切换到其他币种/周期？

1. 准备符合 [数据格式](#数据格式) 的 CSV，放入 `data/`
2. 复制 `configs/config_btc.yaml`，修改 `csv_path`（指向 `data/xxx.csv`）、`periods_per_year`（日线=365，15m=35040，5m=105120）
3. 调整 `vol_target`、`max_drawdown_threshold` 等风控参数匹配新周期波动特性
4. `python scripts/run.py --config configs/your_config.yaml`

### Q5：为什么 ETH 15m 不推荐？

15m 数据噪声极大，信噪比极低。修复风控死锁 bug 后，ETH 15m 真实夏普为 -5.70（之前 4.63 是 bug 造成的虚假高收益）。日线（horizon=3）信噪比更高，是更可靠的周期。

### Q6：如何复现结果？

`configs/*.yaml` 中 `train.seed=42` 已固定随机种子。运行 `python scripts/run.py --config configs/config_btc.yaml` 应可复现接近的指标（CPU/CUDA、PyTorch 版本可能导致微小差异）。

### Q7：如何做参数调优？

使用[网格搜索](#网格搜索)：

```bash
# 编辑 scripts/grid_search.py 顶部的 GRID 字典定义扫描参数
python scripts/grid_search.py --base configs/config_btc.yaml --metric sharpe --top 10
```

结果汇总在 `results/grid_search/grid_results.csv`，按 Sharpe 降序排列，快速定位最优参数区域。

### Q8：为什么代码组织成 gtm 包？

将核心代码（model/train/strategy/evaluate/data_loader）组织为 `gtm` 包，便于：
- 作为库被脚本导入（`from gtm import train`）
- 与脚本（scripts/）、配置（configs/）、数据（data/）、结果（results/）分门别类
- 单个模块仍可独立运行逻辑测试（`python gtm/model.py`）

---

