"""
策略与回测引擎
================================

将 TSMixer + MC Dropout 输出的预测均值 mu 与不确定度 sigma 转换为持仓，
按手续费率与滑点模拟逐期回测，并计算方向正确率、年化收益、年化波动、
夏普比率、最大回撤等关键指标。

包含：
  - signal_to_position：根据 (mu, sigma) 生成持仓（不确定度反向缩放）
  - backtest：逐期模拟收益与交易成本，产出累计收益曲线
  - compute_metrics：计算回测核心指标
  - run_backtest：顶层入口，根据 cfg 配置串联上述流程

本文件自包含，仅依赖 numpy 与 pandas，可独立运行 `python strategy.py` 进行逻辑测试。
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. 信号转持仓
# =============================================================================
def signal_to_position(mu, sigma, conf_threshold=0.05, lam=2.0, max_position=1.0):
    """
    将预测均值 mu 与不确定度 sigma 转换为持仓（原始信号，未含风控）。

    规则：
      - 若 |mu| < conf_threshold：持仓 = 0（信号不够置信，平仓观望）
      - 否则：基础仓位 = min(|mu|/conf_threshold, 1.0) * max_position（过门限后
        迅速达到满仓，再按 max_position 放大杠杆），
        再除以不确定度缩放因子（sigma 反向缩放）：
        持仓 = sign(mu) * base_position / (1 + lam * sigma)
      - 持仓裁剪到 [-max_position, max_position] 防止极端值

    设计说明：
      原始公式 position = sign(mu)*|mu|/(1+lam*sigma) 在 |mu| 较小时（模型输出
      常集中在 0 附近）导致仓位极小，年化收益不足。改为归一化基础仓位后，
      |mu| >= conf_threshold 即接近满仓，不确定度仍做反向缩放。
      max_position > 1 时启用杠杆（已确认无未来函数后允许）。

    注意：本函数仅生成原始信号仓位，回撤控制与波动率目标缩放在 apply_risk_controls 中。

    参数:
      mu: 1D numpy 数组 (N,)，模型预测均值
      sigma: 1D numpy 数组 (N,)，模型预测标准差
      conf_threshold: 置信度门限，|mu| 低于此值不平仓
      lam: 不确定度反向缩放系数
      max_position: 持仓上限（>1 表示杠杆，如 5.0 表示 5 倍杠杆）

    返回:
      positions: 1D numpy 数组 (N,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    max_position = float(max_position)

    # 仓位随 |mu| 线性增长，软门限（不硬清零弱信号）。
    # 设计：|mu| 在 [0, saturation_mu] 区间线性映射到 [0, max_position]。
    # saturation_mu = 3*conf_threshold，信号达到此值即满仓。
    # 弱信号(|mu|<conf)仍保留小仓位，避免大量空仓平缓期。
    # 用 sigmoid 在 conf 附近做平滑过渡，而非硬门限清零。
    saturation_mu = conf_threshold * 3.0  # 信号强度达到 3*conf 时满仓
    # 线性映射：|mu|=0 → 仓位=0，|mu|=saturation_mu → 仓位=max_position
    base_position = np.clip(
        np.abs(mu) / saturation_mu,
        0.0, 1.0
    ) * max_position
    # 符号定方向，不确定度反向缩放
    positions = np.sign(mu) * base_position / (1.0 + lam * sigma)
    # 软门限：|mu| 远低于 conf 时仓位极小但非零，避免硬清零
    # 使用 sigmoid 软门限：|mu|=conf 时 weight=0.5，|mu|<<conf 时趋近0
    soft_gate = 1.0 / (1.0 + np.exp(-10.0 * (np.abs(mu) - conf_threshold) / conf_threshold))
    positions = positions * soft_gate
    # 裁剪到 [-max_position, max_position]，防止极端值
    positions = np.clip(positions, -max_position, max_position)
    return positions


def apply_risk_controls(positions, forward_returns,
                        max_leverage=5.0,
                        vol_target=0.15,
                        vol_window=20,
                        max_drawdown_threshold=0.15,
                        dd_scale=0.5,
                        periods_per_year=365):
    """
    对原始仓位应用风控，降低最大回撤。包含两项独立机制：

    1. 波动率目标缩放（Vol Targeting）：
       - 用过去 vol_window 期已实现收益的滚动 std 估计当期波动率
       - 目标年化波动 vol_target（如 0.15 = 15%）
       - 缩放因子 = 目标波动 / 实际波动（仅下缩放，不上缩放，防过度加杠杆）
       - 高波动期自动减仓，平滑收益曲线
       - 注意：波动率只用过去数据（shift(1)），无未来函数

    2. 回撤止损（Drawdown Control）：
       - 实时跟踪累计收益曲线的回撤
       - 当回撤超过 max_drawdown_threshold 时，按 dd_scale 比例减仓
       - 回撤越深，减仓越多（线性衰减到 0）
       - 防止亏损扩大，保护资本

    参数:
      positions: 原始持仓数组 (N,)
      forward_returns: 真实未来对数收益 (N,)，用于计算已实现波动与累计收益
      max_leverage: 持仓上限（杠杆）
      vol_target: 目标年化波动率（如 0.15）
      vol_window: 波动率估计滚动窗口（期）
      max_drawdown_threshold: 回撤阈值，超过此值开始减仓
      dd_scale: 回撤减仓强度（回撤每增加阈值的一倍，仓位乘以 1-dd_scale）
      periods_per_year: 年化周期数（日线=365，15m=35040），用于波动率年化

    返回:
      adjusted_positions: 风控后持仓数组 (N,)
    """
    positions = np.asarray(positions, dtype=np.float64).copy()
    forward_returns = np.asarray(forward_returns, dtype=np.float64)
    n = positions.shape[0]
    if n == 0:
        return positions

    # ---- 1. 波动率目标缩放（只用过去数据，无未来函数）----
    # 用滚动 std，按 sqrt(periods_per_year) 年化
    # 用 t 时刻之前（不含 t）的 vol_window 期收益估计波动率，shift(1) 防泄露
    ret_series = pd.Series(forward_returns)
    # rolling 窗口含当前点，因此先 shift(1) 使窗口变为 [t-vol_window, t-1]
    rolling_vol = ret_series.shift(1).rolling(vol_window, min_periods=1).std()
    annualized_vol = rolling_vol * np.sqrt(periods_per_year)
    # 缩放因子 = 目标波动 / 实际波动，仅下缩放（cap=1，不上缩放）
    vol_scale = np.where(
        annualized_vol > 1e-8,
        np.minimum(vol_target / annualized_vol, 1.0),
        1.0
    )
    # np.where 返回 ndarray，直接 nan_to_num（无需 .values）
    vol_scale = np.nan_to_num(np.asarray(vol_scale, dtype=np.float64), nan=1.0)

    # ---- 2. 回撤止损（实时跟踪，无未来函数）----
    # 逐期模拟累计收益，用"实际减仓后仓位"计算 equity，形成负反馈：
    #   回撤触发 → dd_scale 减小 → 实际仓位减小 → 亏损减小 → equity 回升
    #   → 回撤缩小 → dd_scale 恢复 → 仓位恢复（避免死锁）
    # 注意：t 期仓位用 t-1 期的 dd_scale（无未来函数），
    #       t 期 dd_scale 基于 t 期 equity（含 t 期收益），供 t+1 期使用
    # 最低保留仓位 min_dd_scale：即使深度回撤也保留少量仓位，让模型在有好信号时
    #   能重新建仓赚钱，从而恢复 equity，避免"永久空仓"死锁
    # 回撤恢复重置：当 drawdown 回到 -threshold/2 以内时，重置 running_max=cur_equity，
    #   使后续回撤从新基准计算，避免历史峰值永久压制
    pos_after_vol = positions * vol_scale
    equity = np.zeros(n)
    running_max = np.zeros(n)
    dd_scale_arr = np.ones(n)   # dd_scale_arr[i] 供 t+1 期仓位使用
    cur_equity = 0.0
    cur_max = 0.0
    dd_scale_prev = 1.0          # t-1 期的 dd_scale，用于 t 期实际仓位
    recovery_threshold = max_drawdown_threshold * 0.5  # 回撤恢复到一半阈值内时重置
    min_dd_scale = 0.15           # 最低保留 15% 仓位
    for i in range(n):
        # t 期实际仓位 = pos_after_vol[i] * (t-1 期的 dd_scale)
        actual_pos_i = pos_after_vol[i] * dd_scale_prev
        # t 期收益实现（仓位在 t 期确定，收益 t→t+1 实现，对齐无未来函数）
        cur_equity = cur_equity + actual_pos_i * forward_returns[i]
        equity[i] = cur_equity
        cur_max = max(cur_max, cur_equity)
        running_max[i] = cur_max
        drawdown = cur_equity - cur_max  # <= 0
        # 回撤恢复重置：drawdown 回到 -recovery_threshold 以内时，更新基准
        if drawdown > -recovery_threshold and cur_max > cur_equity:
            cur_max = cur_equity
        # 回撤超过阈值时减仓：回撤深度 / 阈值 决定减仓比例
        if drawdown < -max_drawdown_threshold:
            excess_dd = (-drawdown - max_drawdown_threshold) / max_drawdown_threshold
            attenuation = max(1.0 - dd_scale * (1.0 + excess_dd), min_dd_scale)
            dd_scale_arr[i] = attenuation
        else:
            dd_scale_arr[i] = 1.0
        # 更新 prev，供下一期使用
        dd_scale_prev = dd_scale_arr[i]

    # dd_scale_arr[i] 基于 t 期 equity，用于 t+1 期仓位 → shift(1) 防未来函数
    dd_scale_arr = np.concatenate([[1.0], dd_scale_arr[:-1]])  # shift(1)

    # ---- 3. 合并风控：仓位 = 原始仓位 * vol_scale * dd_scale ----
    adjusted_positions = positions * vol_scale * dd_scale_arr
    # 裁剪到杠杆上限
    adjusted_positions = np.clip(adjusted_positions, -max_leverage, max_leverage)
    return adjusted_positions


# =============================================================================
# 2. 回测引擎
# =============================================================================
def backtest(positions, forward_returns, fee_rate=0.0006, slippage=0.0005):
    """
    逐期模拟回测，扣除手续费与滑点。

    参数:
      positions: 1D 持仓数组 (N,)，positions[i] 基于 t 时刻信息
      forward_returns: 1D 未来对数收益率数组 (N,)，forward_returns[i] 为 t 时刻未来收益
      fee_rate: 单边手续费率（>0）
      slippage: 滑点（>0）

    逐期逻辑:
      - 当期毛收益 = positions[i] * forward_returns[i]
      - 换手 = |positions[i] - positions[i-1]|（i=0 时换手 = |positions[0]|）
      - 交易成本 = 换手 * (fee_rate + slippage)
      - 净收益 = 当期毛收益 - 交易成本
      - 累计收益曲线 = cumsum(净收益)（对数收益累加，概念上从 0 开始）

    返回:
      dict: net_returns / equity_curve / turnover / costs，均为长度 N 的 1D 数组
    """
    positions = np.asarray(positions, dtype=np.float64)
    forward_returns = np.asarray(forward_returns, dtype=np.float64)
    n = positions.shape[0]

    if n == 0:
        empty = np.array([], dtype=np.float64)
        return {'net_returns': empty, 'equity_curve': empty,
                'turnover': empty, 'costs': empty}

    # 换手：i=0 为 |positions[0]|（建仓），其余为相邻持仓差绝对值
    turnover = np.empty(n, dtype=np.float64)
    turnover[0] = np.abs(positions[0])
    if n > 1:
        turnover[1:] = np.abs(np.diff(positions))

    # 交易成本（换手 * (单边费率 + 滑点)）
    costs = turnover * (fee_rate + slippage)
    # 当期毛收益：持仓方向 * 未来收益
    gross_returns = positions * forward_returns
    # 净收益 = 毛收益 - 交易成本
    net_returns = gross_returns - costs
    # 累计收益曲线（对数收益累加，起始概念值为 0）
    equity_curve = np.cumsum(net_returns)

    return {
        'net_returns': net_returns,
        'equity_curve': equity_curve,
        'turnover': turnover,
        'costs': costs,
    }


# =============================================================================
# 3. 指标计算
# =============================================================================
def compute_metrics(net_returns, positions, forward_returns, periods_per_year=365):
    """
    计算回测核心指标。

    参数:
      net_returns: 回测净收益数组 (N,)
      positions: 持仓数组 (N,)
      forward_returns: 真实未来收益数组 (N,)
      periods_per_year: 年化周期数

    返回:
      dict 含: directional_accuracy, annual_return, annual_volatility,
              sharpe, max_drawdown
    """
    net_returns = np.asarray(net_returns, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    forward_returns = np.asarray(forward_returns, dtype=np.float64)

    # ---- 方向正确率：仅统计有意义的持仓样本（|持仓|>0.05）----
    # 软门限下弱信号也有微小仓位，需用较高阈值过滤噪声小仓位
    # |持仓|>0.05 相当于至少 5% 仓位，反映真实交易决策
    meaningful_threshold = 0.05
    meaningful_mask = np.abs(positions) > meaningful_threshold
    if not np.any(meaningful_mask):
        directional_accuracy = 0.0
    else:
        pos_sign = np.sign(positions[meaningful_mask])
        ret_sign = np.sign(forward_returns[meaningful_mask])
        directional_accuracy = float(np.mean(pos_sign == ret_sign))

    # ---- 年化收益与年化波动 ----
    mean_ret = float(np.mean(net_returns)) if net_returns.size > 0 else 0.0
    std_ret = float(np.std(net_returns)) if net_returns.size > 0 else 0.0
    annual_return = mean_ret * periods_per_year
    annual_volatility = std_ret * np.sqrt(periods_per_year)

    # ---- 夏普比率（无风险利率 = 0；波动为 0 时返回 0）----
    if annual_volatility == 0.0:
        sharpe = 0.0
    else:
        sharpe = annual_return / annual_volatility

    # ---- 最大回撤（基于 cumsum 累计收益曲线，返回正值）----
    if net_returns.size == 0:
        max_drawdown = 0.0
    else:
        equity = np.cumsum(net_returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max  # <= 0
        max_drawdown = float(-np.min(drawdown))  # 转为正值

    return {
        'directional_accuracy': directional_accuracy,
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
    }


# =============================================================================
# 4. 顶层入口
# =============================================================================
def run_backtest(mu, sigma, forward_returns, cfg):
    """
    顶层回测入口：读取 cfg['strategy'] 配置，串联
    signal_to_position → apply_risk_controls → backtest → compute_metrics。

    风控流程：
      1. signal_to_position 生成原始仓位（含杠杆、不确定度缩放、置信度门限）
      2. apply_risk_controls 应用波动率目标缩放 + 回撤止损（降低最大回撤）
      3. backtest 用风控后仓位模拟回测（扣手续费与滑点）
      4. compute_metrics 计算指标

    参数:
      mu, sigma: 模型预测均值与标准差 (N,)
      forward_returns: 真实未来对数收益 (N,)
      cfg: 配置字典，需含 cfg['strategy'] 子 dict

    返回:
      dict 含 5 个 metrics 字段 + equity_curve + positions + net_returns
            （附带 turnover / costs 便于分析）
    """
    s = cfg['strategy']
    # 1. 原始仓位（含杠杆）
    raw_positions = signal_to_position(
        mu, sigma,
        conf_threshold=s['conf_threshold'],
        lam=s['position_lambda'],
        max_position=s.get('max_leverage', 1.0),
    )

    # 2. 风控（波动率目标缩放 + 回撤止损），若启用
    use_risk_control = s.get('use_risk_control', True)
    if use_risk_control:
        positions = apply_risk_controls(
            raw_positions, forward_returns,
            max_leverage=s.get('max_leverage', 1.0),
            vol_target=s.get('vol_target', 0.15),
            vol_window=s.get('vol_window', 20),
            max_drawdown_threshold=s.get('max_drawdown_threshold', 0.15),
            dd_scale=s.get('dd_scale', 0.5),
            periods_per_year=s.get('periods_per_year', 365),
        )
    else:
        positions = raw_positions

    # 3. 回测
    bt = backtest(
        positions, forward_returns,
        fee_rate=s['fee_rate'],
        slippage=s['slippage'],
    )
    # 4. 指标
    metrics = compute_metrics(
        bt['net_returns'], positions, forward_returns,
        periods_per_year=s['periods_per_year'],
    )
    return {
        **metrics,
        'equity_curve': bt['equity_curve'],
        'positions': positions,
        'net_returns': bt['net_returns'],
        'turnover': bt['turnover'],
        'costs': bt['costs'],
    }


# =============================================================================
# 快速逻辑测试
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    N = 300

    # ---- 构造合成数据 ----
    # 真实未来收益：小幅度正态（贴近日线对数收益量级，便于 equity_curve 起始 ≈ 0）
    forward_returns = np.random.normal(0.0, 0.02, N)
    # mu 幅度：均匀 [0, 1]
    mu_mag = np.random.uniform(0.0, 1.0, N)
    # 约 60% 样本 mu 符号与 forward_returns 一致（构造有信号数据）
    correct_mask = np.random.rand(N) < 0.6
    fr_sign = np.sign(forward_returns)
    fr_sign = np.where(fr_sign == 0, 1.0, fr_sign)  # 边界保护
    mu_sign = np.where(correct_mask, fr_sign, -fr_sign)
    mu = mu_mag * mu_sign
    # sigma：均匀 [0, 0.5]
    sigma = np.random.uniform(0.0, 0.5, N)

    # 策略配置（与 config.yaml 一致）
    cfg = {
        'strategy': {
            'conf_threshold': 0.05,
            'position_lambda': 2.0,
            'fee_rate': 0.0006,
            'slippage': 0.0005,
            'periods_per_year': 365,
            'max_leverage': 5.0,
            'use_risk_control': True,
            'vol_target': 0.20,
            'vol_window': 20,
            'max_drawdown_threshold': 0.10,
            'dd_scale': 0.6,
        }
    }
    s = cfg['strategy']

    print("=" * 70)
    print("strategy.py 快速逻辑测试")
    print("=" * 70)

    results = []

    def check(name, cond, info=""):
        tag = "PASS" if cond else "FAIL"
        results.append(bool(cond))
        print(f"[{tag}] {name}  {info}")

    # ---- 1. signal_to_position 测试 ----
    positions = signal_to_position(mu, sigma, s['conf_threshold'], s['position_lambda'],
                                   max_position=s['max_leverage'])
    check("signal_to_position 形状为 (N,)", positions.shape == (N,),
          f"shape={positions.shape}")
    check("signal_to_position 值域 [-max_leverage, max_leverage]",
          float(np.min(positions)) >= -s['max_leverage'] and float(np.max(positions)) <= s['max_leverage'],
          f"min={np.min(positions):.4f}, max={np.max(positions):.4f}, max_leverage={s['max_leverage']}")
    low_conf_mask = np.abs(mu) < s['conf_threshold']
    if np.any(low_conf_mask):
        # 软门限：低置信样本持仓应较小（<10% max_position）
        low_conf_max = float(np.max(np.abs(positions[low_conf_mask])))
        low_conf_ok = low_conf_max < 0.10 * s['max_leverage']
        check("软门限：|mu|<conf 时持仓较小（<10% max_position）", low_conf_ok,
              f"低置信样本数={int(low_conf_mask.sum())}, 最大|持仓|={low_conf_max:.4f}")
    else:
        check("软门限：|mu|<conf 时持仓较小（<10% max_position）", True, "(无低置信样本)")
    # 杠杆生效检查：高置信样本(|mu|>=conf_threshold)持仓应能超过 1.0
    high_conf_mask = np.abs(mu) >= s['conf_threshold']
    if np.any(high_conf_mask):
        has_leverage = bool(np.any(np.abs(positions[high_conf_mask]) > 1.0))
        check("杠杆生效（高置信样本持仓 > 1.0）", has_leverage,
              f"高置信样本最大|持仓|={float(np.max(np.abs(positions[high_conf_mask]))):.4f}")
    else:
        check("杠杆生效（高置信样本持仓 > 1.0）", True, "(无高置信样本)")

    # ---- 2. backtest 测试 ----
    bt = backtest(positions, forward_returns, s['fee_rate'], s['slippage'])
    eq = bt['equity_curve']
    check("equity_curve 长度 = N", len(eq) == N, f"len={len(eq)}")
    check("equity_curve 起始 ≈ 0",
          abs(float(eq[0])) < 0.2,
          f"equity_curve[0]={float(eq[0]):.6f}")
    check("equity_curve == cumsum(net_returns)",
          bool(np.allclose(eq, np.cumsum(bt['net_returns']))), "")

    # ---- 3. compute_metrics 测试 ----
    metrics = compute_metrics(bt['net_returns'], positions, forward_returns,
                              s['periods_per_year'])
    expected_fields = {'directional_accuracy', 'annual_return',
                       'annual_volatility', 'sharpe', 'max_drawdown'}
    check("compute_metrics 返回 5 个字段",
          set(metrics.keys()) == expected_fields,
          f"fields={sorted(metrics.keys())}")

    # 手算验证夏普 = mean/std * sqrt(periods_per_year)
    nr = bt['net_returns']
    if np.std(nr) != 0:
        manual_sharpe = (np.mean(nr) / np.std(nr)) * np.sqrt(s['periods_per_year'])
    else:
        manual_sharpe = 0.0
    check("夏普 = mean/std*sqrt(periods_per_year)",
          bool(np.isclose(metrics['sharpe'], manual_sharpe)),
          f"metrics={metrics['sharpe']:.6f}, manual={manual_sharpe:.6f}")

    # 方向正确率应接近 0.6（构造的信号一致性），且 > 0.5 表示存在信号
    check("方向正确率合理（约 0.6，>0.5 表示有信号）",
          0.5 < metrics['directional_accuracy'] < 0.75,
          f"directional_accuracy={metrics['directional_accuracy']:.4f}")

    # ---- 4. 成本 > 0 ----
    check("fee_rate>0, slippage>0 时成本 > 0",
          float(np.sum(bt['costs'])) > 0.0,
          f"total_cost={float(np.sum(bt['costs'])):.6f}")

    # ---- 5. apply_risk_controls 测试（风控降回撤）----
    print("\n---- 风控测试（apply_risk_controls）----")
    adjusted = apply_risk_controls(
        positions, forward_returns,
        max_leverage=s['max_leverage'],
        vol_target=s['vol_target'],
        vol_window=s['vol_window'],
        max_drawdown_threshold=s['max_drawdown_threshold'],
        dd_scale=s['dd_scale'],
    )
    check("apply_risk_controls 输出形状一致", adjusted.shape == positions.shape,
          f"shape={adjusted.shape}")
    check("风控后 |持仓| ≤ max_leverage",
          float(np.max(np.abs(adjusted))) <= s['max_leverage'] + 1e-9,
          f"max|pos|={float(np.max(np.abs(adjusted))):.4f}")
    # 风控后回测
    bt_rc = backtest(adjusted, forward_returns, s['fee_rate'], s['slippage'])
    metrics_rc = compute_metrics(bt_rc['net_returns'], adjusted, forward_returns,
                                 s['periods_per_year'])
    check("风控后最大回撤 ≤ 风控前最大回撤",
          metrics_rc['max_drawdown'] <= metrics['max_drawdown'] + 1e-9,
          f"前={metrics['max_drawdown']:.4f}, 后={metrics_rc['max_drawdown']:.4f}")
    print(f"  风控前: 回撤={metrics['max_drawdown']:.4f}, 夏普={metrics['sharpe']:.4f}, "
          f"年化波动={metrics['annual_volatility']:.4f}")
    print(f"  风控后: 回撤={metrics_rc['max_drawdown']:.4f}, 夏普={metrics_rc['sharpe']:.4f}, "
          f"年化波动={metrics_rc['annual_volatility']:.4f}")

    # ---- 6. run_backtest 顶层入口（启用风控）----
    rb = run_backtest(mu, sigma, forward_returns, cfg)
    required_keys = ['directional_accuracy', 'annual_return',
                     'annual_volatility', 'sharpe', 'max_drawdown',
                     'equity_curve', 'positions', 'net_returns']
    check("run_backtest 含 metrics + equity_curve + positions + net_returns",
          all(k in rb for k in required_keys),
          f"keys={sorted(rb.keys())}")
    check("run_backtest 启用风控后回撤 ≤ 关闭风控",
          rb['max_drawdown'] <= metrics['max_drawdown'] + 1e-9,
          f"风控后={rb['max_drawdown']:.4f}, 风控前={metrics['max_drawdown']:.4f}")

    # ---- 汇总 ----
    print("-" * 70)
    print("指标汇总：")
    summary = pd.DataFrame({
        '指标': ['方向正确率', '年化收益', '年化波动', '夏普比率', '最大回撤'],
        '值': [metrics['directional_accuracy'], metrics['annual_return'],
               metrics['annual_volatility'], metrics['sharpe'],
               metrics['max_drawdown']],
    })
    print(summary.to_string(index=False))

    print("-" * 70)
    n_pass = int(sum(results))
    n_total = len(results)
    print(f"测试结果：{n_pass}/{n_total} 通过")
    print("总体：" + ("PASS ✓" if all(results) else "FAIL ✗"))
