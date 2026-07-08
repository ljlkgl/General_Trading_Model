"""
评估与可视化模块
================================

将 TSMixer + MC Dropout 的预测结果接入策略回测，并产出可视化图表与指标 JSON。

包含：
  - evaluate：在测试集上 MC Dropout 预测 → 回测 → 汇总指标与曲线
  - plot_equity_curve：绘制累计收益曲线
  - plot_drawdown：绘制回撤曲线（填充下方）
  - plot_pred_with_uncertainty：绘制预测均值与不确定带
  - save_metrics：保存指标为 JSON（中文不乱码）
  - run_evaluation：顶层入口，串联评估 + 出图 + 落盘

本文件自包含，可独立运行 `python evaluate.py` 进行逻辑测试。
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# 用 Agg 后端避免显示问题（无 GUI 环境也能保存图片）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 尝试启用中文字体，避免中文标题/标注渲染为缺字方块（找不到则忽略）
for _font in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']:
    try:
        matplotlib.rcParams['font.sans-serif'] = [_font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

try:
    from .model import mc_dropout_predict   # 作为 gtm 包成员导入
    from .strategy import run_backtest
except ImportError:
    from model import mc_dropout_predict    # 作为脚本直接运行
    from strategy import run_backtest


# =============================================================================
# 1. 评估主函数
# =============================================================================
def evaluate(model, test_loader, test_forward_returns, cfg, device, timestamps=None):
    """
    在测试集上用 MC Dropout 预测，接入回测，汇总指标与曲线。

    参数:
      model: 已训练的 TSMixer 模型
      test_loader: 测试集 DataLoader（不 shuffle），按顺序产出 (x, y) 批
      test_forward_returns: 1D numpy 数组，与 test_loader 顺序对齐的原始未来对数收益率
      cfg: 配置字典，需含 cfg['bayes']['mc_samples'] 与 cfg['strategy']
      device: torch.device
      timestamps: 可选，与测试样本对齐的时间戳列表/数组

    返回:
      dict: {
        'metrics': 指标 dict（directional_accuracy, annual_return, annual_volatility,
                  sharpe, max_drawdown）,
        'mu': 预测均值数组 (N,),
        'sigma': 不确定度数组 (N,),
        'forward_returns': 原始未来收益数组 (N,),
        'positions': 持仓数组 (N,),
        'equity_curve': 累计收益曲线 (N,),
        'net_returns': 净收益数组 (N,),
        'timestamps': 时间戳（可选）,
      }
    """
    mc_samples = cfg['bayes']['mc_samples']

    # ---- 逐批 MC Dropout 预测，拼接为全长 mu / sigma ----
    mu_list, sigma_list = [], []
    for xb, _yb in test_loader:
        xb = xb.to(device)
        mu_b, sigma_b = mc_dropout_predict(model, xb, T=mc_samples)
        mu_list.append(mu_b.detach().cpu())
        sigma_list.append(sigma_b.detach().cpu())

    mu = torch.cat(mu_list).numpy().astype(np.float64)
    sigma = torch.cat(sigma_list).numpy().astype(np.float64)

    # ---- 与原始未来收益对齐（取最小长度，防止尾部不对齐）----
    forward_returns = np.asarray(test_forward_returns, dtype=np.float64).reshape(-1)
    n = min(len(mu), len(forward_returns))
    mu = mu[:n]
    sigma = sigma[:n]
    forward_returns = forward_returns[:n]
    if timestamps is not None:
        timestamps = list(timestamps)[:n]

    # ---- 回测：mu / sigma / 原始未来收益 → 持仓 / 净收益 / 累计曲线 / 指标 ----
    bt = run_backtest(mu, sigma, forward_returns, cfg)

    # ---- 交易时间分散性指标 ----
    # 把测试集等分 10 段，统计有多少段存在非零持仓（避免交易集中在一时）
    positions = bt['positions']
    n = len(positions)
    seg_size = max(1, n // 10)
    active_segs = 0
    for i in range(10):
        s, e = i * seg_size, (i + 1) * seg_size if i < 9 else n
        if np.sum(np.abs(positions[s:e]) > 1e-6) > 0:
            active_segs += 1
    # 非零持仓期数占比
    nonzero_pct = float(np.mean(np.abs(positions) > 1e-6))

    metrics = {
        'directional_accuracy': float(bt['directional_accuracy']),
        'annual_return': float(bt['annual_return']),
        'annual_volatility': float(bt['annual_volatility']),
        'sharpe': float(bt['sharpe']),
        'max_drawdown': float(bt['max_drawdown']),
        'trade_active_segments': int(active_segs),   # 10段中有交易时段数
        'nonzero_position_pct': nonzero_pct,          # 非零持仓占比
    }

    return {
        'metrics': metrics,
        'mu': mu,
        'sigma': sigma,
        'forward_returns': forward_returns,
        'positions': bt['positions'],
        'equity_curve': bt['equity_curve'],
        'net_returns': bt['net_returns'],
        'turnover': bt['turnover'],
        'timestamps': timestamps,
    }


# =============================================================================
# 2. 绘制累计收益曲线
# =============================================================================
def plot_equity_curve(equity_curve, save_path, timestamps=None):
    """
    绘制累计收益曲线（equity_curve），保存到 save_path。

    - 横轴为时间（若提供 timestamps 且长度匹配）或样本序号
    - 含中文标题、网格、合理标注
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64).reshape(-1)
    use_ts = timestamps is not None and len(timestamps) == len(equity_curve)
    x = list(timestamps) if use_ts else np.arange(len(equity_curve))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, equity_curve, color='steelblue', linewidth=1.5, label='累计收益')
    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('时间' if use_ts else '样本序号')
    ax.set_ylabel('累计收益（对数收益累加）')
    ax.set_title('累计收益曲线（Equity Curve）')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =============================================================================
# 3. 绘制回撤曲线
# =============================================================================
def plot_drawdown(equity_curve, save_path, timestamps=None):
    """
    计算回撤曲线 drawdown = equity_curve - running_max(equity_curve)（均 ≤ 0），
    绘制并填充下方，保存到 save_path。
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64).reshape(-1)
    if equity_curve.size == 0:
        running_max = equity_curve
    else:
        running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max  # ≤ 0

    use_ts = timestamps is not None and len(timestamps) == len(drawdown)
    x = list(timestamps) if use_ts else np.arange(len(drawdown))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, drawdown, 0.0, color='salmon', alpha=0.5, label='回撤')
    ax.plot(x, drawdown, color='red', linewidth=1.0)
    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('时间' if use_ts else '样本序号')
    ax.set_ylabel('回撤')
    ax.set_title('回撤曲线（Drawdown）')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =============================================================================
# 4. 绘制预测均值与不确定带
# =============================================================================
def plot_pred_with_uncertainty(mu, sigma, save_path, timestamps=None, y_true=None):
    """
    绘制预测均值 mu 曲线，并用 fill_between 画 mu±sigma 的不确定带。
    可选叠加真实标签 y_true（若提供）。保存到 save_path。
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)

    use_ts = timestamps is not None and len(timestamps) == len(mu)
    x = list(timestamps) if use_ts else np.arange(len(mu))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mu, color='steelblue', linewidth=1.2, label='预测均值 mu')
    ax.fill_between(x, mu - sigma, mu + sigma,
                    color='steelblue', alpha=0.2, label='mu ± sigma')
    if y_true is not None:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        ax.plot(x, y_true, color='orange', linewidth=1.0, alpha=0.7,
                label='真实标签 y')
    ax.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_xlabel('时间' if use_ts else '样本序号')
    ax.set_ylabel('预测值')
    ax.set_title('预测均值与不确定带（MC Dropout）')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =============================================================================
# 5. 保存指标为 JSON
# =============================================================================
def save_metrics(metrics, save_path):
    """
    把 metrics dict 保存为 JSON 文件（含中文不乱码，ensure_ascii=False，indent=2）。
    """
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# =============================================================================
# 6. 顶层评估入口
# =============================================================================
def run_evaluation(model, test_loader, test_forward_returns, cfg, device, results_dir):
    """
    顶层函数：调用 evaluate → 生成 3 张图 → 保存 metrics.json。

    - 创建 results_dir 目录（若不存在）
    - 产出：equity_curve.png, drawdown.png, pred_with_uncertainty.png, metrics.json
    - 返回 evaluate 的结果 dict
    """
    os.makedirs(results_dir, exist_ok=True)

    result = evaluate(model, test_loader, test_forward_returns, cfg, device)

    equity = result['equity_curve']
    mu = result['mu']
    sigma = result['sigma']
    ts = result.get('timestamps')

    plot_equity_curve(equity, os.path.join(results_dir, 'equity_curve.png'),
                      timestamps=ts)
    plot_drawdown(equity, os.path.join(results_dir, 'drawdown.png'),
                  timestamps=ts)
    plot_pred_with_uncertainty(mu, sigma,
                               os.path.join(results_dir, 'pred_with_uncertainty.png'),
                               timestamps=ts)
    save_metrics(result['metrics'], os.path.join(results_dir, 'metrics.json'))

    return result


# =============================================================================
# 合成 Mock 模型（仅供逻辑测试使用）
# =============================================================================
class _MockModel(torch.nn.Module):
    """
    合成 Mock 模型：忽略输入，返回预构造的 base_preds + 小高斯噪声。

    - 用于在逻辑测试中产生可控的 mu（与 forward_returns 有约 58% 符号一致性）
    - 每次前向加小噪声，使 MC Dropout 的 T 次采样能产生 sigma > 0
    - 内含一个未接入 forward 的 Dropout 层，使 mc_dropout_predict 的
      enable_dropout 仍能找到 Dropout 模块而不报错
    """

    def __init__(self, base_preds, noise_std=0.02):
        super().__init__()
        self.register_buffer('base_preds',
                             torch.as_tensor(base_preds, dtype=torch.float32))
        self.noise_std = float(noise_std)
        # 未接入 forward，仅为了让 enable_dropout 找到 Dropout 模块
        self.drop = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        B = x.shape[0]
        preds = self.base_preds[:B]
        noise = torch.randn(B, dtype=preds.dtype) * self.noise_std
        return preds + noise


# =============================================================================
# 快速逻辑测试
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("评估与可视化模块 - 快速逻辑测试")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # ---- 构造合成数据 ----
    N = 200
    L, C = 30, 10
    # 原始未来收益：正态分布，std=0.02（贴近日线对数收益量级）
    forward_returns = np.random.normal(0.0, 0.02, N)
    # mu 幅度：∈ [0.1, 1.0]（避开 0，确保高于 conf_threshold）
    mu_mag = np.random.uniform(0.1, 1.0, N)
    # 故意让 mu 符号与 forward_returns 符号约 58% 一致（构造有信号的合成数据）
    correct_mask = np.random.rand(N) < 0.58
    fr_sign = np.sign(forward_returns)
    fr_sign = np.where(fr_sign == 0, 1.0, fr_sign)  # 边界保护
    mu_sign = np.where(correct_mask, fr_sign, -fr_sign)
    base_preds = mu_mag * mu_sign  # mu ∈ [-1, 1]
    # 合成 sigma：∈ [0, 0.3]（参考用，实际 sigma 由 Mock 模型 MC 采样产生）
    synthetic_sigma = np.random.uniform(0.0, 0.3, N)

    # ---- 合成 test_loader（单 batch，保证 Mock 模型能对齐全长预测）----
    X_syn = torch.randn(N, L, C)
    test_ds = TensorDataset(X_syn, torch.zeros(N))
    test_loader = DataLoader(test_ds, batch_size=N, shuffle=False)

    # ---- cfg（与 config.yaml 的 bayes / strategy 一致）----
    cfg = {
        'bayes': {'mc_samples': 20},
        'strategy': {
            'conf_threshold': 0.05,
            'position_lambda': 2.0,
            'fee_rate': 0.0006,
            'slippage': 0.0005,
            'periods_per_year': 365,
        },
    }
    device = torch.device('cpu')
    # 用系统临时目录存放测试输出，避免污染项目目录
    import tempfile
    results_dir = os.path.join(tempfile.gettempdir(), 'gtm_eval_test')

    # ---- 用 Mock 模型跑完整评估 ----
    mock_model = _MockModel(base_preds, noise_std=0.02)
    result = run_evaluation(mock_model, test_loader, forward_returns,
                            cfg, device, results_dir)

    # ---- 验证 ----
    all_pass = True
    # metrics 含 5 个核心字段（acc/return/vol/sharpe/dd）+ 2 个分散性字段
    # (trade_active_segments / nonzero_position_pct)
    expected_fields = {'directional_accuracy', 'annual_return',
                       'annual_volatility', 'sharpe', 'max_drawdown',
                       'trade_active_segments', 'nonzero_position_pct'}

    # 1. evaluate 返回的 metrics 含全部 7 个字段
    metrics_fields_ok = set(result['metrics'].keys()) == expected_fields
    tag = 'PASS' if metrics_fields_ok else 'FAIL'
    print(f"[{tag}] metrics 含 {len(expected_fields)} 个字段: {sorted(result['metrics'].keys())}")
    if not metrics_fields_ok:
        all_pass = False

    # 2. 3 张图文件生成（存在且非空）
    for fname in ['equity_curve.png', 'drawdown.png', 'pred_with_uncertainty.png']:
        p = os.path.join(results_dir, fname)
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else 0
        ok = exists and size > 0
        tag = 'PASS' if ok else 'FAIL'
        print(f"[{tag}] {fname} 生成且非空 (size={size} bytes)")
        if not ok:
            all_pass = False

    # 3. metrics.json 生成且可被 json.load 读回
    json_path = os.path.join(results_dir, 'metrics.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        json_ok = set(loaded.keys()) == expected_fields
        tag = 'PASS' if json_ok else 'FAIL'
        print(f"[{tag}] metrics.json 可被 json.load 读回: {loaded}")
        if not json_ok:
            all_pass = False
    except Exception as e:
        print(f"[FAIL] metrics.json 读取异常: {e}")
        all_pass = False

    # 4. 方向正确率 > 0.5（因为有约 58% 信号一致性）
    da = result['metrics']['directional_accuracy']
    da_ok = da > 0.5
    tag = 'PASS' if da_ok else 'FAIL'
    print(f"[{tag}] 方向正确率 > 0.5: {da:.4f}")
    if not da_ok:
        all_pass = False

    # 5. 打印指标值
    print("\n指标汇总：")
    for k, v in result['metrics'].items():
        print(f"  {k}: {v:.6f}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print(f"全部测试结果: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 70)
