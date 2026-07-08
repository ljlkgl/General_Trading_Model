#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格搜索：参数扫描优化
================================

对 TSMixer 量化交易模型的关键超参数做笛卡尔积扫描，自动训练 + 评估每组组合，
汇总指标到 CSV，按指定指标（默认 Sharpe）降序排列，快速定位最优参数区域。

用法（从项目根目录运行）:
    # 跑逻辑测试（验证参数解析、笛卡尔积等辅助逻辑）
    python scripts/grid_search.py

    # 跑真实网格搜索（需编辑下方 GRID 定义扫描参数）
    python scripts/grid_search.py --base configs/config_btc.yaml
    python scripts/grid_search.py --base configs/config_btc.yaml --metric sharpe --top 10

工作流程：
  1. 读取基础 config（--base）
  2. 对 GRID 中每组参数组合：深拷贝 config → 覆盖参数 → 训练 → 评估 → 记录指标
  3. 汇总到 results/grid_search/grid_results.csv，按 --metric 降序
  4. 打印 Top N（--top）

注意：
  - 网格搜索计算量大，组合数 = 各参数候选数乘积。建议先用小范围探索。
  - 可在 GRID 中加入 'train.epochs' 减少训练轮数以加速扫描。
  - 每组结果目录为 results/grid_search/run_XXX/，含 metrics.json 与图表。
  - 默认不保存每组 model.pt（save_model=false），避免占空间。
"""
import argparse
import copy
import itertools
import os
import random
import sys

# 将项目根目录加入 sys.path，使 gtm 包可被导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# =============================================================================
# 参数网格定义（用户在此编辑要扫描的参数）
# =============================================================================
# 键为点号分隔的配置路径（如 'strategy.conf_threshold' → cfg['strategy']['conf_threshold']）
# 值为候选值列表。笛卡尔积生成所有组合。
#
# 示例：扫描置信度门限、杠杆、波动率目标
GRID = {
    'strategy.conf_threshold': [0.10, 0.12, 0.15],
    'strategy.max_leverage': [3.0, 5.0, 8.0],
    'strategy.vol_target': [0.5, 1.0, 1.5],
}

# 常用参数路径参考：
#   'strategy.conf_threshold'        置信度门限
#   'strategy.position_lambda'       不确定度反向缩放系数
#   'strategy.max_leverage'          最大杠杆
#   'strategy.vol_target'            目标年化波动率
#   'strategy.max_drawdown_threshold' 回撤阈值
#   'strategy.dd_scale'              回撤减仓强度
#   'train.epochs'                   训练轮数（减小可加速）
#   'train.lr'                       学习率
#   'loss.alpha'                     方向 CE 权重
#   'loss.mse_weight'                MSE 权重
#   'data.window_L'                  输入窗口长度
#   'data.horizon'                   预测视野


# =============================================================================
# 辅助函数
# =============================================================================
def load_config(path):
    """加载 YAML 配置文件"""
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def set_nested(cfg, path, value):
    """
    按点号分隔的路径设置嵌套字典值。

    例：set_nested(cfg, 'strategy.conf_threshold', 0.15)
        → cfg['strategy']['conf_threshold'] = 0.15
    """
    keys = path.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def expand_grid(grid):
    """
    笛卡尔积展开参数网格。

    参数:
      grid: dict，键为参数路径，值为候选值列表

    返回:
      list[dict]：每个 dict 为一组参数组合（路径 → 值）
    """
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


# =============================================================================
# 单组参数运行
# =============================================================================
def run_single(base_cfg, param_overrides, run_id, device, grid_results_dir):
    """
    运行单组参数：深拷贝 config → 覆盖参数 → 训练 → 评估 → 返回指标 + 参数。

    参数:
      base_cfg: 基础配置字典
      param_overrides: dict，参数路径 → 值
      run_id: 组合序号（用于结果目录命名）
      device: torch.device
      grid_results_dir: 网格搜索根结果目录

    返回:
      dict：参数 + 指标 + run_id + best_val_ic
    """
    cfg = copy.deepcopy(base_cfg)
    for path, val in param_overrides.items():
        set_nested(cfg, path, val)

    # 每组结果独立目录
    run_dir = os.path.join(grid_results_dir, f'run_{run_id:03d}')
    cfg['output']['results_dir'] = run_dir
    cfg['output']['save_model'] = False  # 网格搜索不保存每组模型，省空间

    # 数据管道
    from gtm import data_loader, train, evaluate
    pipe = data_loader.build_pipeline(cfg)
    dataloaders = pipe['dataloaders']
    split = pipe['split']
    C, L = pipe['C'], pipe['L']
    raw_forward_returns = pipe.get('raw_forward_returns',
                                   split.get('raw_forward_returns'))
    test_forward_returns = raw_forward_returns['test']
    test_loader = dataloaders['test']

    # 训练
    model = train.build_model(cfg, num_features=C, seq_len=L).to(device)
    train_result = train.train_model(model, dataloaders, cfg, device,
                                     save_path=None)

    # 评估
    eval_result = evaluate.run_evaluation(
        model, test_loader, test_forward_returns, cfg, device, run_dir
    )
    metrics = eval_result['metrics']

    # 合并参数与指标
    row = {**param_overrides, **metrics,
           'run_id': run_id, 'best_val_ic': train_result['best_val_ic']}
    return row


# =============================================================================
# 完整网格搜索
# =============================================================================
def run_grid(base_config_path, grid, sort_metric='sharpe', top=10):
    """
    运行完整网格搜索，输出汇总 CSV。

    参数:
      base_config_path: 基础配置文件路径
      grid: 参数网格 dict
      sort_metric: 排序指标（降序），如 'sharpe' / 'directional_accuracy'
      top: 打印 Top N

    返回:
      pandas.DataFrame：按 sort_metric 降序的结果表
    """
    import pandas as pd
    import torch

    base_cfg = load_config(base_config_path)
    seed = base_cfg.get('train', {}).get('seed', 42)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 设备: {device}, 随机种子: {seed}")

    combos = expand_grid(grid)
    total = len(combos)
    print(f"[网格搜索] 基础配置: {base_config_path}")
    print(f"[网格搜索] 共 {total} 组参数组合")
    print(f"[网格搜索] 排序指标: {sort_metric}")
    print(f"[网格搜索] 扫描参数: {list(grid.keys())}")

    grid_results_dir = os.path.join('results', 'grid_search')
    os.makedirs(grid_results_dir, exist_ok=True)

    rows = []
    for i, combo in enumerate(combos):
        print(f"\n--- Run {i + 1}/{total}: {combo} ---")
        row = run_single(base_cfg, combo, i, device, grid_results_dir)
        rows.append(row)
        print(f"  → sharpe={row['sharpe']:.4f}, "
              f"acc={row['directional_accuracy']:.4f}, "
              f"mdd={row['max_drawdown']:.4f}, "
              f"年化={row['annual_return']:.4f}")

    # 汇总
    df = pd.DataFrame(rows)
    df = df.sort_values(sort_metric, ascending=False).reset_index(drop=True)

    csv_path = os.path.join(grid_results_dir, 'grid_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print(f"网格搜索完成，结果已保存到: {csv_path}")
    print(f"按 {sort_metric} 降序 Top {min(top, len(df))}:")
    print("=" * 70)
    # 打印关键列
    show_cols = ['run_id'] + list(grid.keys()) + [sort_metric, 'directional_accuracy',
                                                   'max_drawdown', 'annual_return']
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(top).to_string(index=False))
    return df


# =============================================================================
# 逻辑测试
# =============================================================================
def run_logic_tests():
    """验证参数解析、笛卡尔积、deepcopy 隔离等辅助逻辑（不跑真实训练）"""
    print("=" * 70)
    print("网格搜索 - 快速逻辑测试")
    print("=" * 70)

    all_pass = True

    # 测试 1：set_nested 正确设置嵌套值
    print("\n[测试 1] set_nested 嵌套路径设置")
    cfg = {'strategy': {'conf_threshold': 0.1, 'max_leverage': 5.0},
           'train': {'lr': 0.001}}
    set_nested(cfg, 'strategy.conf_threshold', 0.15)
    set_nested(cfg, 'train.lr', 0.0005)
    ok1 = (cfg['strategy']['conf_threshold'] == 0.15
           and cfg['train']['lr'] == 0.0005
           and cfg['strategy']['max_leverage'] == 5.0)  # 未改的值保持
    print(f"  set_nested 结果: {cfg} -> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        all_pass = False

    # 测试 2：expand_grid 笛卡尔积
    print("\n[测试 2] expand_grid 笛卡尔积")
    grid = {'a': [1, 2], 'b': [10, 20, 30]}
    combos = expand_grid(grid)
    ok2 = len(combos) == 6  # 2 * 3
    expected_first = {'a': 1, 'b': 10}
    expected_last = {'a': 2, 'b': 30}
    ok2 = ok2 and combos[0] == expected_first and combos[-1] == expected_last
    print(f"  组合数: {len(combos)} (期望 6) -> {'PASS' if ok2 else 'FAIL'}")
    print(f"  首组: {combos[0]}, 末组: {combos[-1]}")
    if not ok2:
        all_pass = False

    # 测试 3：deepcopy 隔离（修改副本不污染原配置）
    print("\n[测试 3] deepcopy 隔离")
    base = {'strategy': {'conf': 0.1}, 'train': {'lr': 0.001}}
    cfg_copy = copy.deepcopy(base)
    set_nested(cfg_copy, 'strategy.conf', 0.99)
    set_nested(cfg_copy, 'train.lr', 0.0001)
    ok3 = (base['strategy']['conf'] == 0.1
           and base['train']['lr'] == 0.001)
    print(f"  原 cfg 未被污染: conf={base['strategy']['conf']}, "
          f"lr={base['train']['lr']} -> {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        all_pass = False

    # 测试 4：默认 GRID 非空且可展开
    print("\n[测试 4] 默认 GRID 非空且可展开")
    combos = expand_grid(GRID)
    ok4 = len(GRID) > 0 and len(combos) > 0
    print(f"  GRID 参数数: {len(GRID)}, 组合数: {len(combos)} "
          f"-> {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        all_pass = False

    # 测试 5：多层嵌套路径设置
    print("\n[测试 5] 多层嵌套路径设置")
    cfg5 = {'a': {'b': {'c': 1}}}
    set_nested(cfg5, 'a.b.c', 999)
    ok5 = cfg5['a']['b']['c'] == 999
    print(f"  a.b.c = {cfg5['a']['b']['c']} (期望 999) -> "
          f"{'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        all_pass = False

    print("\n" + "=" * 70)
    print(f"全部测试结果: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 70)
    return all_pass


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSMixer 网格搜索")
    parser.add_argument(
        "--base", type=str, default=None,
        help="基础配置文件路径（提供则跑真实网格搜索，否则跑逻辑测试）"
    )
    parser.add_argument(
        "--metric", type=str, default="sharpe",
        help="排序指标（降序），默认 sharpe"
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="打印 Top N 结果，默认 10"
    )
    args = parser.parse_args()

    if args.base is None:
        # 无 --base：跑逻辑测试
        run_logic_tests()
    else:
        # 有 --base：跑真实网格搜索
        run_grid(args.base, GRID, sort_metric=args.metric, top=args.top)
