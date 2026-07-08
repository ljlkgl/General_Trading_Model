#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TSMixer 贝叶斯量化交易模型 - 主入口

基于 TSMixer 架构 + MC Dropout 实现不确定性估计，用于加密货币量化交易。

完整端到端流程：
  1. 加载 config（load_config）
  2. set_seed
  3. data_loader.build_pipeline → dataloaders / split / C / L / raw_forward_returns
  4. train.build_model → TSMixer
  5. train.train_model → 训练（早停 + 最优权重恢复）
  6. evaluate.run_evaluation → MC Dropout 预测 + 回测 + 出图 + metrics.json
  7. 打印最终指标，判断是否达标（对比 cfg['targets']）
  8. 保存 model.pt 到 results/

用法（从项目根目录运行）:
    python scripts/run.py --config configs/config_btc.yaml
    python scripts/run.py --help
"""
import argparse
import os
import random
import sys

# 将项目根目录加入 sys.path，使 gtm 包可被导入
# run.py 位于 scripts/，项目根目录为其父目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_config(path: str) -> dict:
    """加载 YAML 配置文件

    Args:
        path: 配置文件路径

    Returns:
        解析后的配置字典
    """
    import yaml  # 延迟导入，确保 --help 不依赖第三方包
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """设置全局随机种子，保证实验可复现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    # numpy（若已安装）
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    # torch（若已安装）
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main() -> None:
    """主流程：数据 -> 模型 -> 训练 -> 评估 -> 判断达标"""
    parser = argparse.ArgumentParser(
        description="TSMixer 贝叶斯量化交易模型主入口"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config_btc.yaml",
        help="配置文件路径（默认: configs/config_btc.yaml）"
    )
    args = parser.parse_args()

    # 1. 加载配置
    cfg = load_config(args.config)
    print(f"[INFO] 已加载配置文件: {args.config}")

    # 2. 设置随机种子
    seed = cfg.get("train", {}).get("seed", 42)
    set_seed(seed)
    print(f"[INFO] 已设置随机种子: {seed}")

    # 选择设备（cuda / cpu）
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 准备结果目录
    results_dir = cfg.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, "model.pt")
    print(f"[INFO] 结果目录: {results_dir}")

    # 3. 数据准备：build_pipeline
    print("[STEP 1/4] data_loader: 加载数据 + 特征工程 + 滑窗 + 划分")
    from gtm import data_loader
    pipe = data_loader.build_pipeline(cfg)
    dataloaders = pipe['dataloaders']
    split = pipe['split']
    C = pipe['C']
    L = pipe['L']
    # raw_forward_returns 既在 pipe 顶层，也在 split['raw_forward_returns']
    raw_forward_returns = pipe.get('raw_forward_returns', split.get('raw_forward_returns'))
    test_forward_returns = raw_forward_returns['test']
    test_loader = dataloaders['test']
    test_timestamps = split['timestamps']['test']
    print(f"  特征数 C={C}, 窗口 L={L}")
    print(f"  样本数: train={len(split['train']['X'])}, "
          f"val={len(split['val']['X'])}, test={len(split['test']['X'])}")
    print(f"  测试集原始未来收益长度: {len(test_forward_returns)}")

    # 4. 模型构建：TSMixer
    print("[STEP 2/4] model: 构建 TSMixer")
    from gtm import train
    model = train.build_model(cfg, num_features=C, seq_len=L)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params}")

    # 5. 训练
    print("[STEP 3/4] train: 训练 TSMixer")
    train_result = train.train_model(
        model, dataloaders, cfg, device, save_path=model_path
    )
    best_val_ic = train_result.get('best_val_ic', 0.0)
    n_epochs_run = len(train_result['history']['train_losses'])
    print(f"  训练完成: 实际 epochs={n_epochs_run}, best_val_ic={best_val_ic:.6f}")

    # 确保 model.pt 落盘（train_model 内部已 save_checkpoint，但显式再保存一次以防万一）
    if cfg.get('output', {}).get('save_model', True):
        train.save_checkpoint(model, model_path)
        print(f"  模型已保存到: {model_path}")

    # 6. 评估：MC Dropout 预测 + 回测 + 出图 + metrics.json
    print("[STEP 4/4] evaluate: MC 采样预测 + 回测 + 指标 + 出图")
    from gtm import evaluate
    eval_result = evaluate.run_evaluation(
        model, test_loader, test_forward_returns, cfg, device, results_dir
    )
    metrics = eval_result['metrics']

    # 7. 打印最终指标，判断是否达标
    print("\n" + "=" * 70)
    print("最终评估指标（测试集）")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    targets = cfg.get('targets', {})
    min_acc = targets.get('min_accuracy', 0.54)
    min_sharpe = targets.get('min_sharpe', 1.1)
    min_active_segs = targets.get('min_active_segments', 7)  # 10段中至少7段有交易
    acc_ok = metrics['directional_accuracy'] >= min_acc
    sharpe_ok = metrics['sharpe'] >= min_sharpe
    active_segs = metrics.get('trade_active_segments', 10)
    active_ok = active_segs >= min_active_segs

    print("-" * 70)
    print(f"目标: 方向正确率 ≥ {min_acc}, 夏普比率 ≥ {min_sharpe}, 交易分散段数 ≥ {min_active_segs}/10")
    print(f"方向正确率 {metrics['directional_accuracy']:.4f} "
          f"{'✓ 达标' if acc_ok else '✗ 未达标'} (阈值 {min_acc})")
    print(f"夏普比率   {metrics['sharpe']:.4f} "
          f"{'✓ 达标' if sharpe_ok else '✗ 未达标'} (阈值 {min_sharpe})")
    print(f"交易分散   {active_segs}/10 段有交易 "
          f"{'✓ 达标' if active_ok else '✗ 未达标'} (阈值 {min_active_segs})")

    # 验证回测含手续费与滑点（不得为 0）
    s = cfg['strategy']
    fee_ok = s.get('fee_rate', 0) > 0
    slip_ok = s.get('slippage', 0) > 0
    print(f"手续费率 fee_rate={s.get('fee_rate')} ({'✓ >0' if fee_ok else '✗ =0 作弊'})")
    print(f"滑点     slippage={s.get('slippage')} ({'✓ >0' if slip_ok else '✗ =0 作弊'})")

    all_ok = acc_ok and sharpe_ok and fee_ok and slip_ok and active_ok
    print("=" * 70)
    print(f"总体: {'全部达标 ✓' if all_ok else '未全部达标 ✗'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
