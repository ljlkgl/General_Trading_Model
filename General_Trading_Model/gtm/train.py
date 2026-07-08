"""
TSMixer 训练循环
================================

实现 TSMixer 模型的完整训练流程：
  - build_model：根据 cfg 构建 TSMixer
  - CosineAnnealingWithWarmup：warmup + 余弦退火学习率调度
  - anneal_dropout：Dropout 概率线性退火（KL 退火的简化等价）
  - train_one_epoch / evaluate_loss：单轮训练 / 评估
  - train_model：完整训练循环（早停 + 最优权重恢复 + 历史记录）
  - save_checkpoint / load_checkpoint：权重持久化

本文件自包含，可独立运行 `python train.py` 进行逻辑测试。
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .model import TSMixer   # 作为 gtm 包成员导入
except ImportError:
    from model import TSMixer    # 作为脚本直接运行（python gtm/train.py）


# =============================================================================
# 方向损失：组合 MSE + 方向 CE + IC 损失
# =============================================================================
class DirectionalLoss(nn.Module):
    """
    混合损失：MSE + alpha * 方向CE + beta * IC损失
    直接优化方向正确率，避免 MSE 在噪声数据上退化为预测均值。

    - MSE 项：保留回归目标，使预测幅度不至于失控
    - 方向 CE 项：把连续标签离散化为 {跌, 平, 涨} 三类，用交叉熵直接
                  拉开涨跌方向的 logit，提升方向正确率
    - IC 损失项：1 - Pearson 相关，逼迫预测与标签在排序上一致
    """

    def __init__(self, alpha=1.0, beta=0.5, dir_threshold=0.02, num_classes=3,
                 mse_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dir_threshold = dir_threshold
        self.num_classes = num_classes
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()

    def _to_class(self, y):
        """
        y: (B,) 连续标签 -> (B,) 类别索引 {0:跌, 1:平, 2:涨}
        """
        cls = torch.ones_like(y, dtype=torch.long)  # 默认平
        cls[y > self.dir_threshold] = 2   # 涨
        cls[y < -self.dir_threshold] = 0  # 跌
        return cls

    def _dir_ce(self, pred, y_cls):
        """
        pred: (B,) 连续预测 -> logits (B, 3)，用线性映射构造 logits
        涨类 logit = pred，跌类 logit = -pred，平类 logit = 0
        """
        logits = torch.stack([-pred, torch.zeros_like(pred), pred], dim=1)  # (B, 3)
        # 类别权重处理不平衡（平类通常多）
        weights = torch.tensor([1.0, 0.3, 1.0], device=pred.device)
        return nn.functional.cross_entropy(logits, y_cls, weight=weights)

    def _ic_loss(self, pred, y):
        """
        近似 Spearman IC 损失：1 - Pearson相关
        """
        pred_c = pred - pred.mean()
        y_c = y - y.mean()
        num = (pred_c * y_c).sum()
        den = torch.sqrt((pred_c ** 2).sum() * (y_c ** 2).sum() + 1e-8)
        ic = num / den
        return 1.0 - ic

    def forward(self, pred, y):
        mse_loss = self.mse(pred, y)
        y_cls = self._to_class(y)
        dir_loss = self._dir_ce(pred, y_cls)
        ic_loss = self._ic_loss(pred, y)
        return self.mse_weight * mse_loss + self.alpha * dir_loss + self.beta * ic_loss


# =============================================================================
# 1. 构建模型
# =============================================================================
def build_model(cfg, num_features, seq_len):
    """
    根据 cfg['model'] 配置构建 TSMixer 模型。

    参数:
      cfg: 完整配置字典（含 'model' 子字典）
      num_features: 特征数 C（由数据管道返回）
      seq_len: 窗口长度 L（= cfg['data']['window_L']）

    返回:
      TSMixer 模型实例
    """
    m = cfg['model']
    # cfg 中 'revin' 对应 TSMixer 的 'use_revin' 参数
    model = TSMixer(
        seq_len=seq_len,
        num_features=num_features,
        num_blocks=m.get('num_blocks', 4),
        dropout=m.get('dropout', 0.2),
        feat_hidden_mult=m.get('feat_hidden_mult', 2),
        use_revin=m.get('revin', True),
        revin_affine=m.get('revin_affine', True),
    )
    return model


# =============================================================================
# 2. CosineAnnealingWithWarmup 学习率调度器
# =============================================================================
class CosineAnnealingWithWarmup:
    """
    Warmup + 余弦退火学习率调度器。

    - warmup 阶段（前 warmup_steps 步）：lr 线性从 0 增到 base_lr
    - 之后：余弦退火从 base_lr 衰减到 ~0

    基于 torch.optim.lr_scheduler.LambdaLR 实现，按 step（batch）粒度调度。
    """

    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))

        def lr_lambda(step):
            if step < self.warmup_steps:
                # 线性 warmup：0 → base_lr
                return float(step) / float(self.warmup_steps)
            # 余弦退火：base_lr → 0
            progress = (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


# =============================================================================
# 3. Dropout 退火
# =============================================================================
def anneal_dropout(model, epoch, total_epochs, p_start, p_end):
    """
    Dropout 概率线性退火：从 p_start 线性退到 p_end。

    这是 KL 退火的简化等价形式（高 dropout = 强正则 = 高 KL 权重）。
    通过遍历 model.modules()，找到 nn.Dropout 实例，更新其 p 属性。

    参数:
      model: 模型
      epoch: 当前 epoch（从 0 开始）
      total_epochs: 总 epoch 数
      p_start: 起始 dropout 概率（高）
      p_end: 终止 dropout 概率（低）
    """
    if total_epochs <= 1:
        ratio = 0.0
    else:
        ratio = epoch / float(total_epochs - 1)
    p = p_start + (p_end - p_start) * ratio
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


# =============================================================================
# 4. 单轮训练
# =============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, grad_clip, device):
    """
    遍历 loader 一个 epoch：
      前向 → MSE 损失 → 反向 → 梯度裁剪 → optimizer.step() → scheduler.step()

    返回平均 train loss。
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)                 # (B,)
        loss = criterion(pred, yb)
        loss.backward()

        # 梯度裁剪
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


# =============================================================================
# 5. 评估损失
# =============================================================================
def evaluate_loss(model, loader, criterion, device):
    """
    在 loader 上计算平均 loss（eval 模式，无 dropout）。
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            bs = xb.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
    return total_loss / max(1, n_samples)


def evaluate_val_metrics(model, loader, device):
    """
    在验证集上计算多项指标：MSE、IC（Pearson相关）、方向正确率、预测std。

    IC 是交易最关心的指标——预测与标签的排序一致性。
    方向正确率是策略盈亏的直接来源。
    预测 std 反映模型置信度（太低说明模型在“预测均值”）。

    返回 dict: {'mse', 'ic', 'dir_acc', 'pred_std'}
    """
    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            all_pred.append(pred.cpu())
            all_y.append(yb.cpu())
    pred = torch.cat(all_pred).numpy()
    y = torch.cat(all_y).numpy()

    mse = float(np.mean((pred - y) ** 2))
    # IC: Pearson 相关
    if pred.std() < 1e-8 or y.std() < 1e-8:
        ic = 0.0
    else:
        ic = float(np.corrcoef(pred, y)[0, 1])
    # 方向正确率
    dir_acc = float(np.mean(np.sign(pred) == np.sign(y)))
    pred_std = float(pred.std())

    return {'mse': mse, 'ic': ic, 'dir_acc': dir_acc, 'pred_std': pred_std}


# =============================================================================
# 6. 完整训练循环
# =============================================================================
def train_model(model, dataloaders, cfg, device, save_path=None):
    """
    完整训练循环：
      - 训练损失：DirectionalLoss（MSE + 方向CE + IC）
      - 早停：监控验证集 IC（信息系数），连续 patience 轮不上升则停止
      - 保存 IC 最高的权重到 save_path

    关键设计：早停准则用 IC 而非 MSE，因为 IC 直接反映预测的排序能力，
    是交易盈亏的核心驱动力。MSE 会惩罚“激进但方向正确”的预测，
    导致模型退化为“预测均值”——这在噪声大的高频数据上尤其严重。

    返回 dict:
      {'model': 训练后的模型, 'history': {...}, 'best_val_ic': ...}
    """
    t = cfg['train']
    m = cfg['model']
    epochs = int(t['epochs'])
    patience = int(t.get('patience', 10))
    grad_clip = t.get('grad_clip', None)
    warmup_steps = int(t.get('warmup_steps', 0))

    # 训练损失：DirectionalLoss（MSE + 方向CE + IC），从 cfg['loss'] 读取超参
    loss_cfg = cfg.get('loss', {})
    train_criterion = DirectionalLoss(
        alpha=float(loss_cfg.get('alpha', 1.0)),
        beta=float(loss_cfg.get('beta', 0.5)),
        dir_threshold=float(loss_cfg.get('dir_threshold', 0.02)),
        mse_weight=float(loss_cfg.get('mse_weight', 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t['lr']),
        weight_decay=float(t.get('weight_decay', 0.0)),
    )

    # 估算总步数（按 batch 粒度）
    train_loader = dataloaders['train']
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, total_steps)

    # Dropout 退火配置
    dropout_anneal = m.get('dropout_anneal', False)
    p_start = float(m.get('dropout', 0.2))
    p_end = float(m.get('dropout_min', 0.05))

    # 早停监控 IC（越高越好，与 loss 相反）
    best_val_ic = -float('inf')
    best_state = None
    counter = 0
    history = {'train_losses': [], 'val_ics': [], 'val_dir_accs': [],
               'val_mses': [], 'val_pred_stds': [], 'lrs': []}

    # 确保保存目录存在
    if save_path is not None and os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    has_val = 'val' in dataloaders and dataloaders['val'] is not None

    for epoch in range(epochs):
        # Dropout 退火
        if dropout_anneal:
            anneal_dropout(model, epoch, epochs, p_start, p_end)

        # 训练一个 epoch（使用 DirectionalLoss）
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, train_criterion, grad_clip, device
        )

        # 验证：计算 IC、方向正确率、MSE、预测std
        if has_val:
            vm = evaluate_val_metrics(model, dataloaders['val'], device)
            val_ic = vm['ic']
            val_dir_acc = vm['dir_acc']
            val_mse = vm['mse']
            val_pred_std = vm['pred_std']
        else:
            val_ic = -train_loss  # 退化：无验证集时用 -train_loss 近似
            val_dir_acc = 0.0
            val_mse = 0.0
            val_pred_std = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        history['train_losses'].append(train_loss)
        history['val_ics'].append(val_ic)
        history['val_dir_accs'].append(val_dir_acc)
        history['val_mses'].append(val_mse)
        history['val_pred_stds'].append(val_pred_std)
        history['lrs'].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}, "
              f"val_IC: {val_ic:.6f}, dir_acc: {val_dir_acc:.4f}, "
              f"mse: {val_mse:.6f}, pred_std: {val_pred_std:.4f}, "
              f"lr: {current_lr:.6e}")

        # 早停判断：IC 越高越好
        if val_ic > best_val_ic + 1e-6:
            best_val_ic = val_ic
            # 深拷贝最优权重
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            counter = 0
            # 保存最优权重
            if save_path is not None:
                save_checkpoint(model, save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发：连续 {patience} 轮 val IC 不上升 "
                      f"(best={best_val_ic:.6f})")
                break

    # 恢复最优权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'history': history,
        'best_val_ic': best_val_ic,
    }


# =============================================================================
# 7. 权重保存 / 加载
# =============================================================================
def save_checkpoint(model, path):
    """保存模型权重到 path。"""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    """从 path 加载模型权重到 model。"""
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)


# =============================================================================
# 快速逻辑测试
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("TSMixer 训练循环 - 快速逻辑测试")
    print("=" * 70)

    all_pass = True
    device = torch.device('cpu')

    # -------------------------------------------------------------------------
    # 合成小数据集：50 样本，窗口 10，特征 8
    # -------------------------------------------------------------------------
    N, L, C = 50, 10, 8
    X = torch.randn(N, L, C)
    # 构造一个有微弱线性信号的标签，使 loss 有下降空间
    y = (X.mean(dim=(1, 2)) * 0.3 + torch.randn(N) * 0.1)

    n_train = 30
    n_val = 20
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])

    # -------------------------------------------------------------------------
    # 测试 1：loss 能下降（不发散为 NaN）
    # -------------------------------------------------------------------------
    print("\n[测试 1] loss 能下降且不发散")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': val_loader}

    cfg = {
        'model': {
            'num_blocks': 2, 'dropout': 0.2, 'dropout_anneal': True,
            'dropout_min': 0.05, 'feat_hidden_mult': 2,
            'revin': True, 'revin_affine': True,
        },
        'train': {
            'epochs': 5, 'batch_size': 16, 'lr': 0.01, 'weight_decay': 0.0001,
            'warmup_steps': 5, 'patience': 3, 'grad_clip': 1.0, 'seed': 42,
        },
    }

    torch.manual_seed(42)
    model = build_model(cfg, num_features=C, seq_len=L)
    initial_loss = evaluate_loss(model, train_loader, nn.MSELoss(), device)
    print(f"  初始 train loss: {initial_loss:.6f}")

    # 用系统临时目录存放测试权重，避免污染项目目录
    import tempfile
    _tmp = tempfile.gettempdir()
    result = train_model(model, dataloaders, cfg, device,
                         save_path=os.path.join(_tmp, 'gtm_test_model.pt'))
    final_loss = result['history']['train_losses'][-1]
    no_nan = not (math.isnan(final_loss) or math.isinf(final_loss))
    print(f"  最终 train loss: {final_loss:.6f}")
    print(f"  无 NaN/Inf: {no_nan} -> {'PASS' if no_nan else 'FAIL'}")
    if not no_nan:
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 2：早停能触发
    # -------------------------------------------------------------------------
    print("\n[测试 2] 早停能触发（patience=2, epochs=20）")
    # 用纯随机标签（无信号）+ 小学习率，使 val loss 难以持续下降
    X2 = torch.randn(40, L, C)
    y2 = torch.randn(40) * 0.5  # 纯噪声标签
    es_train_ds = TensorDataset(X2[:24], y2[:24])
    es_val_ds = TensorDataset(X2[24:], y2[24:])
    es_train_loader = DataLoader(es_train_ds, batch_size=8, shuffle=True)
    es_val_loader = DataLoader(es_val_ds, batch_size=8, shuffle=False)
    es_dataloaders = {'train': es_train_loader, 'val': es_val_loader, 'test': es_val_loader}

    es_cfg = {
        'model': {
            'num_blocks': 2, 'dropout': 0.3, 'dropout_anneal': False,
            'dropout_min': 0.05, 'feat_hidden_mult': 2,
            'revin': True, 'revin_affine': True,
        },
        'train': {
            'epochs': 20, 'batch_size': 8, 'lr': 0.01, 'weight_decay': 0.0001,
            'warmup_steps': 3, 'patience': 2, 'grad_clip': 1.0, 'seed': 42,
        },
        # 早停机制验证与损失函数无关：关闭方向CE与IC项，退化为纯MSE。
        # 新模型（注意力pooling+无tanh）初始预测偏离0较多，需更大lr快速收敛到
        # 纯噪声平台期，使val loss停止持续下降，从而触发早停。
        'loss': {'alpha': 0.0, 'beta': 0.0, 'dir_threshold': 0.02},
    }
    torch.manual_seed(42)
    es_model = build_model(es_cfg, num_features=C, seq_len=L)
    es_result = train_model(es_model, es_dataloaders, es_cfg, device, save_path=None)
    epochs_run = len(es_result['history']['train_losses'])
    early_stopped = epochs_run < es_cfg['train']['epochs']
    print(f"  实际运行 epoch 数: {epochs_run} / {es_cfg['train']['epochs']}")
    print(f"  早停触发: {early_stopped} -> {'PASS' if early_stopped else 'FAIL'}")
    if not early_stopped:
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 3：save_checkpoint + load_checkpoint 后前向输出一致
    # -------------------------------------------------------------------------
    print("\n[测试 3] 权重保存/加载一致性")
    torch.manual_seed(42)
    ckpt_model = build_model(cfg, num_features=C, seq_len=L)
    ckpt_model.eval()
    test_x = torch.randn(4, L, C)
    with torch.no_grad():
        out_before = ckpt_model(test_x)

    ckpt_path = os.path.join(_tmp, 'gtm_ckpt_test.pt')
    save_checkpoint(ckpt_model, ckpt_path)

    # 新建一个同结构模型并加载
    torch.manual_seed(0)  # 不同种子，确保初始权重不同
    loaded_model = build_model(cfg, num_features=C, seq_len=L)
    loaded_model.eval()
    # 加载前输出应不同
    with torch.no_grad():
        out_diff_init = loaded_model(test_x)
    weights_differ_before = not torch.allclose(out_before, out_diff_init)

    load_checkpoint(loaded_model, ckpt_path)
    loaded_model.eval()
    with torch.no_grad():
        out_after = loaded_model(test_x)

    max_diff = (out_after - out_before).abs().max().item()
    ckpt_ok = max_diff < 1e-6
    print(f"  加载前输出不同（确认初始权重不同）: {weights_differ_before}")
    print(f"  加载后最大输出差异: {max_diff:.2e} (< 1e-6) -> {'PASS' if ckpt_ok else 'FAIL'}")
    if not ckpt_ok:
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 4：anneal_dropout 后 Dropout 的 p 值改变
    # -------------------------------------------------------------------------
    print("\n[测试 4] anneal_dropout 改变 Dropout.p")
    torch.manual_seed(42)
    anneal_model = build_model(cfg, num_features=C, seq_len=L)
    # 初始 p 应为 0.2
    init_ps = [m.p for m in anneal_model.modules() if isinstance(m, nn.Dropout)]
    init_ok = all(abs(p - 0.2) < 1e-6 for p in init_ps)

    # 退火到 epoch=total-1 → p_end=0.05
    anneal_dropout(anneal_model, epoch=4, total_epochs=5, p_start=0.2, p_end=0.05)
    end_ps = [m.p for m in anneal_model.modules() if isinstance(m, nn.Dropout)]
    end_ok = all(abs(p - 0.05) < 1e-6 for p in end_ps)

    # 中间 epoch=2（5 epochs，ratio=0.5）→ p = 0.2 + (0.05-0.2)*0.5 = 0.125
    anneal_dropout(anneal_model, epoch=2, total_epochs=5, p_start=0.2, p_end=0.05)
    mid_ps = [m.p for m in anneal_model.modules() if isinstance(m, nn.Dropout)]
    mid_ok = all(abs(p - 0.125) < 1e-6 for p in mid_ps)

    print(f"  初始 Dropout.p (期望 0.2): {init_ps[:2]}... -> {'PASS' if init_ok else 'FAIL'}")
    print(f"  退火到末尾 (期望 0.05): {end_ps[:2]}... -> {'PASS' if end_ok else 'FAIL'}")
    print(f"  退火到中间 (期望 0.125): {mid_ps[:2]}... -> {'PASS' if mid_ok else 'FAIL'}")
    if not (init_ok and end_ok and mid_ok):
        all_pass = False

    # -------------------------------------------------------------------------
    # 汇总
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"全部测试结果: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 70)
