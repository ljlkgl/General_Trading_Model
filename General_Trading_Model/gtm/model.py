"""
TSMixer 贝叶斯量化交易模型
================================

基于全 MLP 的 TSMixer 架构（Time-Mixing + Feature-Mixing），
用于预测资产多空强度（连续值 y ∈ [-1, 1]）。

包含：
  - RevIN：可逆实例归一化（应对非平稳时序的分布漂移）
  - TSMixerBlock：交替的 Time-MLP 与 Feat-MLP 块（残差 + LayerNorm + Dropout）
  - TSMixer：主模型，输出标量信号
  - mc_dropout_predict：MC Dropout 推理，输出预测均值与不确定度

本文件自包含，仅依赖 torch，可独立运行 `python model.py` 进行逻辑测试。
"""

import torch
import torch.nn as nn


# =============================================================================
# RevIN：可逆实例归一化
# =============================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization（可逆实例归一化）

    对每个样本、每个特征，在时间维度 L 上独立计算 mean/std 并归一化；
    反归一化时使用 forward 阶段保存的统计量，保证可逆性。
    affine=True 时引入可学习的缩放 gamma 与平移 beta。
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # 可学习仿射参数（按特征维度 C 共享）
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

        # 保存 forward 时的统计量，供 reverse 使用
        self.mean = None
        self.std = None

    def forward(self, x):
        """
        输入: x ∈ (B, L, C)
        输出: 归一化后的 (B, L, C)

        对每个样本 b、每个特征 c，在时间维 L 上计算 mean/std。
        """
        # 在时间维 L（dim=1）上计算均值与方差，保持形状便于广播
        # mean/std: (B, 1, C)
        self.mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        self.std = torch.sqrt(var + self.eps)

        # 实例归一化
        x_norm = (x - self.mean) / self.std

        # 仿射变换
        if self.affine:
            # gamma/beta: (C,) -> 广播到 (B, L, C)
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def reverse(self, x):
        """
        反归一化（演示 RevIN 的可逆性）。
        输入: x ∈ (B, L, C)（在归一化空间下的张量）
        输出: 复原到原始空间的 (B, L, C)

        使用 forward 时保存的 mean/std。
        注意：若 affine=True，需先撤销 gamma/beta 再反归一化。
        """
        assert self.mean is not None and self.std is not None, \
            "reverse 前必须先调用 forward 以保存统计量"

        if self.affine:
            # 撤销仿射变换（假设 gamma 不为 0）
            x = (x - self.beta) / (self.gamma + self.eps)

        # 反归一化
        x_rev = x * self.std + self.mean
        return x_rev


# =============================================================================
# Time-MLP：时间混合
# =============================================================================
class TimeMLP(nn.Module):
    """
    时间混合 MLP：对每个特征在时间维度 L 上做全连接变换。
    结构：LayerNorm(C) → Linear(L→L) → GELU → Dropout，带残差连接。
    """

    def __init__(self, seq_len, dropout):
        super().__init__()
        # 对特征维 C 做归一化
        self.norm = nn.LayerNorm(seq_len)
        self.fc = nn.Linear(seq_len, seq_len)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x ∈ (B, L, C)
        输出: (B, L, C)
        """
        residual = x
        # 转置为 (B, C, L)，使时间维 L 成为最后一维
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.norm(x)       # LayerNorm 作用在 L 维（normalized_shape=seq_len）
        x = self.fc(x)         # Linear(L → L)
        x = self.act(x)
        x = self.drop(x)
        x = x.transpose(1, 2)  # 转回 (B, L, C)
        return x + residual


# =============================================================================
# Feat-MLP：特征混合
# =============================================================================
class FeatMLP(nn.Module):
    """
    特征混合 MLP：对每个时间步在特征维度 C 上做两层全连接。
    结构：LayerNorm(C) → Linear(C→hidden) → GELU → Dropout → Linear(hidden→C) → Dropout，带残差连接。
    """

    def __init__(self, num_features, hidden_mult=2, dropout=0.2):
        super().__init__()
        hidden = num_features * hidden_mult
        self.norm = nn.LayerNorm(num_features)
        self.fc1 = nn.Linear(num_features, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x ∈ (B, L, C)
        输出: (B, L, C)
        """
        residual = x
        x = self.norm(x)         # LayerNorm 作用在最后一维 C
        x = self.fc1(x)          # (B, L, hidden)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)          # (B, L, C)
        x = self.drop2(x)
        return x + residual


# =============================================================================
# TSMixer Block：一个 Time-MLP + 一个 Feat-MLP
# =============================================================================
class TSMixerBlock(nn.Module):
    """
    TSMixer 块：先时间混合，再特征混合。
    """

    def __init__(self, seq_len, num_features, hidden_mult, dropout):
        super().__init__()
        self.timemlp = TimeMLP(seq_len, dropout)
        self.featmlp = FeatMLP(num_features, hidden_mult, dropout)

    def forward(self, x):
        x = self.timemlp(x)
        x = self.featmlp(x)
        return x


# =============================================================================
# 注意力 Pooling：替代 mean pooling
# =============================================================================
class AttentionPooling(nn.Module):
    """
    注意力 pooling：用一个可学习的打分函数对时间维各时间步打分，
    softmax 得到权重后加权求和，替代 mean pooling。

    相比 mean pooling（对所有时间步等权平均），注意力 pooling 能让
    模型自适应地聚焦于近期或信息量更大的时间步，避免丢失时序细节。
    """

    def __init__(self, dim):
        super().__init__()
        # 打分函数：Linear(C → 1)，对每个时间步输出一个标量分数
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        """
        输入: x ∈ (B, L, C)
        输出: (B, C)
        """
        # 对每个时间步打分并去掉最后一维
        scores = self.score(x).squeeze(-1)            # (B, L)
        # 在时间维 L 上做 softmax 得到归一化权重
        weights = torch.softmax(scores, dim=1)        # (B, L)
        # 加权求和：广播 weights 到 (B, L, C) 后按时间维求和
        out = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, C)
        return out


# =============================================================================
# TSMixer 主模型
# =============================================================================
class TSMixer(nn.Module):
    """
    TSMixer 主模型：输入 (B, L, C)，输出纯线性标量信号 y ∈ (B,)。

    结构：
      1. 可选 RevIN 输入归一化
      2. K 个 TSMixerBlock
      3. 注意力 pooling（替代 mean pooling）→ (B, C)
      4. Linear(C → 1) → (B,)（纯线性输出，无 tanh 压缩）

    说明：移除 tanh 压缩层，使模型可以自由表达信号强度，不被限制在 [-1,1]；
         原先的 mean pooling 会等权平均所有时间步、丢失近期时序信息，
         替换为注意力 pooling 后可让模型聚焦于重要时间步。
    """

    def __init__(self, seq_len, num_features, num_blocks=4, dropout=0.2,
                 feat_hidden_mult=2, use_revin=True, revin_affine=True):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.use_revin = use_revin

        # RevIN 输入归一化
        if self.use_revin:
            self.revin = RevIN(num_features, affine=revin_affine)

        # K 个 TSMixer 块
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, num_features, feat_hidden_mult, dropout)
            for _ in range(num_blocks)
        ])

        # 注意力 pooling（替代 mean pooling）
        self.pool = AttentionPooling(num_features)

        # 输出层：纯线性 Linear(C → 1)，无 tanh 压缩
        self.head = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        输入: x ∈ (B, L, C)
        输出: y ∈ (B,)，纯线性标量信号（不再被 tanh 限制到 [-1,1]）
        """
        # 1. RevIN 输入归一化
        if self.use_revin:
            x = self.revin(x)

        # 2. 过 K 个 TSMixer 块
        for block in self.blocks:
            x = block(x)

        # 3. 注意力 pooling → (B, C)
        x = self.pool(x)  # (B, C)

        # 4. Linear(C → 1) → (B,)，纯线性输出
        y = self.head(x).squeeze(-1)  # (B,)
        return y


# =============================================================================
# 辅助函数：开启 Dropout（用于 MC Dropout 推理）
# =============================================================================
def enable_dropout(model):
    """
    将模型中所有 nn.Dropout 层设置为 train 模式（即启用 Dropout），
    其余层保持 eval 模式。这是 MC Dropout 推理的关键。
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# =============================================================================
# MC Dropout 推理
# =============================================================================
def mc_dropout_predict(model, x, T=20):
    """
    MC Dropout 推理：对同一输入 x 做 T 次随机前向，返回均值与标准差。

    参数:
      model: TSMixer 模型
      x: 输入张量 (B, L, C)
      T: 采样次数

    返回:
      mu:    (B,) 预测均值
      sigma: (B,) 预测标准差（不确定度），sigma ≥ 0
    """
    # 先整体 eval（关闭 BatchNorm 等的更新，并关闭 Dropout）
    model.eval()
    # 再单独启用 Dropout
    enable_dropout(model)

    # T 次随机前向
    preds = []
    with torch.no_grad():
        for _ in range(T):
            y = model(x)        # (B,)
            preds.append(y)
    # 堆叠为 (T, B)
    preds = torch.stack(preds, dim=0)

    # 均值与标准差
    mu = preds.mean(dim=0)              # (B,)
    sigma = preds.std(dim=0)            # (B,)
    # 保证 sigma ≥ 0（数值安全）
    sigma = torch.clamp(sigma, min=0.0)
    return mu, sigma


# =============================================================================
# 快速逻辑测试
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 70)
    print("TSMixer 贝叶斯量化交易模型 - 快速逻辑测试")
    print("=" * 70)

    # 测试参数
    B, L, C = 4, 30, 10
    x = torch.randn(B, L, C)

    all_pass = True

    # -------------------------------------------------------------------------
    # 测试 1：TSMixer 输出形状与值域
    # -------------------------------------------------------------------------
    print("\n[测试 1] TSMixer 前向输出形状与值域")
    model = TSMixer(
        seq_len=L, num_features=C, num_blocks=4,
        dropout=0.2, feat_hidden_mult=2,
        use_revin=True, revin_affine=True
    )
    model.eval()
    with torch.no_grad():
        y = model(x)
    shape_ok = (y.shape == (B,))
    # 移除 tanh 后输出为纯线性标量，不再限制在 [-1,1]，只检查为有限实数
    finite_ok = bool(torch.isfinite(y).all())
    print(f"  输出 shape: {tuple(y.shape)}，期望 ({B},)  -> {'PASS' if shape_ok else 'FAIL'}")
    print(f"  输出为有限实数: min={y.min().item():.4f}, max={y.max().item():.4f} -> {'PASS' if finite_ok else 'FAIL'}")
    if not (shape_ok and finite_ok):
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 2：RevIN 可逆性（forward 后 reverse 能恢复原值）
    # -------------------------------------------------------------------------
    print("\n[测试 2] RevIN 可逆性")
    revin = RevIN(num_features=C, affine=True)
    x_norm = revin(x)
    x_rev = revin.reverse(x_norm)
    err = (x_rev - x).abs().max().item()
    revin_ok = err < 1e-4
    print(f"  原始 x shape: {tuple(x.shape)}")
    print(f"  归一化后 mean≈0: {x_norm.mean().item():.4e}, std≈1: {x_norm.std().item():.4e}")
    print(f"  reverse 最大绝对误差: {err:.4e} (< 1e-4) -> {'PASS' if revin_ok else 'FAIL'}")
    if not revin_ok:
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 3：MC Dropout 推理输出形状与不确定性
    # -------------------------------------------------------------------------
    print("\n[测试 3] MC Dropout 推理")
    T = 20
    mu, sigma = mc_dropout_predict(model, x, T=T)
    mu_shape_ok = (mu.shape == (B,))
    sigma_shape_ok = (sigma.shape == (B,))
    sigma_nonneg = bool((sigma >= 0).all())
    # T>1 时至少有一个 sigma > 0（Dropout 引入随机性）
    sigma_has_pos = bool((sigma > 0).any().item())
    print(f"  mu shape: {tuple(mu.shape)}, 期望 ({B},) -> {'PASS' if mu_shape_ok else 'FAIL'}")
    print(f"  sigma shape: {tuple(sigma.shape)}, 期望 ({B},) -> {'PASS' if sigma_shape_ok else 'FAIL'}")
    print(f"  sigma 全部 ≥ 0: {bool(sigma_nonneg)} -> {'PASS' if sigma_nonneg else 'FAIL'}")
    print(f"  sigma 至少有一个 > 0 (T={T}): min={sigma.min().item():.4e}, max={sigma.max().item():.4e} -> {'PASS' if sigma_has_pos else 'FAIL'}")
    print(f"  mu 值域 ∈ [-1,1]: min={mu.min().item():.4f}, max={mu.max().item():.4f}")
    if not (mu_shape_ok and sigma_shape_ok and sigma_nonneg and sigma_has_pos):
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 4：关闭 use_revin 时模型仍能正常前向
    # -------------------------------------------------------------------------
    print("\n[测试 4] 关闭 RevIN 时模型前向")
    model_no_revin = TSMixer(
        seq_len=L, num_features=C, num_blocks=4,
        dropout=0.2, use_revin=False
    )
    model_no_revin.eval()
    with torch.no_grad():
        y2 = model_no_revin(x)
    no_revin_ok = (y2.shape == (B,)) and bool(torch.isfinite(y2).all())
    print(f"  输出 shape: {tuple(y2.shape)}, 输出为有限实数: {bool(no_revin_ok)} -> {'PASS' if no_revin_ok else 'FAIL'}")
    if not no_revin_ok:
        all_pass = False

    # -------------------------------------------------------------------------
    # 汇总
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"全部测试结果: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 70)
