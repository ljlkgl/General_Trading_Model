"""
General Trading Model (GTM)
============================

基于 TSMixer + MC Dropout 的贝叶斯量化交易模型包。

子模块：
  - data_loader：数据管道（CSV → 特征 → 滑窗 → 划分）
  - model：TSMixer 模型（RevIN + Blocks + AttentionPooling）
  - train：训练循环（DirectionalLoss + IC 早停）
  - strategy：策略回测（信号 → 仓位 → 风控 → 指标）
  - evaluate：评估可视化（MC Dropout + 出图 + JSON）
"""

from . import data_loader, model, train, strategy, evaluate

__all__ = ['data_loader', 'model', 'train', 'strategy', 'evaluate']
