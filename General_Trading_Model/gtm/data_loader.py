"""
量化交易模型数据管道
================================

实现从原始 K 线 CSV 到 TSMixer 模型可用 DataLoader 的完整流程：
  load_csv → add_features → make_labels → make_windows → split_time → get_dataloaders

严格防止未来数据泄露：
  - make_labels 的归一化统计量只来自训练集（前 60%）
  - make_windows 的每个样本只用过去窗口（第 i 个样本输入为 i:i+L，标签为 i+L-1）
  - split_time 按时间顺序划分，不随机打乱

本文件自包含，可独立运行 `python data_loader.py` 进行逻辑测试。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# 1. 加载 CSV
# =============================================================================
def load_csv(path):
    """
    读取 CSV，把 datetime 列解析为 DatetimeIndex。

    - 按时间升序排序
    - 前向填充缺失值；若首行仍为 NaN 则向后填充
    - 返回 DataFrame（索引为 DatetimeIndex，列为原始 10 列）
    """
    df = pd.read_csv(path)
    # 解析 datetime 列并设为索引
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()  # 按时间升序

    # 前向填充；若首行仍为 NaN 则向后填充
    df = df.ffill().bfill()

    return df


# =============================================================================
# 2. 特征工程
# =============================================================================
def add_features(df, cfg):
    """
    在原始 10 列基础上新增连续数值衍生特征（不引入人工类别/阈值）。

    cfg: features 配置字典，含
      - log_return_windows: list[int]
      - momentum_windows: list[int]
      - volatility_windows: list[int]
      - ma_short: int
      - ma_long: int

    新增特征：
      - 对数收益率：log(close/close.shift(n)) for n in log_return_windows
      - 动量：(close - close.shift(n)) / close.shift(n) for n in momentum_windows
      - 波动率：1 期对数收益率的滚动 std（窗口在 volatility_windows）
      - MA 差值：(MA_short - MA_long) / close 归一化
      - taker 买卖差额与买卖压力比
      - RSI/100、ATR/close 归一化
      - 量价相关性：volume 滚动均值 / close 滚动均值（窗口 ma_long）
      - 价格相对 MA 的偏离度（多窗口）
      - 波动率比（短期/长期）
      - 成交量变化率

    生成后丢弃含 NaN 的前若干行（dropna）。
    """
    df = df.copy()
    close = df['close']

    # --- 对数收益率 ---
    for n in cfg['log_return_windows']:
        df[f'log_ret_{n}'] = np.log(close / close.shift(n))

    # --- 动量 ---
    for n in cfg['momentum_windows']:
        df[f'momentum_{n}'] = (close - close.shift(n)) / close.shift(n)

    # --- 波动率（1 期对数收益率的滚动 std）---
    log_ret_1 = np.log(close / close.shift(1))
    for w in cfg['volatility_windows']:
        df[f'vol_{w}'] = log_ret_1.rolling(w).std()

    # --- MA 差值（归一化）---
    ma_short = cfg['ma_short']
    ma_long = cfg['ma_long']
    df['ma_diff'] = (
        close.rolling(ma_short).mean() - close.rolling(ma_long).mean()
    ) / close

    # --- taker 买卖差额（归一化）---
    tb = df['taker_buy']
    ts = df['taker_sell']
    df['taker_diff'] = (tb - ts) / (tb + ts + 1.0)
    # 买卖压力比：taker_buy / (taker_buy + taker_sell)，范围 (0, 1)
    df['taker_buy_ratio'] = tb / (tb + ts + 1.0)

    # --- RSI / ATR 归一化 ---
    df['rsi_norm'] = df['RSI'] / 100.0
    df['atr_norm'] = df['ATR'] / close

    # --- 量价相关性（滚动均值之比，窗口 ma_long）---
    df['vol_price_ratio'] = (
        df['volume'].rolling(ma_long).mean() / close.rolling(ma_long).mean()
    )

    # --- 价格相对 MA 的偏离度（多窗口归一化）---
    for w in [5, 10, 20]:
        df[f'price_ma_dev_{w}'] = (close - close.rolling(w).mean()) / close.rolling(w).mean()

    # --- 波动率比（短期 5 / 长期 20）---
    vol_short = log_ret_1.rolling(5).std()
    vol_long = log_ret_1.rolling(20).std()
    df['vol_ratio'] = vol_short / (vol_long + 1e-8)

    # --- 成交量变化率（归一化）---
    df['vol_change'] = df['volume'].pct_change().clip(-1.0, 1.0)

    # 丢弃含 NaN 的前若干行
    df = df.dropna()

    return df


# =============================================================================
# 3. 生成标签
# =============================================================================
def make_labels(df, horizon, clip_quantile=0.05):
    """
    生成未来 horizon 期对数收益率标签，并归一化到 [-1, 1]。

    - 未来收益：np.log(df['close'].shift(-horizon) / df['close'])，最后 horizon 行为 NaN
    - 丢弃最后 horizon 行的 NaN
    - 按训练集（前 60%）分位数裁剪边界，全局裁剪
    - 减去训练集均值（中心化，避免模型学到系统性看多/看空偏差）
    - 用训练集 std（裁剪后）归一化，clip 到 [-1, 1]
    - 归一化统计量只来自训练集，避免数据泄露

    返回 labels Series（与 df 对齐，已丢弃最后 horizon 行）。
    归一化统计量通过 Series.attrs 返回（clip_lo, clip_hi, train_std, train_mean）。
    """
    # 未来 horizon 期对数收益率（shift(-horizon) 取未来值）
    future_log_ret = np.log(df['close'].shift(-horizon) / df['close'])

    # 丢弃最后 horizon 行的 NaN
    labels = future_log_ret.dropna()

    # 训练集 = 前 60%
    n = len(labels)
    n_train = int(n * 0.6)
    train_labels = labels.iloc[:n_train]

    # 训练集分位数裁剪边界（用于全局裁剪）
    lo = train_labels.quantile(clip_quantile)
    hi = train_labels.quantile(1.0 - clip_quantile)

    # 全局裁剪（用训练集边界裁剪所有标签）
    labels_clipped = labels.clip(lower=lo, upper=hi)

    # 训练集均值与 std（裁剪后），用于中心化与归一化
    train_mean = labels_clipped.iloc[:n_train].mean()
    train_std = labels_clipped.iloc[:n_train].std()
    # 防止 std 为 0 或 NaN
    if np.isnan(train_std) or train_std < 1e-8:
        train_std = 1.0
    if np.isnan(train_mean):
        train_mean = 0.0

    # 中心化（减训练集均值）+ 归一化到大致 [-1, 1] 范围
    labels_norm = (labels_clipped - train_mean) / train_std

    # 硬裁剪到 [-1, 1]
    labels_norm = labels_norm.clip(-1.0, 1.0)

    # 附带归一化统计量（供后续反归一化或解释使用）
    labels_norm.attrs['clip_lo'] = float(lo)
    labels_norm.attrs['clip_hi'] = float(hi)
    labels_norm.attrs['train_std'] = float(train_std)
    labels_norm.attrs['train_mean'] = float(train_mean)

    return labels_norm


# =============================================================================
# 3.5 原始未来对数收益率（未归一化，供回测使用）
# =============================================================================
def make_raw_forward_returns(df, horizon):
    """
    生成原始未来 horizon 期对数收益率（未归一化），与 make_labels 共享对齐逻辑。

    - 未来收益 = np.log(df['close'].shift(-horizon) / df['close'])
    - 丢弃最后 horizon 行的 NaN（与 make_labels 一致）

    返回 Series（与 df 对齐，已丢弃最后 horizon 行），索引与 make_labels 输出相同。
    回测需要原始（未归一化）的未来收益，而非 [-1,1] 的标签。
    """
    future_log_ret = np.log(df['close'].shift(-horizon) / df['close'])
    return future_log_ret.dropna()


# =============================================================================
# 4. 滑窗生成
# =============================================================================
def make_windows(df, labels, L, raw_returns=None):
    """
    对齐 df 和 labels 后生成滑窗样本。

    - 第 i 个样本输入为 df.iloc[i:i+L]（形状 L×C）
    - 标签为 labels.iloc[i+L-1]（窗口最后一个时间步对应的标签）
    - num_samples = N - L + 1（对齐后）

    参数:
      raw_returns: 可选，与 labels 同索引的原始未来收益 Series。提供时，
                   额外返回与样本对齐的原始未来收益数组（供回测使用）。

    返回:
      若 raw_returns 为 None: (X, y, timestamps)
      否则: (X, y, raw_ret_values, timestamps)
        X: np.ndarray (num_samples, L, C)
        y: np.ndarray (num_samples,)
        raw_ret_values: np.ndarray (num_samples,) 原始未来对数收益
        timestamps: 窗口末尾时间戳列表
    """
    # 对齐 df 和 labels（取共同索引，避免未来标签泄露）
    common_idx = df.index.intersection(labels.index)
    df = df.loc[common_idx]
    labels = labels.loc[common_idx]

    N = len(df)
    if N < L:
        raise ValueError(f"对齐后长度 {N} 小于窗口长度 {L}")

    num_samples = N - L + 1
    C = df.shape[1]

    values = df.values.astype(np.float32)            # (N, C)
    label_values = labels.values.astype(np.float32)  # (N,)

    # 向量化滑窗：在 axis=0 上滑动窗口 L，结果形状 (num_samples, C, L)
    windows = np.lib.stride_tricks.sliding_window_view(values, L, axis=0)
    # 转置为 (num_samples, L, C) 并复制为连续内存
    X = windows.transpose(0, 2, 1).copy()

    # 标签：第 i 个样本对应窗口末尾索引 i+L-1
    y = label_values[L - 1: L - 1 + num_samples].copy()

    # 时间戳：窗口末尾
    timestamps = list(common_idx[L - 1: L - 1 + num_samples])

    # 原始未来收益（与标签共享对齐与滑窗逻辑，只是未做归一化裁剪）
    if raw_returns is not None:
        raw_aligned = raw_returns.loc[common_idx].values.astype(np.float32)
        raw_ret_values = raw_aligned[L - 1: L - 1 + num_samples].copy()
        return X, y, raw_ret_values, timestamps
    return X, y, timestamps


# =============================================================================
# 5. 按时间划分
# =============================================================================
def split_time(X, y, timestamps, ratios):
    """
    按时间顺序（样本顺序）划分，不随机打乱。

    ratios = [train_ratio, val_ratio, test_ratio]，和为 1。

    返回 dict:
      {'train': {'X':..., 'y':...}, 'val': {...}, 'test': {...},
       'timestamps': {'train': [...], 'val': [...], 'test': [...]}}
    """
    n = len(X)
    r_train, r_val, _r_test = ratios
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    # test 取剩余，避免比例误差导致丢样本

    split = {
        'train': {'X': X[:n_train], 'y': y[:n_train]},
        'val':   {'X': X[n_train:n_train + n_val], 'y': y[n_train:n_train + n_val]},
        'test':  {'X': X[n_train + n_val:], 'y': y[n_train + n_val:]},
        'timestamps': {
            'train': timestamps[:n_train],
            'val':   timestamps[n_train:n_train + n_val],
            'test':  timestamps[n_train + n_val:],
        },
    }
    return split


# =============================================================================
# 6. 构建 DataLoader
# =============================================================================
def get_dataloaders(split, batch_size, shuffle_train=True):
    """
    把 split 中的 numpy 数组转为 TensorDataset 和 DataLoader。
    train 可 shuffle，val/test 不 shuffle。

    返回 {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    def to_loader(seg, shuffle):
        X = torch.from_numpy(seg['X']).float()
        y = torch.from_numpy(seg['y']).float()
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return {
        'train': to_loader(split['train'], shuffle_train),
        'val':   to_loader(split['val'], shuffle=False),
        'test':  to_loader(split['test'], shuffle=False),
    }


# =============================================================================
# 7. 顶层流水线
# =============================================================================
def build_pipeline(cfg):
    """
    顶层函数：根据完整 cfg 调用上述函数构建数据管道。

    cfg: 完整 config 字典，含 'data' 和 'features' 子字典。

    流程：load_csv → add_features → make_labels → make_windows → split_time → get_dataloaders

    返回 dict:
      {
        'dataloaders': {'train':..., 'val':..., 'test':...},
        'split': split_time 的返回值（含 raw_forward_returns 子字典）,
        'C': int,            # 特征数
        'L': int,            # 窗口长度
        'label_stats': {...} # 标签归一化统计量（clip_lo, clip_hi, train_std）
        'raw_forward_returns': {'train':..., 'val':..., 'test':...}  # 原始未来收益（供回测）
      }
    """
    data_cfg = cfg['data']
    feat_cfg = cfg['features']

    # 1. 加载 CSV
    df = load_csv(data_cfg['csv_path'])

    # 2. 特征工程
    df = add_features(df, feat_cfg)

    # 3. 生成标签
    horizon = data_cfg['horizon']
    clip_q = data_cfg['label_clip_quantile']
    labels = make_labels(df, horizon, clip_quantile=clip_q)

    # 3.5 原始未来对数收益率（未归一化，供回测使用）
    raw_returns = make_raw_forward_returns(df, horizon)

    # 4. 滑窗（同时产出与样本对齐的原始未来收益）
    L = data_cfg['window_L']
    X, y, raw_ret, timestamps = make_windows(df, labels, L, raw_returns=raw_returns)

    # 5. 按时间划分
    split = split_time(X, y, timestamps, data_cfg['split_ratios'])

    # 同步切分原始未来收益（与 split_time 的切分逻辑保持一致）
    n = len(X)
    n_train = int(n * data_cfg['split_ratios'][0])
    n_val = int(n * data_cfg['split_ratios'][1])
    raw_split = {
        'train': raw_ret[:n_train],
        'val':   raw_ret[n_train:n_train + n_val],
        'test':  raw_ret[n_train + n_val:],
    }
    split['raw_forward_returns'] = raw_split

    # 6. 构建 DataLoader
    batch_size = cfg.get('train', {}).get('batch_size', 64)
    dataloaders = get_dataloaders(split, batch_size, shuffle_train=True)

    # 收集标签归一化统计量（用于后续反归一化或解释）
    label_stats = {
        'clip_lo': labels.attrs.get('clip_lo'),
        'clip_hi': labels.attrs.get('clip_hi'),
        'train_std': labels.attrs.get('train_std'),
    }

    return {
        'dataloaders': dataloaders,
        'split': split,
        'C': int(X.shape[2]),
        'L': L,
        'label_stats': label_stats,
        'raw_forward_returns': raw_split,
    }


# =============================================================================
# 快速逻辑测试
# =============================================================================
if __name__ == "__main__":
    # 项目根目录 = gtm/ 的父目录；CSV 已归类到 data/
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "BTCUSDT_1D_kline.csv")

    # 硬编码配置（与 config.yaml 默认值一致）
    FEAT_CFG = {
        'log_return_windows': [1, 3, 5, 10],
        'momentum_windows': [3, 5, 10],
        'volatility_windows': [5, 10, 20],
        'ma_short': 5,
        'ma_long': 20,
    }
    HORIZON = 5
    CLIP_Q = 0.05
    L = 30
    RATIOS = [0.6, 0.2, 0.2]

    print("=" * 70)
    print("量化交易模型数据管道 - 快速逻辑测试")
    print("=" * 70)

    all_pass = True

    # -------------------------------------------------------------------------
    # 测试 1：load_csv
    # -------------------------------------------------------------------------
    print("\n[测试 1] load_csv")
    df = load_csv(CSV_PATH)
    no_nan = not df.isna().any().any()
    is_ascending = df.index.equals(df.index.sort_values())
    is_datetime_idx = isinstance(df.index, pd.DatetimeIndex)
    print(f"  形状: {df.shape}")
    print(f"  无 NaN: {no_nan} -> {'PASS' if no_nan else 'FAIL'}")
    print(f"  升序: {is_ascending} -> {'PASS' if is_ascending else 'FAIL'}")
    print(f"  DatetimeIndex: {is_datetime_idx} -> {'PASS' if is_datetime_idx else 'FAIL'}")
    if not (no_nan and is_ascending and is_datetime_idx):
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 2：add_features
    # -------------------------------------------------------------------------
    print("\n[测试 2] add_features")
    df_feat = add_features(df, FEAT_CFG)
    no_nan_feat = not df_feat.isna().any().any()
    feat_count_ok = df_feat.shape[1] > 10
    print(f"  形状: {df_feat.shape}（原始 10 列 + 衍生 {df_feat.shape[1] - 10} 列）")
    print(f"  无 NaN: {no_nan_feat} -> {'PASS' if no_nan_feat else 'FAIL'}")
    print(f"  特征数 > 10: {feat_count_ok} -> {'PASS' if feat_count_ok else 'FAIL'}")
    if not (no_nan_feat and feat_count_ok):
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 3：make_labels
    # -------------------------------------------------------------------------
    print("\n[测试 3] make_labels")
    labels = make_labels(df_feat, HORIZON, clip_quantile=CLIP_Q)
    labels_in_range = bool(((labels >= -1.0) & (labels <= 1.0)).all())
    len_ok = (len(labels) == len(df_feat) - HORIZON)
    print(f"  标签长度: {len(labels)}（特征 DataFrame 长度 {len(df_feat)}，应少 {HORIZON}）")
    print(f"  长度正确: {len_ok} -> {'PASS' if len_ok else 'FAIL'}")
    print(f"  值域 ∈ [-1,1]: min={labels.min():.4f}, max={labels.max():.4f} -> {'PASS' if labels_in_range else 'FAIL'}")
    print(f"  归一化统计量: clip_lo={labels.attrs['clip_lo']:.6f}, "
          f"clip_hi={labels.attrs['clip_hi']:.6f}, train_std={labels.attrs['train_std']:.6f}")
    if not (labels_in_range and len_ok):
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 4：make_windows
    # -------------------------------------------------------------------------
    print("\n[测试 4] make_windows")
    X, y, timestamps = make_windows(df_feat, labels, L)
    # 对齐后 N = labels 长度（labels 是 df_feat 去掉最后 horizon 行）
    N_aligned = len(labels)
    expected_num = N_aligned - L + 1
    C = df_feat.shape[1]
    x_shape_ok = (X.shape == (expected_num, L, C))
    y_shape_ok = (y.shape == (expected_num,))
    x_no_nan = not np.isnan(X).any()
    y_no_nan = not np.isnan(y).any()
    ts_len_ok = (len(timestamps) == expected_num)
    print(f"  对齐后长度 N: {N_aligned}")
    print(f"  X 形状: {X.shape}, 期望 ({expected_num}, {L}, {C}) -> {'PASS' if x_shape_ok else 'FAIL'}")
    print(f"  y 形状: {y.shape}, 期望 ({expected_num},) -> {'PASS' if y_shape_ok else 'FAIL'}")
    print(f"  X 无 NaN: {x_no_nan} -> {'PASS' if x_no_nan else 'FAIL'}")
    print(f"  y 无 NaN: {y_no_nan} -> {'PASS' if y_no_nan else 'FAIL'}")
    print(f"  timestamps 长度: {len(timestamps)}, 期望 {expected_num} -> {'PASS' if ts_len_ok else 'FAIL'}")
    if not (x_shape_ok and y_shape_ok and x_no_nan and y_no_nan and ts_len_ok):
        all_pass = False

    # -------------------------------------------------------------------------
    # 测试 5：split_time
    # -------------------------------------------------------------------------
    print("\n[测试 5] split_time")
    split = split_time(X, y, timestamps, RATIOS)
    n_total = len(X)
    n_tr = len(split['train']['X'])
    n_va = len(split['val']['X'])
    n_te = len(split['test']['X'])
    # 检查无重叠：train 末尾时间戳 < val 开头 < val 末尾 < test 开头
    no_overlap = True
    if n_tr > 0 and n_va > 0:
        no_overlap = no_overlap and (split['timestamps']['train'][-1] < split['timestamps']['val'][0])
    if n_va > 0 and n_te > 0:
        no_overlap = no_overlap and (split['timestamps']['val'][-1] < split['timestamps']['test'][0])
    # 检查总样本数守恒
    count_conserved = (n_tr + n_va + n_te == n_total)
    # 比例近似 60/20/20
    r_tr = n_tr / n_total
    r_va = n_va / n_total
    r_te = n_te / n_total
    ratio_ok = abs(r_tr - 0.6) < 0.05 and abs(r_va - 0.2) < 0.05 and abs(r_te - 0.2) < 0.05
    print(f"  划分: train={n_tr}, val={n_va}, test={n_te} (总 {n_total})")
    print(f"  比例: train={r_tr:.3f}, val={r_va:.3f}, test={r_te:.3f}")
    print(f"  样本数守恒: {count_conserved} -> {'PASS' if count_conserved else 'FAIL'}")
    print(f"  无重叠（时间顺序）: {no_overlap} -> {'PASS' if no_overlap else 'FAIL'}")
    print(f"  比例近似 60/20/20: {ratio_ok} -> {'PASS' if ratio_ok else 'FAIL'}")
    if not (no_overlap and ratio_ok and count_conserved):
        all_pass = False

    # -------------------------------------------------------------------------
    # 额外测试：get_dataloaders & build_pipeline
    # -------------------------------------------------------------------------
    print("\n[额外测试] get_dataloaders & build_pipeline")
    try:
        dataloaders = get_dataloaders(split, batch_size=64, shuffle_train=True)
        xb, yb = next(iter(dataloaders['train']))
        dl_ok = (xb.shape[1:] == (L, C)) and (yb.dim() == 1)
        print(f"  train batch: X={tuple(xb.shape)}, y={tuple(yb.shape)} -> {'PASS' if dl_ok else 'FAIL'}")
        if not dl_ok:
            all_pass = False
    except Exception as e:
        print(f"  get_dataloaders 异常: {e} -> FAIL")
        all_pass = False

    try:
        full_cfg = {
            'data': {
                'csv_path': CSV_PATH,
                'split_ratios': RATIOS,
                'window_L': L,
                'horizon': HORIZON,
                'label_clip_quantile': CLIP_Q,
            },
            'features': FEAT_CFG,
            'train': {'batch_size': 64},
        }
        pipe = build_pipeline(full_cfg)
        pipe_ok = (pipe['C'] == C) and (pipe['L'] == L) and ('train' in pipe['dataloaders']) \
                  and (pipe['label_stats']['train_std'] is not None)
        print(f"  build_pipeline: C={pipe['C']}, L={pipe['L']}, "
              f"label_stats={pipe['label_stats']} -> {'PASS' if pipe_ok else 'FAIL'}")
        if not pipe_ok:
            all_pass = False
    except Exception as e:
        print(f"  build_pipeline 异常: {e} -> FAIL")
        all_pass = False

    # -------------------------------------------------------------------------
    # 汇总
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"全部测试结果: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("=" * 70)
