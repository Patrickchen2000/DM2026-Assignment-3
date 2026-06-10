"""Feature engineering + caching for HAR v3.
Reuses the proven v2 feature set, adds caching to .npz so we only parse
the 11020 + 6849 CSV files once.
"""
import numpy as np
import pandas as pd
import os, glob
from scipy import stats
from scipy.fft import fft
from scipy.stats import entropy as sp_entropy

CHANNELS = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
BASE_DIR = '/Users/patrick/Desktop/528/nycu-data-mining-assignment-3'
BASE_TRAIN = os.path.join(BASE_DIR, 'train', 'train')
BASE_TEST = os.path.join(BASE_DIR, 'test', 'test')
SAMPLE_SUB = os.path.join(BASE_DIR, 'sample_submission.csv')
CACHE_DIR = '/Users/patrick/Desktop/528/v3/cache'
os.makedirs(CACHE_DIR, exist_ok=True)
LABEL_COL = 'label'


def summary_stats(arr, prefix):
    return {
        f'{prefix}_mean': np.mean(arr),
        f'{prefix}_std': np.std(arr),
        f'{prefix}_min': np.min(arr),
        f'{prefix}_max': np.max(arr),
        f'{prefix}_median': np.median(arr),
        f'{prefix}_q25': np.percentile(arr, 25),
        f'{prefix}_q75': np.percentile(arr, 75),
        f'{prefix}_kurtosis': stats.kurtosis(arr),
        f'{prefix}_skew': stats.skew(arr),
    }


def fft_features(arr, prefix, n_low=5):
    N = len(arr)
    freqs = np.abs(fft(arr))[:N // 2]
    power = freqs ** 2
    p_norm = power / (power.sum() + 1e-10)
    return {
        f'{prefix}_fft_low_power': np.sum(power[:n_low]),
        f'{prefix}_fft_peak_freq': float(np.argmax(power)),
        f'{prefix}_fft_entropy': sp_entropy(p_norm + 1e-10),
    }


def extract_window_features(df_win, user_position):
    feats = {}
    for ch in CHANNELS:
        feats.update(summary_stats(df_win[ch].values, ch))
    mx, my, mz = df_win['mean_x'].values, df_win['mean_y'].values, df_win['mean_z'].values
    sx, sy, sz = df_win['std_x'].values, df_win['std_y'].values, df_win['std_z'].values
    vm_mean = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    vm_std = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    feats.update(summary_stats(vm_mean, 'vm_mean'))
    feats.update(summary_stats(vm_std, 'vm_std'))
    feats.update(summary_stats(np.abs(vm_mean - 1.0), 'grav_delta'))
    n_seg, seg_len = 5, len(df_win) // 5
    for ch in CHANNELS:
        arr = df_win[ch].values
        for i in range(n_seg):
            seg = arr[i * seg_len:(i + 1) * seg_len]
            feats[f'{ch}_seg{i}_mean'] = np.mean(seg)
            feats[f'{ch}_seg{i}_std'] = np.std(seg)
            feats[f'{ch}_seg{i}_range'] = np.max(seg) - np.min(seg)
    for ch in CHANNELS:
        d = np.diff(df_win[ch].values)
        feats[f'{ch}_diff_mean'] = np.mean(d)
        feats[f'{ch}_diff_std'] = np.std(d)
        feats[f'{ch}_diff_abs_mean'] = np.mean(np.abs(d))
    for ch in CHANNELS:
        feats.update(fft_features(df_win[ch].values, ch))
    for a, b in [('mean_x', 'mean_y'), ('mean_x', 'mean_z'), ('mean_y', 'mean_z')]:
        c = np.corrcoef(df_win[a].values, df_win[b].values)[0, 1]
        feats[f'corr_{a}_{b}'] = float(c) if np.isfinite(c) else 0.0
    for ch in ['mean_x', 'mean_y', 'mean_z']:
        s = df_win[ch].values - df_win[ch].mean()
        norm = np.dot(s, s) + 1e-10
        for lag in [1, 5, 10]:
            feats[f'{ch}_acf{lag}'] = float(np.dot(s[:-lag], s[lag:]) / norm)
    feats['user_position'] = user_position
    return feats


def load_split(base_dir, is_train=True):
    all_dfs = []
    csv_files = sorted(glob.glob(os.path.join(base_dir, '*', '*.csv')))
    for fpath in csv_files:
        parts = fpath.replace('\\', '/').split('/')
        user_id = parts[-2]
        win_id = parts[-1][:-4]
        df = pd.read_csv(fpath, header=0)
        df.columns = df.columns.str.strip().str.lower()
        df['window_id'] = win_id
        df['user_id'] = user_id
        all_dfs.append(df)
    raw = pd.concat(all_dfs, ignore_index=True)
    if is_train:
        raw[LABEL_COL] = raw[LABEL_COL].astype(int)
    return raw


def build_features(raw_df):
    keep = CHANNELS + ['window_id', 'user_id']
    if LABEL_COL in raw_df.columns:
        keep.append(LABEL_COL)
    raw_df = raw_df[[c for c in keep if c in raw_df.columns]].copy()
    meta = raw_df.drop_duplicates('window_id')[['window_id', 'user_id']].copy()
    meta = meta.sort_values(['user_id', 'window_id'])
    meta['user_position'] = meta.groupby('user_id').cumcount()
    grp_size = meta.groupby('user_id')['window_id'].transform('count') - 1
    meta['user_position'] = meta['user_position'] / grp_size.clip(lower=1)
    pos_map = meta.set_index('window_id')['user_position'].to_dict()
    rows = []
    for wid, df_win in raw_df.groupby('window_id'):
        feat = extract_window_features(df_win, pos_map.get(wid, 0.5))
        feat['window_id'] = wid
        feat['user_id'] = df_win['user_id'].iloc[0]
        rows.append(feat)
    return pd.DataFrame(rows)


def get_features():
    """Return (train_feat, test_feat) DataFrames, cached as parquet."""
    tr_path = os.path.join(CACHE_DIR, 'train_feat.parquet')
    te_path = os.path.join(CACHE_DIR, 'test_feat.parquet')
    if os.path.exists(tr_path) and os.path.exists(te_path):
        return pd.read_parquet(tr_path), pd.read_parquet(te_path)
    print('Loading raw CSVs (one-time)...')
    train_raw = load_split(BASE_TRAIN, is_train=True)
    test_raw = load_split(BASE_TEST, is_train=False)
    print(f'  train_raw {train_raw.shape}, test_raw {test_raw.shape}')
    print('Building train features...')
    train_feat = build_features(train_raw)
    print('Building test features...')
    test_feat = build_features(test_raw)
    label_map = train_raw.drop_duplicates('window_id').set_index('window_id')[LABEL_COL]
    train_feat['label'] = train_feat['window_id'].map(label_map)
    train_feat.to_parquet(tr_path)
    test_feat.to_parquet(te_path)
    print(f'Cached. train {train_feat.shape}, test {test_feat.shape}')
    return train_feat, test_feat


if __name__ == '__main__':
    tr, te = get_features()
    print('train', tr.shape, 'test', te.shape)
    print(tr['label'].value_counts().sort_index())
