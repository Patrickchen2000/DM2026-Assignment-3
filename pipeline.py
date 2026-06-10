"""HAR v3 core pipeline.
Step 2-3: base LGBM OOF + OOF-validated post-processing (greedy Markov vs Viterbi).
The whole point: measure post-processing on OOF so we can tune it instead of guessing.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import warnings, os
warnings.filterwarnings('ignore')

from features import get_features, SAMPLE_SUB

SEED, N_FOLDS, N_CLASSES = 42, 5, 6
CACHE_DIR = '/Users/patrick/Desktop/528/v3/cache'
EXCLUDE = {'window_id', 'user_id', 'label', 'index', 'file_id'}

LGBM_PARAMS = dict(
    objective='multiclass', num_class=N_CLASSES, num_leaves=63,
    learning_rate=0.03, n_estimators=1000, min_child_samples=10,
    reg_alpha=0.05, reg_lambda=0.5, random_state=SEED, n_jobs=-1, verbose=-1,
)


def macro_f1(y, p):
    return f1_score(y, p, average='macro')


def empirical_transition(train_feat):
    """P(next | current) from training labels, ordered by user+window."""
    df = train_feat[['user_id', 'window_id', 'label']].copy()
    df = df.sort_values(['user_id', 'window_id'])
    T = np.zeros((N_CLASSES, N_CLASSES))
    for _, g in df.groupby('user_id'):
        labs = g['label'].values
        for a, b in zip(labs[:-1], labs[1:]):
            T[a, b] += 1
    T += 1e-6
    T = T / T.sum(axis=1, keepdims=True)
    return T


def viterbi_decode(log_emis, log_trans, log_prior):
    """Standard Viterbi. log_emis: (T, C). Returns (T,) labels."""
    Tn, C = log_emis.shape
    dp = np.full((Tn, C), -np.inf)
    bp = np.zeros((Tn, C), dtype=int)
    dp[0] = log_prior + log_emis[0]
    for t in range(1, Tn):
        scores = dp[t - 1][:, None] + log_trans  # (C_prev, C_next)
        bp[t] = np.argmax(scores, axis=0)
        dp[t] = scores[bp[t], np.arange(C)] + log_emis[t]
    path = np.zeros(Tn, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(Tn - 2, -1, -1):
        path[t] = bp[t + 1, path[t]]
    return path


def decode_all(proba, meta, method, cw=None, T=None, tau=1.0, alpha_pc=None):
    """Decode per user. meta has user_id, window_id aligned to proba rows.
    Returns predictions aligned to proba rows.
    method: 'argmax' | 'greedy' | 'viterbi'
    """
    cw = np.ones(N_CLASSES) if cw is None else cw
    order = meta.sort_values(['user_id', 'window_id']).index.values
    pos = {idx: i for i, idx in enumerate(meta.index.values)}
    preds = np.zeros(len(proba), dtype=int)

    if method == 'argmax':
        return np.argmax(proba * cw, axis=1)

    # group ordered indices by user
    df = meta.loc[order]
    if method == 'viterbi':
        log_trans = np.log(np.power(T, 1.0 / tau) + 1e-12) if T is not None else np.zeros((N_CLASSES, N_CLASSES))
        log_prior = np.log(cw / cw.sum() + 1e-12)
    for uid, g in df.groupby('user_id', sort=False):
        idxs = g.index.values
        rows = np.array([pos[i] for i in idxs])
        p = proba[rows] * cw
        p = p / p.sum(axis=1, keepdims=True)
        if method == 'greedy':
            prev = None
            for k, r in enumerate(rows):
                if prev is None:
                    blended = p[k]
                else:
                    a = alpha_pc[prev]
                    blended = (1 - a) * p[k] + a * T[prev]
                lab = int(np.argmax(blended))
                preds[r] = lab
                prev = lab
        elif method == 'viterbi':
            path = viterbi_decode(np.log(p + 1e-12), log_trans, log_prior)
            preds[rows] = path
    return preds


def calibrate_weights(proba, y, base_w=None):
    """Greedy per-class weight search to maximize OOF macro-F1 (argmax)."""
    cand = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
    w = np.ones(N_CLASSES) if base_w is None else base_w.copy()
    best = macro_f1(y, np.argmax(proba * w, axis=1))
    for c in range(N_CLASSES):
        bw = w[c]
        for cv in cand:
            t = w.copy(); t[c] = cv
            f = macro_f1(y, np.argmax(proba * t, axis=1))
            if f > best:
                best, bw = f, cv
        w[c] = bw
    return w, best


def train_oof(X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    cw_inv = counts.sum() / (N_CLASSES * counts)
    cw_dict = dict(enumerate(cw_inv / cw_inv.min()))
    sw = np.array([cw_dict[l] for l in y])
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        clf = LGBMClassifier(**LGBM_PARAMS, class_weight=cw_dict)
        clf.fit(X[tr], y[tr], sample_weight=sw[tr])
        oof[va] = clf.predict_proba(X[va])
    return oof, cw_dict, sw


if __name__ == '__main__':
    train_feat, test_feat = get_features()
    FEATS = [c for c in train_feat.columns if c not in EXCLUDE]
    X = train_feat[FEATS].values.astype(np.float32)
    y = train_feat['label'].values
    groups = train_feat['user_id'].values
    train_feat = train_feat.reset_index(drop=True)

    print(f'Features: {len(FEATS)}')
    oof, cw_dict, sw = train_oof(X, y, groups)
    np.save(os.path.join(CACHE_DIR, 'oof_base.npy'), oof)

    T = empirical_transition(train_feat)
    np.save(os.path.join(CACHE_DIR, 'transition.npy'), T)
    print('\nEmpirical transition (diag stay-prob):', np.round(np.diag(T), 3))

    meta = train_feat[['user_id', 'window_id']].copy()
    meta['window_id'] = meta['window_id'].astype(int)

    # --- baseline argmax ---
    f_raw = macro_f1(y, np.argmax(oof, axis=1))
    print(f'\n[1] OOF argmax (no calib):        {f_raw:.4f}')

    cw_arr, f_cal = calibrate_weights(oof, y)
    print(f'[2] OOF argmax + class-weight cal: {f_cal:.4f}   w={np.round(cw_arr,2)}')

    # --- greedy markov on OOF (the previously-unmeasured step) ---
    alpha_pc = np.array([0.40, 0.30, 0.05, 0.05, 0.20, 0.20])
    pred_g = decode_all(oof, meta, 'greedy', cw=cw_arr, T=T, alpha_pc=alpha_pc)
    f_g = macro_f1(y, pred_g)
    print(f'[3] OOF greedy Markov (v2 params): {f_g:.4f}')

    # --- viterbi on OOF, scan tau ---
    print('\n[4] OOF Viterbi, scan temperature tau:')
    best_tau, best_fv = None, -1
    for tau in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        pv = decode_all(oof, meta, 'viterbi', cw=cw_arr, T=T, tau=tau)
        fv = macro_f1(y, pv)
        flag = ''
        if fv > best_fv:
            best_fv, best_tau, flag = fv, tau, '  <-- best'
        print(f'    tau={tau:<4}  F1={fv:.4f}{flag}')
    print(f'\nSummary: argmax {f_raw:.4f} -> +calib {f_cal:.4f} -> greedy {f_g:.4f} -> viterbi(tau={best_tau}) {best_fv:.4f}')
