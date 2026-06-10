"""Reproduce the submitted model (public LB macro-F1 = 0.8032).
Multi-seed LightGBM over the 220 engineered features + inverse-frequency class weights
+ per-class OOF calibration + light per-class Markov label smoothing.
Run after features.py. Writes submission.csv with an ID-alignment check.
"""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')
from features import get_features, SAMPLE_SUB

SEED, N_FOLDS, N_CLASSES = 42, 5, 6
SEEDS = [42, 123, 456, 789, 2024]
EXCLUDE = {'window_id', 'user_id', 'label', 'index', 'file_id'}
ALPHA_PC = np.array([0.40, 0.30, 0.05, 0.05, 0.20, 0.20])   # per-class Markov strength
OUT = '/Users/patrick/Desktop/528/v3/submission.csv'


def macro_f1(y, p): return f1_score(y, p, average='macro')


def params(seed, cw):
    return dict(objective='multiclass', num_class=N_CLASSES, num_leaves=63, learning_rate=0.03,
                n_estimators=1000, min_child_samples=10, reg_alpha=0.05, reg_lambda=0.5,
                class_weight=cw, random_state=seed, n_jobs=-1, verbose=-1)


def empirical_transition(tr):
    df = tr[['user_id', 'window_id', 'label']].copy()
    df['window_id'] = df['window_id'].astype(int)
    df = df.sort_values(['user_id', 'window_id'])
    T = np.zeros((N_CLASSES, N_CLASSES))
    for _, g in df.groupby('user_id'):
        l = g['label'].values
        for a, b in zip(l[:-1], l[1:]):
            T[a, b] += 1
    T += 1e-6
    return T / T.sum(1, keepdims=True)


def calibrate(proba, y):
    cand = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
    w = np.ones(N_CLASSES); best = macro_f1(y, (proba * w).argmax(1))
    for c in range(N_CLASSES):
        bw = w[c]
        for cv in cand:
            t = w.copy(); t[c] = cv
            f = macro_f1(y, (proba * t).argmax(1))
            if f > best: best, bw = f, cv
        w[c] = bw
    return w, best


def greedy_markov(proba, meta, cw, T):
    order = meta.sort_values(['user_id', 'window_id']).index.values
    pos = {idx: i for i, idx in enumerate(meta.index.values)}
    preds = np.zeros(len(proba), dtype=int)
    for uid, g in meta.loc[order].groupby('user_id', sort=False):
        rows = np.array([pos[i] for i in g.index.values])
        p = proba[rows] * cw; p = p / p.sum(1, keepdims=True)
        prev = None
        for k, r in enumerate(rows):
            blended = p[k] if prev is None else (1 - ALPHA_PC[prev]) * p[k] + ALPHA_PC[prev] * T[prev]
            prev = int(np.argmax(blended)); preds[r] = prev
    return preds


if __name__ == '__main__':
    tr, te = get_features(); tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
    feats = [c for c in tr.columns if c not in EXCLUDE]
    X = tr[feats].values.astype(np.float32); Xt = te[feats].values.astype(np.float32)
    y = tr['label'].values; groups = tr['user_id'].values

    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    cwf = counts.sum() / (N_CLASSES * counts); cw = dict(enumerate(cwf / cwf.min()))
    sw = np.array([cw[l] for l in y])

    # OOF (seed 42) for calibration
    oof = np.zeros((len(X), N_CLASSES), np.float32)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for trn, va in sgkf.split(X, y, groups):
        m = LGBMClassifier(**params(SEED, cw)); m.fit(X[trn], y[trn], sample_weight=sw[trn])
        oof[va] = m.predict_proba(X[va])
    print(f'OOF macro-F1 (argmax): {macro_f1(y, oof.argmax(1)):.4f}')
    weights, f_cal = calibrate(oof, y)
    print(f'OOF macro-F1 (calibrated): {f_cal:.4f}   weights={weights}')

    # multi-seed full-data models -> averaged test proba
    test_proba = np.zeros((len(Xt), N_CLASSES), np.float32)
    for s in SEEDS:
        m = LGBMClassifier(**params(s, cw)); m.fit(X, y, sample_weight=sw)
        test_proba += m.predict_proba(Xt) / len(SEEDS)
        print(f'  seed {s} done')

    T = empirical_transition(tr)
    meta_te = te[['user_id', 'window_id']].copy(); meta_te['window_id'] = meta_te['window_id'].astype(int)
    pred = greedy_markov(test_proba, meta_te, weights, T)

    sample = pd.read_csv(SAMPLE_SUB)
    d = dict(zip(meta_te['window_id'].values, pred))
    assert len(set(sample.Id) - set(d)) == 0, 'ID mismatch with sample_submission!'
    sample['Label'] = sample['Id'].map(d).astype(int)
    assert sample['Label'].between(0, 5).all()
    sample.to_csv(OUT, index=False)
    print(f'\nSaved {OUT}  (rows={len(sample)}, IDs aligned)')
    print('label distribution:', dict(zip(*np.unique(pred, return_counts=True))))
