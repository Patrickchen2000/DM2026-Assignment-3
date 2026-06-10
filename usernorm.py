"""Per-user relative features: subtract each user's own mean of every feature.
Hypothesis: per-user wrist-orientation variance masks the activity signal for rare
classes (esp. class 2). Removing the user baseline should expose it. Leak-free &
transductive (uses only a user's own windows, no labels). Validated by robust CV.
"""
import numpy as np, warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')
from features import get_features, CACHE_DIR

SEED, N_FOLDS, N_CLASSES = 42, 5, 6
EXCLUDE = {'window_id', 'user_id', 'label', 'index', 'file_id'}


def macro_f1(y, p): return f1_score(y, p, average='macro')


def add_user_relative(df, feat_cols):
    """For each feature, add value - user_mean and value - user_median (per user)."""
    g = df.groupby('user_id')[feat_cols]
    umean = g.transform('mean')
    out = df.copy()
    for c in feat_cols:
        out[c + '_urel'] = df[c] - umean[c]
    return out


def oof_lgbm(X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    c = np.bincount(y, minlength=N_CLASSES).astype(float); w = c.sum()/(N_CLASSES*c)
    cw = dict(enumerate(w/w.min())); sw = np.array([cw[l] for l in y])
    oof = np.zeros((len(X), N_CLASSES), np.float32)
    for tr, va in sgkf.split(X, y, groups):
        m = LGBMClassifier(objective='multiclass', num_class=N_CLASSES, num_leaves=63,
            learning_rate=0.03, n_estimators=1000, min_child_samples=10, reg_alpha=0.05,
            reg_lambda=0.5, random_state=SEED, n_jobs=-1, verbose=-1, class_weight=cw)
        m.fit(X[tr], y[tr], sample_weight=sw[tr])
        oof[va] = m.predict_proba(X[va])
    return oof


def calib(proba, y, rounds=2):
    cand=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.8,2.0,2.5]
    w=np.ones(N_CLASSES); best=macro_f1(y,(proba*w).argmax(1))
    for _ in range(rounds):
        for cc in range(N_CLASSES):
            bw=w[cc]
            for cv in cand:
                t=w.copy(); t[cc]=cv; f=macro_f1(y,(proba*t).argmax(1))
                if f>best: best,bw=f,cv
            w[cc]=bw
    return w


def robust(proba, y, users, w, n=300):
    uu=np.unique(users); rng=np.random.default_rng(1); s=[]; pred=(proba*w).argmax(1)
    for _ in range(n):
        sel=rng.choice(uu,len(uu)//2,replace=False); m=np.isin(users,sel)
        s.append(macro_f1(y[m],pred[m]))
    return np.array(s)


if __name__ == '__main__':
    tr, te = get_features(); tr=tr.reset_index(drop=True)
    feat_cols = [c for c in tr.columns if c not in EXCLUDE]
    y = tr['label'].values; groups = tr['user_id'].values

    # baseline (cached)
    base_oof = np.load(f'{CACHE_DIR}/lgbm_oof.npy')
    wb = calib(base_oof, y); rb = robust(base_oof, y, groups, wb)
    fb = f1_score(y, (base_oof*wb).argmax(1), average=None)
    print(f'BASE: robust {rb.mean():.4f}+-{rb.std():.4f}  class2 F1 {fb[2]:.3f}')

    # add per-user relative features
    tr2 = add_user_relative(tr, feat_cols)
    feat2 = feat_cols + [c+'_urel' for c in feat_cols]
    X2 = tr2[feat2].values.astype(np.float32)
    print(f'features: {len(feat_cols)} -> {len(feat2)}  (training OOF, ~4min)')
    oof2 = oof_lgbm(X2, y, groups)
    np.save(f'{CACHE_DIR}/lgbm_urel_oof.npy', oof2)
    w2 = calib(oof2, y); r2 = robust(oof2, y, groups, w2)
    f2 = f1_score(y, (oof2*w2).argmax(1), average=None)
    print(f'USER-REL: robust {r2.mean():.4f}+-{r2.std():.4f}  class2 F1 {f2[2]:.3f}  win-vs-base {np.mean(r2>rb)*100:.0f}%')
    print('per-class F1 base -> user-rel:')
    for c in range(6): print(f'  class{c}: {fb[c]:.3f} -> {f2[c]:.3f} ({f2[c]-fb[c]:+.3f})')

    # ensemble base + user-rel
    ens = 0.5*base_oof + 0.5*oof2; we = calib(ens, y); re = robust(ens, y, groups, we)
    print(f'\nENSEMBLE base+userrel: robust {re.mean():.4f}+-{re.std():.4f}  win-vs-base {np.mean(re>rb)*100:.0f}%')
