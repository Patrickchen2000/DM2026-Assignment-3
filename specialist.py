"""Class-2 specialist: binary LGBM on the class 1-vs-2 boundary (where 208/358 class-2 are lost).
Inject its signal into the main model's class-2 probability, tune boost beta with ROBUST CV
(mean+-std over user resamples + win-rate), NOT single OOF (which misled us before).
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


def calib(proba, y, rounds=2, lo=0.5, hi=2.5):
    cand = [c for c in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.8,2.0,2.5] if lo<=c<=hi]
    w = np.ones(N_CLASSES); best = macro_f1(y, (proba*w).argmax(1))
    for _ in range(rounds):
        for c in range(N_CLASSES):
            bw = w[c]
            for cv in cand:
                t = w.copy(); t[c]=cv; f = macro_f1(y, (proba*t).argmax(1))
                if f>best: best,bw=f,cv
            w[c]=bw
    return w


def robust_eval(proba, y, users, calib_w, n=300, seed=1):
    """mean+-std macro-F1 over half-user resamples, with the SAME calib applied."""
    uu = np.unique(users); rng = np.random.default_rng(seed); s=[]
    pred = (proba*calib_w).argmax(1)
    for _ in range(n):
        sel = rng.choice(uu, size=len(uu)//2, replace=False); m=np.isin(users, sel)
        s.append(macro_f1(y[m], pred[m]))
    return np.array(s)


def train_specialist(X, y, groups, Xt, pos_cls=2, neg_cls=(1,)):
    """Binary LGBM: pos_cls vs neg_cls, trained only on those samples.
    Returns OOF P(pos) over ALL train rows (0 where not in subset is meaningless,
    so we predict for every row at inference) and test P(pos)."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), np.float32); test = np.zeros(len(Xt), np.float32)
    subset_classes = (pos_cls,) + tuple(neg_cls)
    yb = (y == pos_cls).astype(int)
    for tr, va in sgkf.split(X, y, groups):
        # train only on subset classes within the train fold
        tr_sub = tr[np.isin(y[tr], subset_classes)]
        spec = LGBMClassifier(objective='binary', num_leaves=31, learning_rate=0.03,
            n_estimators=600, min_child_samples=15, reg_lambda=1.0, random_state=SEED,
            n_jobs=-1, verbose=-1, class_weight='balanced')
        spec.fit(X[tr_sub], yb[tr_sub])
        oof[va] = spec.predict_proba(X[va])[:, 1]
        test += spec.predict_proba(Xt)[:, 1] / N_FOLDS
    return oof, test


def inject(p_main, s, beta, cls=2):
    """Boost class `cls` in proba space by specialist score s in [0,1]."""
    p = p_main.copy()
    p[:, cls] = p[:, cls] * np.exp(beta * (s - 0.5))
    return p / p.sum(1, keepdims=True)


if __name__ == '__main__':
    tr, te = get_features(); tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
    feats = [c for c in tr.columns if c not in EXCLUDE]
    X = tr[feats].values.astype(np.float32); Xt = te[feats].values.astype(np.float32)
    y = tr['label'].values; groups = tr['user_id'].values

    # main = cached single-seed lgbm (prototype). Will scale to multiseed if specialist helps.
    p_main = np.load(f'{CACHE_DIR}/lgbm_oof.npy')
    p_main_te = np.load(f'{CACHE_DIR}/lgbm_test.npy')

    w0 = calib(p_main, y)
    base = robust_eval(p_main, y, groups, w0)
    print(f'MAIN (lgbm) robust macro-F1: {base.mean():.4f} +- {base.std():.4f}')

    for neg in [(1,), (1, 3)]:
        print(f'\n=== specialist class2 vs {neg} ===')
        s_oof, s_te = train_specialist(X, y, groups, Xt, pos_cls=2, neg_cls=neg)
        # AUC-ish: how well s separates class2 from neg on OOF
        from sklearn.metrics import roc_auc_score
        msk = np.isin(y, (2,)+neg)
        print(f'  specialist OOF AUC (2 vs {neg}): {roc_auc_score((y[msk]==2).astype(int), s_oof[msk]):.4f}')
        best = (None, base.mean())
        for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            p_adj = inject(p_main, s_oof, beta)
            w = calib(p_adj, y)
            r = robust_eval(p_adj, y, groups, w)
            wins = np.mean(r > base)
            flag = ''
            if r.mean() > best[1]: best = (beta, r.mean()); flag=' <-- best'
            print(f'  beta={beta:<4} robust F1 {r.mean():.4f}+-{r.std():.4f}  win-vs-main {wins*100:.0f}%{flag}')
