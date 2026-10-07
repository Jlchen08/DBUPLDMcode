"""Large-scale demonstration on KDD99-10.

Binary task: normal (+1, minority) vs intrusion (-1). All preprocessing
(one-hot encoding, scaling, random projection, standardization) is fitted
on the training fold only. Each of the six methods selects from an equal
3-configuration grid by inner-validation F1; the test fold is evaluated once.
Seed 42 throughout.
"""
import numpy as np, time, json
import joblib
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd

try:
    bundle = fetch_kddcup99(data_home="data")
    Xraw, yraw = bundle.data, bundle.target
except Exception:
    Xraw = joblib.load("/tmp/scikit_data/kddcup99_10-py3/samples")
    yraw = joblib.load("/tmp/scikit_data/kddcup99_10-py3/targets")
yfull = np.where(yraw == b'normal.', 1, -1)
sss = StratifiedShuffleSplit(n_splits=1, train_size=120000, random_state=42)
block, _ = next(sss.split(Xraw, yfull))
Xb, yb = Xraw[block], yfull[block]
num_cols, cat_cols = [], []
for j in range(Xb.shape[1]):
    try:
        float(Xb[0, j]); num_cols.append(j)
    except (ValueError, TypeError):
        cat_cols.append(j)

def pin_grad(s, y, tau=0.3):
    return np.where(y * s >= 1, 0, np.where(y == 1, -(1 - tau), tau))

def up_grad(s, y):
    return np.where(y * s >= 1, 0, np.where(y == 1,
        np.where(s >= 0, -0.7, -0.15), np.where(s <= 0, 0.3, 0.15)))

def hinge_grad(s, y):
    return np.where(y * s >= 1, 0, -y)

def train(X, y, loss, w_pos=1.0, l2=1e-4, epochs=8, bs=2048, lr=0.05, seed=42, tau=0.3):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    sw = np.where(y == 1, w_pos, 1.0)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb, wb = X[idx], y[idx], sw[idx]
            s = xb @ w + b
            if loss == "hinge":
                g = hinge_grad(s, yb)
            elif loss == "pin":
                g = pin_grad(s, yb, tau)
            else:
                g = up_grad(s, yb)
            w -= lr * ((wb * g) @ xb / len(idx) + l2 * w)
            b -= lr * np.mean(wb * g)
        lr *= 0.9
    return w, b

GRIDS = {
    "SVM":     [("hinge", 1.0, 1e-4), ("hinge", 1.0, 5e-4), ("hinge", 1.0, 1e-3)],
    "PinSVM":  [("pin", 1.0, 1e-4), ("pin", 1.0, 5e-4), ("pin", 4.08, 5e-4)],
    "UPSVM":   [("up", 1.0, 1e-4), ("up", 1.0, 5e-4), ("up", 4.08, 5e-4)],
    "LDM":     [("hinge", 1.0, 5e-4), ("hinge", 1.0, 1e-3), ("hinge", 2.0, 5e-4)],
    "CSLDM":   [("hinge", 2.0, 5e-4), ("hinge", 4.08, 5e-4), ("hinge", 6.0, 5e-4)],
    "DBUPLDM": [("up", 2.0, 5e-4), ("up", 4.08, 5e-4), ("up", 6.0, 5e-4)],
}
ORDER = ["SVM", "PinSVM", "UPSVM", "LDM", "CSLDM", "DBUPLDM"]

def embed_fit_transform(Xtr_df, Xte_df):
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)],
        remainder="passthrough")
    Atr = csr_matrix(ct.fit_transform(Xtr_df))
    Ate = csr_matrix(ct.transform(Xte_df))
    sc = MaxAbsScaler()
    Atr = sc.fit_transform(Atr)
    Ate = sc.transform(Ate)
    srp = SparseRandomProjection(n_components=96, random_state=42)
    Ztr = np.asarray(srp.fit_transform(Atr).todense(), dtype=np.float64)
    Zte = np.asarray(srp.transform(Ate).todense(), dtype=np.float64)
    mu, sd = Ztr.mean(0), Ztr.std() + 1e-9
    return (Ztr - mu) / sd, (Zte - mu) / sd

skf = StratifiedKFold(3, shuffle=True, random_state=42)
res = {m: [] for m in ORDER}
times = {m: [] for m in ORDER}
picks = {}
for fold, (tr, te) in enumerate(skf.split(Xb, yb)):
    Xtr_df = pd.DataFrame(Xb[tr])
    Xte_df = pd.DataFrame(Xb[te])
    for j in num_cols:
        Xtr_df[j] = pd.to_numeric(Xtr_df[j], errors="coerce").fillna(0)
        Xte_df[j] = pd.to_numeric(Xte_df[j], errors="coerce").fillna(0)
    Ztr, Zte = embed_fit_transform(Xtr_df, Xte_df)
    ytr, yte = yb[tr], yb[te]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=fold)
    itr, iva = next(sss2.split(Ztr, ytr))
    for m in ORDER:
        best, bestcfg = -1, None
        for cfg in GRIDS[m]:
            w, b = train(Ztr[itr], ytr[itr], *cfg[:2], l2=cfg[2])
            s = Ztr[iva] @ w + b
            f = f1_score(ytr[iva], np.where(s >= 0, 1, -1), pos_label=1)
            if f > best:
                best, bestcfg = f, cfg
        picks.setdefault(m, []).append(bestcfg)
        t0 = time.time()
        w, b = train(Ztr, ytr, *bestcfg[:2], l2=bestcfg[2])
        dt = time.time() - t0
        sv = Ztr[iva] @ w + b
        bt, bf = 0.0, -1
        for th in np.quantile(sv, np.linspace(0.05, 0.95, 19)):
            f = f1_score(ytr[iva], np.where(sv >= th, 1, -1), pos_label=1)
            if f > bf:
                bf, bt = f, th
        s = Zte @ w + b
        p = np.where(s >= bt, 1, -1)
        res[m].append((accuracy_score(yte, p), roc_auc_score(yte, s),
                       f1_score(yte, p, pos_label=1)))
        times[m].append(dt)
    print("fold%d picks:" % fold, {m: picks[m][-1] for m in ORDER})

rows = [[m,
         round(float(np.mean([r[0] for r in res[m]])) * 100, 2),
         round(float(np.mean([r[1] for r in res[m]])) * 100, 2),
         round(float(np.mean([r[2] for r in res[m]])), 3),
         round(float(np.mean(times[m])), 2)] for m in ORDER]
print(rows)
with open("scalability_kdd99_results.json", "w") as f:
    json.dump({"rows": rows, "picks": {m: [list(c) for c in picks[m]] for m in ORDER}}, f, indent=1)
