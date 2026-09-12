"""Learned prediction head over a tracer's per-skill states (downstream of the state; never changes it).

    ../.venv/bin/python head.py --tracer gauss+lmkt --cv mathdial --features full      # 5-group cross-fitting over dialogues
    ../.venv/bin/python head.py --tracer gauss+lmkt --cv comta --features full         # uses CoMTA's own folds
    ../.venv/bin/python head.py --tracer gauss+lmkt --fit mathdial --eval comta         # fit on all MathDial dialogues, transfer

Inputs: out/<tracer>_<dataset>_fold*.pkl (tracer trajectories) and head/gpt-4.1_<dataset>.json (LLM-read weights,
difficulty, all-skills flag per predicted turn). An L2 logistic regression with C tuned on a held-out fifth of the
training dialogues. Output: out/<tracer>+head-<features>_<dataset>_fold*.pkl whose records carry "head_preds" {t: p};
metrics.py then scores prediction from the head while direction / counterfactual metrics stay on the states.
"""
import argparse, glob, json, os, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); HEAD = os.path.join(HERE, "head")
FEATS = {"base": ["mean", "min", "max", "n_skills", "turn"],
         "weights": ["mean", "min", "max", "n_skills", "turn", "wmean", "all"],
         "full": ["mean", "min", "max", "n_skills", "turn", "wmean", "all", "difficulty", "wmean_x_diff"]}


def safe_auc(y, p):
    y = np.asarray(y); return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")


def rows(rec, hf):
    """One (t, label, features) per predicted turn t >= 1."""
    out = []
    for t in range(1, len(rec["labels"])):
        kcs = rec["kcs"][t]
        if not kcs: continue
        s = rec["states"][t - 1][kcs]
        if np.isnan(s).all(): continue
        s = np.where(np.isnan(s), np.nanmean(s), s)
        h = (hf or {}).get(str(t), {})
        w = np.array([float(h.get("w", {}).get(str(k), 1.0 / len(kcs))) for k in kcs]) if h else np.full(len(kcs), 1.0 / len(kcs))
        w = w / (w.sum() or 1.0); diff = float(h.get("difficulty", 0.5)) if h else 0.5
        wm = float((w * s).sum())
        out.append((t, int(rec["labels"][t]), {"mean": s.mean(), "min": s.min(), "max": s.max(), "n_skills": len(kcs), "turn": t,
                                               "wmean": wm, "all": float(bool(h.get("all", False))) if h else 0.0,
                                               "difficulty": diff, "wmean_x_diff": wm * diff}))
    return out


def load(tracer, dataset):
    files = [f for f in sorted(glob.glob(os.path.join(OUT, f"{tracer}_{dataset}_fold*.pkl"))) if "_partial" not in f and "+head" not in f]
    hf = json.load(open(os.path.join(HEAD, f"gpt-4.1_{dataset}.json")))
    return [(f, pickle.load(open(f, "rb"))) for f in files], hf


def group_of(dataset, file_idx, dialogue_idx):
    return file_idx if dataset == "comta" else dialogue_idx % 5   # CoMTA: its own folds; MathDial: dialogues split by index


def matrix(items, feats):
    X = np.array([[f[k] for k in feats] for _, _, f in items]); y = np.array([lab for _, lab, _ in items]); return X, y


def fit(train_items, feats, seed=0):
    """L2 logistic regression; C chosen on a held-out fifth of the training dialogues (grouped by dialogue)."""
    rng = np.random.default_rng(seed)
    dids = sorted({d for d, _ in train_items}); rng.shuffle(dids); hold = set(dids[: max(1, len(dids) // 5)])
    tr = [it for d, it in train_items if d not in hold]; va = [it for d, it in train_items if d in hold]
    Xtr, ytr = matrix(tr, feats); Xva, yva = matrix(va, feats)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; z = lambda X: (X - mu) / sd
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        m = LogisticRegression(C=C, max_iter=2000).fit(z(Xtr), ytr)
        ll = log_loss(yva, np.clip(m.predict_proba(z(Xva))[:, 1], 1e-6, 1 - 1e-6))
        if best is None or ll < best[0]: best = (ll, C, m)
    ll, C, m = best
    Xall, yall = matrix([it for _, it in train_items], feats)   # refit on all training dialogues at the chosen C
    m = LogisticRegression(C=C, max_iter=2000).fit(z(Xall), yall)
    return m, z, C, ll, len(yall)


def predict_into(m, z, feats, d, hf, keep):
    """Write head_preds into the records of dump d for dialogues in `keep` (None = all); return (auc, n)."""
    ys, ps = [], []
    for r in d["records"]:
        idx = int(r["dialogue_idx"])
        if keep is not None and idx not in keep: r["head_preds"] = {}; continue
        preds = {}
        for t, lab, f in rows(r, hf.get(str(idx))):
            p = float(m.predict_proba(z(np.array([[f[k] for k in feats]])))[0, 1]); preds[t] = p; ys.append(lab); ps.append(p)
        r["head_preds"] = preds
    return safe_auc(ys, ps), len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default="gauss+lmkt"); ap.add_argument("--features", default="full", choices=list(FEATS))
    ap.add_argument("--cv", default=None, help="dataset to cross-fit on"); ap.add_argument("--fit", default=None); ap.add_argument("--eval", nargs="*", default=[])
    a = ap.parse_args(); feats = FEATS[a.features]; tag = f"{a.tracer}+head-{a.features}"

    if a.cv:
        dumps, hf = load(a.tracer, a.cv)
        items = []   # (group, dialogue_idx, (t, label, feats))
        for fi, (f, d) in enumerate(dumps):
            for r in d["records"]:
                idx = int(r["dialogue_idx"])
                for it in rows(r, hf.get(str(idx))): items.append((group_of(a.cv, fi, idx), idx, it))
        groups = sorted({g for g, _, _ in items}); coefs = []
        for g in groups:
            m, z, C, ll, n = fit([(idx, it) for gg, idx, it in items if gg != g], feats)
            coefs.append(m.coef_[0]); keep = {idx for gg, idx, _ in items if gg == g}
            aucs = []
            for fi, (f, d) in enumerate(dumps):
                if a.cv == "comta" and fi != g: continue
                auc, k = predict_into(m, z, feats, d, hf, keep if a.cv != "comta" else None)
                if k: aucs.append((auc, k))
            print(f"{a.cv} group {g}: fit n={n} (C={C}, val log-loss {ll:.3f}); test AUC " + ", ".join(f"{x:.3f} (n={k})" for x, k in aucs))
        for fi, (f, d) in enumerate(dumps):
            d["model"] = tag; pickle.dump(d, open(f.replace(f"{a.tracer}_{a.cv}", f"{tag}_{a.cv}"), "wb"))
        co = np.array(coefs)
        print("standardised coefficients, mean ± sd over groups:")
        for k, mu, sd in zip(feats, co.mean(0), co.std(0)): print(f"   {k:14s} {mu:+.3f} ± {sd:.3f}")

    if a.fit:
        dumps, hf = load(a.tracer, a.fit)
        items = [(int(r["dialogue_idx"]), it) for _, d in dumps for r in d["records"] for it in rows(r, hf.get(str(int(r["dialogue_idx"]))))]
        m, z, C, ll, n = fit(items, feats)
        print(f"fit on all {a.fit} dialogues (n={n}, C={C}); transfer:")
        for ds in a.eval:
            dumps_e, hf_e = load(a.tracer, ds)
            for f, d in dumps_e:
                auc, k = predict_into(m, z, feats, d, hf_e, None)
                d["model"] = f"{tag}-from-{a.fit}"; pickle.dump(d, open(f.replace(f"{a.tracer}_{ds}", f"{tag}-from-{a.fit}_{ds}"), "wb"))
                print(f"   {ds} {os.path.basename(f)}: head AUC {auc:.3f} on {k} turns")


if __name__ == "__main__":
    main()
