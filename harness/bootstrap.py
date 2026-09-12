"""Paired bootstrap over dialogues for differences between two models (pooled across folds).

    ../.venv/bin/python bootstrap.py comta gauss+lmkt lmkt dkt-sem cbkt   # first tag is the reference
Reports 95% CIs for reference-minus-other on AUC, accuracy, ECE, cold-start AUC and wrong-direction rate.
"""
import glob, os, pickle, sys
import numpy as np
from metrics import predictions, safe_auc, ece

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")


def per_dialogue(tag, dataset):
    """dialogue_idx -> dict(y, p, t, dir: list of +1/-1 direction outcomes)"""
    D = {}
    for f in sorted(glob.glob(os.path.join(OUT, f"{tag}_{dataset}_fold*.pkl"))):
        if "_partial" in f: continue
        for r in pickle.load(open(f, "rb"))["records"]:
            pr = predictions(r)
            t, y, p = zip(*pr) if pr else ([], [], [])   # predictions() yields (turn index, label, probability)
            S, L = r["states"], len(r["labels"]); wrong = []
            for i in range(1, L):
                sgn = 1 if r["labels"][i] == 1 else -1
                for k in r["kcs"][i]:
                    if not (np.isnan(S[i][k]) or np.isnan(S[i - 1][k])):
                        wrong.append(int(sgn * (S[i][k] - S[i - 1][k]) < 0))
            D[int(r["dialogue_idx"])] = dict(y=np.array(y), p=np.array(p), t=np.array(t), wrong=np.array(wrong))
    return D


def stats(D, idxs):
    y = np.concatenate([D[i]["y"] for i in idxs]); p = np.concatenate([D[i]["p"] for i in idxs]); t = np.concatenate([D[i]["t"] for i in idxs])
    w = np.concatenate([D[i]["wrong"] for i in idxs])
    early = t <= 2
    return dict(auc=safe_auc(y, p), acc=float(((p >= 0.5) == y).mean()), ece=float(ece(y, p)),
                auc_early=safe_auc(y[early], p[early]), wrong=float(w.mean()) if len(w) else np.nan)


def main(dataset, ref, others, B=1000, seed=0):
    Dref = per_dialogue(ref, dataset); rng = np.random.default_rng(seed)
    for other in others:
        Do = per_dialogue(other, dataset)
        common = sorted(set(Dref) & set(Do)); n = len(common)
        base = {k: stats(Dref, common)[k] - stats(Do, common)[k] for k in ("auc", "acc", "ece", "auc_early", "wrong")}
        samples = {k: [] for k in base}
        for _ in range(B):
            idxs = [common[j] for j in rng.integers(0, n, n)]
            a, b = stats(Dref, idxs), stats(Do, idxs)
            for k in base: samples[k].append(a[k] - b[k])
        print(f"{dataset}: {ref} minus {other}  (n = {n} dialogues, {B} resamples)")
        for k in base:
            lo, hi = np.nanpercentile(samples[k], [2.5, 97.5])
            print(f"   {k:10s} {base[k]:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {'*' if lo > 0 or hi < 0 else ''}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
