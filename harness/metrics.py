"""Robustness-first evaluation over dumped state trajectories (WP0 harness, v0).

    ../.venv/bin/python ../harness/metrics.py out/bkt_comta_fold*.pkl out/dktsem_comta_fold*.pkl

Per model: prediction quality (AUC, accuracy, Brier, ECE), early-sequence quality
(first two predicted turns vs later), Yeung & Yeung 2018 consistency (m1, m2) and waviness (w1, w2),
a wrong-direction update rate (Hooshyar et al. 2025), and the counterfactual monotonicity
violation rate (CMKT, Zhang et al. 2023). Reported per fold (mean +- std) and pooled.
"""
import glob, json, pickle, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score


def predictions(rec):
    """(turn index t>=1, label, predicted P(correct)) using mean-ar over the turn's KCs from states[t-1];
    if the record carries a learned prediction head ("head_preds": {t: p}), those probabilities are used instead
    (the state, and therefore the direction/counterfactual metrics, are unchanged)."""
    out = []
    hp = rec.get("head_preds")
    if hp is not None:
        return [(t, int(rec["labels"][t]), float(p)) for t, p in sorted(hp.items())]
    for t in range(1, len(rec["labels"])):
        kcs = rec["kcs"][t]
        if not kcs:
            continue
        vals = rec["states"][t - 1][kcs]
        if np.isnan(vals).all():
            continue
        out.append((t, int(rec["labels"][t]), float(np.nanmean(vals))))
    return out


def ece(labels, probs, bins=10):
    labels, probs = np.asarray(labels), np.asarray(probs)
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            total += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return total


def safe_auc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")


def fold_metrics(dump):
    recs = dump["records"]; K = dump["num_kcs"]
    y, p, t_idx = [], [], []
    for r in recs:
        for t, lab, prob in predictions(r):
            y.append(lab); p.append(prob); t_idx.append(t)
    y, p, t_idx = np.array(y), np.array(p), np.array(t_idx)
    m = {"n_pred": len(y), "auc": safe_auc(y, p), "acc": float(((p >= 0.5) == y).mean()),
         "brier": float(np.mean((p - y) ** 2)), "ece": float(ece(y, p))}
    early = t_idx <= 2
    m["auc_early(t<=2)"] = safe_auc(y[early], p[early]); m["err_early(t<=2)"] = float(((p[early] >= 0.5) != y[early]).mean())
    m["auc_late(t>2)"] = safe_auc(y[~early], p[~early]); m["err_late(t>2)"] = float(((p[~early] >= 0.5) != y[~early]).mean())
    # consistency / waviness / direction / counterfactual
    m1, m2, wrong, cf_viol, cf_margin, w1, w2 = [], [], [], [], [], [], []
    for r in recs:
        S, Scf, L = r["states"], r["states_cf"], len(r["labels"])
        full = not np.isnan(S).any()   # models that expose the whole state vector (not query-only LLMs)
        for t in range(1, L):
            if full:
                w1.append(np.abs(S[t] - S[t - 1]).sum() / K); w2.append(((S[t] - S[t - 1]) ** 2).sum() / K)
        for t in range(L):
            a = int(r["labels"][t]); sgn = 1 if a == 1 else -1
            for k in r["kcs"][t]:
                if t >= 1 and not (np.isnan(S[t][k]) or np.isnan(S[t - 1][k])):
                    d = S[t][k] - S[t - 1][k]
                    m1.append(sgn * np.sign(d)); m2.append(sgn * d); wrong.append(sgn * d < 0)
                # counterfactual: factual mastery must be >= counterfactual when a=1, <= when a=0
                if not (np.isnan(S[t][k]) or np.isnan(Scf[t][k])):
                    gap = sgn * (S[t][k] - Scf[t][k])
                    cf_viol.append(gap < 0); cf_margin.append(gap)
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    m.update({"m1": mean(m1), "m2": mean(m2), "w1": mean(w1), "w2": mean(w2), "wrong_direction_rate": mean(wrong),
              "cf_violation_rate": mean(cf_viol), "cf_mean_margin": mean(cf_margin)})
    return m, (y, p, t_idx)


def main(paths, subset_path=None):
    keep = None
    if subset_path:   # restrict every model to the dialogues present in this dump (fair partial comparisons)
        keep = {int(r["dialogue_idx"]) for r in pickle.load(open(subset_path, "rb"))["records"]}
        print(f"restricting to {len(keep)} dialogues from {subset_path}")
    by_model = defaultdict(list)
    for path in paths:
        dump = pickle.load(open(path, "rb"))
        if keep is not None:
            dump["records"] = [r for r in dump["records"] if int(r["dialogue_idx"]) in keep]
        by_model[f"{dump['model']}/{dump['dataset']}"].append(dump)
    summary = {}
    for name, dumps in by_model.items():
        per_fold, pooled = [], [[], [], []]
        for d in sorted(dumps, key=lambda d: d["fold"]):
            m, (y, p, t) = fold_metrics(d); per_fold.append(m)
            pooled[0].extend(y); pooled[1].extend(p); pooled[2].extend(t)
        keys = list(per_fold[0].keys())
        agg = {k: (float(np.nanmean([f[k] for f in per_fold])), float(np.nanstd([f[k] for f in per_fold]))) for k in keys}
        y, p = np.array(pooled[0]), np.array(pooled[1])
        agg["auc_pooled"] = (safe_auc(y, p), 0.0); agg["acc_pooled"] = (float(((p >= 0.5) == y).mean()), 0.0)
        summary[name] = {"folds": len(per_fold), "per_fold": per_fold, "agg": agg}
    # print table
    models = list(summary.keys())
    rows = ["n_pred", "auc", "auc_pooled", "acc", "brier", "ece", "auc_early(t<=2)", "err_early(t<=2)", "auc_late(t>2)", "err_late(t>2)",
            "m1", "m2", "w1", "w2", "wrong_direction_rate", "cf_violation_rate", "cf_mean_margin"]
    print("| metric | " + " | ".join(models) + " |"); print("|---|" + "---|" * len(models))
    for k in rows:
        cells = []
        for mname in models:
            mu, sd = summary[mname]["agg"][k]
            cells.append("–" if np.isnan(mu) else (f"{mu:.0f}" if k == "n_pred" else (f"{mu:.3f}" if sd == 0 else f"{mu:.3f} ± {sd:.3f}")))
        print(f"| {k} | " + " | ".join(cells) + " |")
    json.dump(summary, open("harness_summary.json", "w"), indent=1, default=float)
    print("\nsaved harness_summary.json")


if __name__ == "__main__":
    argv = sys.argv[1:]; subset = None
    if "--subset" in argv:
        i = argv.index("--subset"); subset = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    files = [f for a in argv for f in glob.glob(a)]
    main(files, subset)
