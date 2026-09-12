"""Mixture prediction head: p = w * p_LLMKT + (1-w) * p_tracer, with w chosen by cross-fitting over dialogues
(the tracer's state is untouched; only the *prediction* mixes in the sensor's own probability).
Also a turn-indexed variant: w_t = w0 for the first two predicted turns, w1 afterwards.
    ../.venv/bin/python mix_head.py comta gauss+lmkt lmkt
Reports cross-fitted AUC / ECE of tracer, LLMKT, mixture, and the paired bootstrap CI of (mixture - LLMKT).
"""
import sys, numpy as np
from bootstrap import per_dialogue
from metrics import safe_auc, ece

def collect(D):
    rows = []
    for d, v in D.items():
        for t, y, p in zip(v["t"], v["y"], v["p"]): rows.append((d, t, y, p))
    return rows

def main(dataset, tracer, sensor, B=1000):
    A, S = per_dialogue(tracer, dataset), per_dialogue(sensor, dataset)
    ra = {(d, t): (y, p) for d, t, y, p in collect(A)}; rs = {(d, t): (y, p) for d, t, y, p in collect(S)}
    keys = sorted(set(ra) & set(rs)); print(f"{dataset}: {len(keys)} predicted turns shared by {tracer} and {sensor}")
    d_idx = np.array([k[0] for k in keys]); t_idx = np.array([k[1] for k in keys])
    y = np.array([ra[k][0] for k in keys]); pa = np.array([ra[k][1] for k in keys]); ps = np.array([rs[k][1] for k in keys])
    groups = d_idx % 5; grid = np.linspace(0, 1, 21)
    def mix(w0, w1): w = np.where(t_idx <= 2, w0, w1); return w * ps + (1 - w) * pa
    pm_single = np.zeros_like(pa); pm_turn = np.zeros_like(pa); chosen = []
    for g in range(5):
        tr, te = groups != g, groups == g
        best = max(grid, key=lambda w: safe_auc(y[tr], (w * ps + (1 - w) * pa)[tr]))
        best2 = max(((w0, w1) for w0 in grid for w1 in grid), key=lambda ww: safe_auc(y[tr], mix(*ww)[tr]))
        pm_single[te] = (best * ps + (1 - best) * pa)[te]; pm_turn[te] = mix(*best2)[te]; chosen.append((best, best2))
    print("chosen weights per group (single w; (w_first2, w_later)):", [(round(a, 2), (round(b[0], 2), round(b[1], 2))) for a, b in chosen])
    for name, p in [(tracer, pa), (sensor, ps), ("mixture (one w)", pm_single), ("mixture (turn-indexed)", pm_turn)]:
        print(f"  {name:24s} AUC {safe_auc(y, p):.3f}  ECE {ece(y, p):.3f}  cold(t<=2) AUC {safe_auc(y[t_idx<=2], p[t_idx<=2]):.3f}  late AUC {safe_auc(y[t_idx>2], p[t_idx>2]):.3f}")
    rng = np.random.default_rng(0); uniq = np.unique(d_idx); idx_by = {u: np.where(d_idx == u)[0] for u in uniq}
    for name, p in [("mixture (one w)", pm_single), ("mixture (turn-indexed)", pm_turn), (tracer, pa)]:
        diffs = []
        for _ in range(B):
            pick = rng.choice(uniq, len(uniq), replace=True); ii = np.concatenate([idx_by[u] for u in pick])
            diffs.append(safe_auc(y[ii], p[ii]) - safe_auc(y[ii], ps[ii]))
        lo, hi = np.percentile(diffs, [2.5, 97.5]); print(f"  AUC({name}) - AUC({sensor}) = {safe_auc(y, p) - safe_auc(y, ps):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])


def write_dumps(dataset, tracer, sensor):
    """Write out/<tracer>+mix_<dataset>_fold*.pkl: the tracer's records with cross-fitted mixture probabilities in head_preds."""
    import glob, os, pickle
    A, S = per_dialogue(tracer, dataset), per_dialogue(sensor, dataset)
    ra = {(d, t): (y, p) for d, t, y, p in collect(A)}; rs = {(d, t): (y, p) for d, t, y, p in collect(S)}
    keys = sorted(set(ra) & set(rs)); d_idx = np.array([k[0] for k in keys]); y = np.array([ra[k][0] for k in keys])
    pa = np.array([ra[k][1] for k in keys]); ps = np.array([rs[k][1] for k in keys]); groups = d_idx % 5; grid = np.linspace(0, 1, 21)
    pm = np.zeros_like(pa); ws = []
    for g in range(5):
        tr, te = groups != g, groups == g
        w = max(grid, key=lambda w: safe_auc(y[tr], (w * ps + (1 - w) * pa)[tr])); pm[te] = (w * ps + (1 - w) * pa)[te]; ws.append(w)
    head = {k: float(p) for k, p in zip(keys, pm)}
    here = os.path.dirname(os.path.abspath(__file__)); n = 0
    for f in sorted(glob.glob(os.path.join(here, "out", f"{tracer}_{dataset}_fold*.pkl"))):
        if "_partial" in f or "+head" in f or "+mix" in f: continue
        d = pickle.load(open(f, "rb"))
        for r in d["records"]:
            idx = int(r["dialogue_idx"]); r["head_preds"] = {t: head[(idx, t)] for t in range(1, len(r["labels"])) if (idx, t) in head}; n += len(r["head_preds"])
        d["model"] = f"{tracer}+mix"; d["config"] = dict(d.get("config", {}), mixture_weights=ws, sensor=sensor)
        out = f.replace(f"{tracer}_{dataset}", f"{tracer}+mix_{dataset}"); pickle.dump(d, open(out, "wb"))
    print(f"{dataset}: wrote {tracer}+mix dumps with {n} head predictions; weights {ws}")
