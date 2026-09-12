"""Fit our transparent tracers per fold (tune on val, refit on train+val), dump trajectories.

Run from the dialogue-kt directory:
    ../.venv/bin/python ../harness/dump_tracer.py --model cbkt --folds 1 2 3 4 5
    ../.venv/bin/python ../harness/dump_tracer.py --model gauss --folds 1 2 3 4 5
Resumable: existing per-fold outputs are skipped.
"""
import argparse, glob, itertools, json, os, pickle, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.training import load_annotated_data, load_kc_dict  # noqa: E402
from dialogue_kt.kt_data_loading import DKTDataset  # noqa: E402
from tracers import ConstrainedBKT, GaussianTracer, predict_sequence  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

GRIDS = {
    "cbkt": [dict(max_guess=mg, max_slip=ms, shrink=sh) for mg, ms, sh in itertools.product([0.3, 0.5], [0.3, 0.5], [0.0, 5.0, 20.0, 100.0])],
    "gauss": [dict(beta=b, drift=dr, difficulty=d) for b, dr, d in itertools.product([0.5, 1.0, 2.0], [0.0, 0.2, 0.5], [-0.5, 0.0, 0.5])],
}


def make_args(dataset):
    return argparse.Namespace(dataset=dataset, model_type="bkt", model_name=None, agg="mean-ar", tag_src="atc",
                              split_by_subject=False, typical_cutoff=1, inc_first_label=False, testonval=False, debug=False)


def to_sequences(df, kc_dict):
    ds = DKTDataset(df, kc_dict, None, None)
    return [([(list(k), int(a)) for k, a in zip(s["kc_ids"], s["labels"])], s["dialogue_idx"]) for s in ds.data]


def build(model_name, K, cfg):
    return ConstrainedBKT(K, **cfg) if model_name == "cbkt" else GaussianTracer(K, **cfg)


def load_judgments(pattern):
    """Merge LLM dumps (one per fold) into {dialogue_idx: {i: {kc: y}}}; every dialogue appears in exactly one fold's test set."""
    J = {}
    for path in glob.glob(pattern):
        for r in pickle.load(open(path, "rb"))["records"]:
            per_turn = {}
            for i in range(len(r["labels"])):
                row = r["states"][i]; ks = np.where(~np.isnan(row))[0]
                if len(ks): per_turn[i] = {int(k): float(row[k]) for k in ks}
            J[int(r["dialogue_idx"])] = per_turn
    return J


def score(model, seqs, priors=None, J=None):
    y, p = [], []
    for seq, idx in seqs:
        for lab, prob in predict_sequence(model, seq, priors.get(str(idx)) if priors else None, J.get(int(idx)) if J else None):
            y.append(lab); p.append(prob)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return log_loss(y, p), (roc_auc_score(y, p) if 0 < sum(y) < len(y) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cbkt", "gauss"], required=True)
    ap.add_argument("--dataset", default="comta")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--prior", default=None, help="priors json from priors_api.py; enables the text-informed prior arm")
    ap.add_argument("--judgments", default=None, help="glob of LLM dump files whose per-skill judgments form the second observation channel")
    ap.add_argument("--tag", default=None, help="model tag suffix for the output files (default derived from --judgments)")
    ap.add_argument("--jmodel", default="bins4", choices=["bins4", "bins8", "beta"], help="cbkt judgment likelihood (ablation)")
    a = ap.parse_args()
    priors = None; tag = a.model
    J = None
    if a.judgments:
        J = load_judgments(a.judgments)
        tag = a.model + "+" + (a.tag or os.path.basename(a.judgments).split("_")[0]) + ("" if a.jmodel == "bins4" else "-" + a.jmodel)
        print(f"judgments loaded for {len(J)} dialogues from {a.judgments}")
    if a.prior:
        raw = json.load(open(a.prior)); priors = {d: {int(k): v for k, v in kcs.items()} for d, kcs in raw.items()}
        tag = a.model + "+prior"
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_args(a.dataset); kc_dict = load_kc_dict(args); K = len(kc_dict)
    for fold in a.folds:
        out = os.path.join(OUT_DIR, f"{tag}_{a.dataset}_fold{fold}.pkl")
        if os.path.exists(out):
            print(f"fold {fold}: exists, skipping"); continue
        t0 = time.time()
        train_df, val_df, test_df = load_annotated_data(args, fold)
        tr, va, te = (to_sequences(d, kc_dict) for d in (train_df, val_df, test_df))
        # tune on validation by log-loss
        best = None
        grid = GRIDS[a.model] if not priors else [dict(c, prior_weight=w) for c in GRIDS[a.model] for w in (0.5, 1.0)]
        if J is not None:
            jm = {"bins4": dict(n_bins=4), "bins8": dict(n_bins=8), "beta": dict(judgment_model="beta")}[a.jmodel]
            grid = [dict(c, judgment_weight=w, **jm) for c in grid for w in (0.5, 1.0)] if a.model == "cbkt" else \
                   [dict(c, beta_j=bj, judgment_weight=w) for c in grid for bj in (0.5, 1.0, 2.0) for w in (0.5, 1.0)]
        jtr = [J.get(int(i)) for _, i in tr] if J else None
        for cfg in grid:
            m = build(a.model, K, cfg).fit([s for s, _ in tr], jtr)
            ll, auc = score(m, va, priors, J)
            if best is None or ll < best[0]:
                best = (ll, auc, cfg)
        ll, auc, cfg = best
        print(f"fold {fold}: best cfg {cfg} (val log-loss {ll:.3f}, val AUC {auc:.3f})")
        model = build(a.model, K, cfg).fit([s for s, _ in tr + va], [J.get(int(i)) for _, i in tr + va] if J else None)
        if a.model == "cbkt" and getattr(model, "theta", None) is not None:
            print("          judgment emissions P(bin | unlearned) =", np.round(model.theta[0], 3), " P(bin | learned) =", np.round(model.theta[1], 3))
        records = []
        for seq, idx in te:
            states, states_cf = model.trajectories(seq, priors.get(str(idx)) if priors else None, J.get(int(idx)) if J else None)
            records.append({"dialogue_idx": idx, "labels": np.array([x[1] for x in seq], dtype=int),
                            "kcs": [x[0] for x in seq], "states": states, "states_cf": states_cf})
        extra = {}
        if a.model == "cbkt":
            P = model.params
            extra = {"global_params": model.global_params, "seen": int(model.seen.sum()),
                     "guess>0.5": int((P[:, 2] > 0.5).sum()), "slip>0.5": int((P[:, 3] > 0.5).sum()), "g+s>1": int(((P[:, 2] + P[:, 3]) > 1).sum())}
            print(f"          global (p0, T, g, s) = {tuple(round(x, 3) for x in model.global_params)}; degenerate g+s>1: {extra['g+s>1']}")
        with open(out, "wb") as f:
            pickle.dump({"model": tag, "dataset": a.dataset, "fold": fold, "num_kcs": K, "records": records, "config": cfg, **extra}, f)
        print(f"fold {fold}: {len(records)} test dialogues -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
