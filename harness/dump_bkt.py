"""Fit (or load) pyBKT per fold exactly as dialogue-kt does, then dump full per-KC state
trajectories with an explicit BKT forward pass over the fitted parameters.

Run from the dialogue-kt directory:
    ../.venv/bin/python ../harness/dump_bkt.py --folds 1 2 3 4 5

Output format matches dump_dktsem.py (states / states_cf are L x K).
Resumable: fitted models are saved to saved_models/bkt_{dataset}_{fold}.pkl and reused;
an existing output file for a fold is skipped.
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from dialogue_kt.training import bkt_prep_data, load_annotated_data, load_kc_dict  # noqa: E402
from pyBKT.models import Model as BKT  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
UNSEEN = {"prior": 0.5, "learns": 0.0, "guesses": 0.5, "slips": 0.5}  # pyBKT predicts 0.5 for unseen skills


def make_args(dataset):
    return argparse.Namespace(dataset=dataset, model_type="bkt", model_name=None, agg="mean-ar", tag_src="atc",
                              split_by_subject=False, typical_cutoff=1, inc_first_label=False, testonval=False, debug=False)


def fitted_params(model, num_kcs):
    """Per-skill (prior, learn, guess, slip) arrays indexed by KC id."""
    P = model.params()  # index: (skill, param, class); column: value
    P = P.reset_index()
    P = P[P["class"] == "default"] if "class" in P.columns else P
    table = {k: dict(UNSEEN) for k in range(num_kcs)}
    for skill, grp in P.groupby("skill"):
        k = int(skill)
        table[k] = {row["param"]: float(row["value"]) for _, row in grp.iterrows()}
        table[k].setdefault("forgets", 0.0)
    prior = np.array([table[k]["prior"] for k in range(num_kcs)])
    learn = np.array([table[k]["learns"] for k in range(num_kcs)])
    guess = np.array([table[k]["guesses"] for k in range(num_kcs)])
    slip = np.array([table[k]["slips"] for k in range(num_kcs)])
    forget = np.array([table[k].get("forgets", 0.0) for k in range(num_kcs)])
    unseen = sum(1 for k in range(num_kcs) if table[k] == UNSEEN)
    # pyBKT can fit exactly 0 or 1; clip so the Bayes update never divides 0 by 0
    eps = 1e-6
    prior, learn, guess, slip = (np.clip(v, eps, 1 - eps) for v in (prior, learn, guess, slip))
    return prior, learn, guess, slip, forget, unseen


def bkt_update(state, k, a, learn, guess, slip, forget):
    """Posterior after observing answer a on skill k, then the learn/forget transition."""
    L = state[k]
    if a == 1:
        post = L * (1 - slip[k]) / (L * (1 - slip[k]) + (1 - L) * guess[k])
    else:
        post = L * slip[k] / (L * slip[k] + (1 - L) * (1 - guess[k]))
    state[k] = post * (1 - forget[k]) + (1 - post) * learn[k]


def trajectories(sample, prior, learn, guess, slip, forget):
    L = len(sample["labels"]); K = len(prior)
    states = np.zeros((L, K)); states_cf = np.zeros((L, K))
    s = prior.copy()
    for t in range(L):
        a = int(sample["labels"][t])
        s_cf = s.copy()
        for k in sample["kc_ids"][t]:
            bkt_update(s, k, a, learn, guess, slip, forget)
            bkt_update(s_cf, k, 1 - a, learn, guess, slip, forget)
        states[t] = s; states_cf[t] = s_cf
    return states, states_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comta")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_args(a.dataset)
    kc_dict = load_kc_dict(args); K = len(kc_dict)
    for fold in a.folds:
        out = os.path.join(OUT_DIR, f"bkt_{a.dataset}_fold{fold}.pkl")
        if os.path.exists(out):
            print(f"fold {fold}: {out} exists, skipping"); continue
        t0 = time.time()
        train_df, val_df, test_df = load_annotated_data(args, fold)
        train_flat, _ = bkt_prep_data(pd.concat([train_df, val_df]), kc_dict)
        ckpt = f"saved_models/bkt_{a.dataset}_{fold}.pkl"
        model = BKT(seed=221, num_fits=1)
        if os.path.exists(ckpt):
            model.load(ckpt); print(f"fold {fold}: loaded {ckpt}")
        else:
            model.fit(data=train_flat); model.save(ckpt)
            print(f"fold {fold}: fitted and saved {ckpt} ({time.time()-t0:.0f}s)")
        prior, learn, guess, slip, forget, unseen = fitted_params(model, K)
        _, test_ds = bkt_prep_data(test_df, kc_dict)
        records = []
        for sample in test_ds.data:
            states, states_cf = trajectories(sample, prior, learn, guess, slip, forget)
            records.append({"dialogue_idx": sample["dialogue_idx"], "labels": np.array(sample["labels"], dtype=int),
                            "kcs": [list(k) for k in sample["kc_ids"]], "states": states, "states_cf": states_cf})
        # sanity: our forward pass must agree with pyBKT's own one-step predictions
        test_flat, _ = bkt_prep_data(test_df, kc_dict)
        pred = model.predict(data=test_flat).sort_values("order_id")
        ours = []
        for r in records:
            s = prior.copy()
            for t in range(len(r["labels"])):
                for k in r["kcs"][t]:
                    ours.append(s[k] * (1 - slip[k]) + (1 - s[k]) * guess[k])
                    bkt_update(s, k, int(r["labels"][t]), learn, guess, slip, forget)
        theirs = pred["correct_predictions"].to_numpy()
        diff = np.nanmax(np.abs(np.array(ours) - theirs)); n_nan = int(np.isnan(theirs).sum())
        with open(out, "wb") as f:
            pickle.dump({"model": "bkt", "dataset": a.dataset, "fold": fold, "num_kcs": K, "records": records,
                         "params": {"prior": prior, "learn": learn, "guess": guess, "slip": slip, "forget": forget}}, f)
        print(f"fold {fold}: {len(records)} dialogues, unseen skills {unseen}/{K}, max |ours - pyBKT| = {diff:.2e} (pyBKT NaN preds: {n_nan}) -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
