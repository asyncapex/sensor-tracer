"""Dump full per-KC state trajectories from trained DKT-Sem fold checkpoints.

Run from the dialogue-kt directory:
    ../.venv/bin/python ../harness/dump_dktsem.py --folds 1 2 3 4 5

Output: ../harness/out/dktsem_comta_fold{f}.pkl, a list of dialogue records:
    dialogue_idx, labels (L), kcs (list of L lists of KC ids),
    states (L x K): mastery vector after observing turn t (used to predict turn t+1),
    states_cf (L x K): same, but with turn t's label flipped (counterfactual, CMKT-style).
Resumable: an existing output file for a fold is skipped.
"""
import argparse, os, pickle, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from dialogue_kt.training import get_baseline_model, compute_kc_emb_matrix, load_annotated_data, load_kc_dict, select_flat_baseline_out_vectors  # noqa: E402
from dialogue_kt.kt_data_loading import DKTDataset, DKTCollator  # noqa: E402
from dialogue_kt.utils import device, get_checkpoint_path  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def make_args(dataset, model_name, emb_size, model_type="dkt-sem"):
    return argparse.Namespace(
        dataset=dataset, model_type=model_type, model_name=model_name, emb_size=emb_size, agg="mean-ar",
        tag_src="atc", split_by_subject=False, typical_cutoff=1, inc_first_label=False, testonval=False,
        debug=False, batch_size=1, pack_kcs=True, base_model=None, quantize=False, prompt_inc_labels=False,
    )


def forward(model, batch, model_type):
    if model_type == "dkt":   # label-only DKT (pyKT) over flattened (skill, label) steps; one output per turn
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        return select_flat_baseline_out_vectors(y, batch, False)[0]
    return model(batch)[0]


def flipped_sample(sample, t):
    f = dict(sample)
    labels = list(sample["labels"]); labels[t] = 1 - labels[t]; f["labels"] = labels
    if "labels_flat" in sample:   # flip the pseudo-steps of turn t too
        lf = list(sample["labels_flat"]); ends = list(sample["turn_end_idxs"])
        start = ends[t - 1] + 1 if t > 0 else 0
        for j in range(start, ends[t] + 1): lf[j] = 1 - lf[j]
        f["labels_flat"] = lf
    return f


@torch.no_grad()
def run_dialogue(model, collator, sample, model_type="dkt-sem"):
    """Return (states L x K, states_cf L x K) for one dialogue sample."""
    batch = collator([sample])
    y = forward(model, batch, model_type)  # L x K
    L = y.shape[0]
    states = y.cpu().numpy()
    states_cf = np.zeros_like(states)
    for t in range(L):
        b = collator([flipped_sample(sample, t)])
        states_cf[t] = forward(model, b, model_type)[t].cpu().numpy()
    return states, states_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comta")
    ap.add_argument("--model_name", default="dkt-sem_comta")
    ap.add_argument("--emb_size", type=int, default=256)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model_type", default="dkt-sem", choices=["dkt-sem", "dkt"])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_args(a.dataset, a.model_name, a.emb_size, a.model_type)
    kc_dict = load_kc_dict(args)
    if a.model_type == "dkt-sem":
        sbert = SentenceTransformer("all-mpnet-base-v2"); kc_emb_matrix = compute_kc_emb_matrix(sbert, kc_dict)
    else:
        sbert, kc_emb_matrix = None, None
    collator = DKTCollator(flatten_kcs=(a.model_type == "dkt"))
    tag = "dktsem" if a.model_type == "dkt-sem" else "dkt"
    for fold in a.folds:
        out = os.path.join(OUT_DIR, f"{tag}_{a.dataset}_fold{fold}.pkl")
        if os.path.exists(out):
            print(f"fold {fold}: {out} exists, skipping"); continue
        ckpt = get_checkpoint_path(f"{a.model_name}_{fold}.pt")
        if not os.path.exists(ckpt):
            ckpt = get_checkpoint_path(f"{a.model_name}.pt")   # MathDial: single split, no fold suffix
        if not os.path.exists(ckpt):
            print(f"fold {fold}: checkpoint {ckpt} missing, skipping"); continue
        t0 = time.time()
        model = get_baseline_model(kc_dict, kc_emb_matrix, args)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        _, _, test_df = load_annotated_data(args, fold)
        ds = DKTDataset(test_df, kc_dict, kc_emb_matrix, sbert)
        records = []
        for sample in ds.data:
            states, states_cf = run_dialogue(model, collator, sample, a.model_type)
            records.append({
                "dialogue_idx": sample["dialogue_idx"],
                "labels": np.array(sample["labels"], dtype=int),
                "kcs": [list(k) for k in sample["kc_ids"]],
                "states": states, "states_cf": states_cf,
            })
        with open(out, "wb") as f:
            pickle.dump({"model": tag, "dataset": a.dataset, "fold": fold, "num_kcs": len(kc_dict), "records": records}, f)
        print(f"fold {fold}: {len(records)} dialogues, {sum(len(r['labels']) for r in records)} turns -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
