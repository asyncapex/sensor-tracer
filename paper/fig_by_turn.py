"""Figure 3: next-turn AUC as a function of the predicted turn's index, per model and dataset.
    cd wp1/paper && ../.venv/bin/python fig_by_turn.py   -> figs/auc_by_turn.pdf
"""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "harness", "out")
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
from metrics import predictions, safe_auc  # noqa: E402

MODELS = [("cbkt", "Constrained BKT", "#3b6ea5", "--"), ("gauss", "Gaussian", "#7fc8be", "--"),
          ("dktsem", "DKT-Sem", "#8a8a8a", "-"), ("gauss+lmkt", "Gaussian + LLMKT judg.", "#2a9d8f", "-"),
          ("gauss+lmkt+mix", "+ mixture head", "#e07b00", "-."), ("lmkt", "LLMKT", "#c1121f", "-"), ("dkt", "DKT (labels only)", "#bdbdbd", ":")]
BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 99, "4+")]


def by_turn(tag, dataset):
    files = [f for f in sorted(glob.glob(os.path.join(OUT, f"{tag}_{dataset}_fold*.pkl"))) if "_partial" not in f]
    if not files:
        return None
    y, p, t = [], [], []
    for f in files:
        for r in pickle.load(open(f, "rb"))["records"]:
            for ti, lab, prob in predictions(r):
                y.append(lab); p.append(prob); t.append(ti)
    y, p, t = np.array(y), np.array(p), np.array(t)
    out = []
    for lo, hi, _ in BUCKETS:
        m = (t >= lo) & (t <= hi)
        out.append((safe_auc(y[m], p[m]) if m.sum() >= 20 else np.nan, int(m.sum())))
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, (dataset, title) in zip(axes, [("comta", "CoMTA (5 folds pooled)"), ("mathdial", "MathDial")]):
        for tag, name, color, ls in MODELS:
            res = by_turn(tag, dataset)
            if res is None:
                continue
            ax.plot(range(4), [a for a, _ in res], ls, color=color, marker="o", ms=3.5, lw=1.4, label=name)
        ax.set_xticks(range(4)); ax.set_xticklabels([b[2] for b in BUCKETS]); ax.set_xlabel("index of the predicted turn")
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3); ax.axhline(0.5, color="k", lw=0.5, alpha=0.4)
        counts = by_turn("cbkt", dataset)
        ax.set_xticklabels([f"{b[2]}\n(n={c})" for b, (_, c) in zip(BUCKETS, counts)], fontsize=7)
    axes[0].set_ylabel("AUC"); axes[0].set_ylim(0.45, 0.85)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs", "auc_by_turn.pdf")); fig.savefig(os.path.join(HERE, "figs", "auc_by_turn.png"), dpi=150)
    for dataset in ("comta", "mathdial"):
        print(dataset)
        for tag, name, _, _ in MODELS:
            res = by_turn(tag, dataset)
            if res: print(f"  {name:24s}", " ".join(f"{b[2]}:{a:.3f}" for b, (a, _) in zip(BUCKETS, res)))


if __name__ == "__main__":
    main()
