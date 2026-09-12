"""Figure 2: per-skill state trajectories of three tracers on one MathDial dialogue.

Selection rule (stated in the caption): among test dialogues with >= 6 labelled turns and a skill practised
>= 4 times, take the one whose LLMKT wrong-direction rate is closest to the dataset-level rate, i.e. a typical
dialogue rather than the worst case. Plots the two most-practised skills.
    cd wp1/paper && ../.venv/bin/python fig_trajectory.py   -> figs/trajectory.pdf
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "harness", "out")
PANELS = [("lmkt", "LLMKT (fine-tuned LLM as tracer)"), ("dktsem", "DKT-Sem (neural, reads text)"), ("gauss+lmkt", "Gaussian sensor-tracer (ours)")]
DATASET = "mathdial"


def load(tag):
    return {int(r["dialogue_idx"]): r for r in pickle.load(open(os.path.join(OUT, f"{tag}_{DATASET}_fold1.pkl"), "rb"))["records"]}


def wrong_rate(r):
    S = r["states"]; L = len(r["labels"]); n = w = 0
    for t in range(1, L):
        sgn = 1 if r["labels"][t] == 1 else -1
        for k in r["kcs"][t]:
            if not (np.isnan(S[t][k]) or np.isnan(S[t - 1][k])):
                n += 1; w += (sgn * (S[t][k] - S[t - 1][k]) < 0)
    return (w / n) if n else np.nan, n


def main():
    recs = {tag: load(tag) for tag, _ in PANELS}
    lm = recs["lmkt"]
    rates = [(idx, *wrong_rate(r)) for idx, r in lm.items()]
    overall = np.nansum([w * n for _, w, n in rates]) / np.nansum([n for _, _, n in rates])
    cands = []
    for idx, r in lm.items():
        L = len(r["labels"])
        if L < 6: continue
        counts = {}
        for kcs in r["kcs"]:
            for k in kcs: counts[k] = counts.get(k, 0) + 1
        if max(counts.values()) < 4: continue
        wr, n = wrong_rate(r)
        if n >= 8: cands.append((abs(wr - overall), idx, wr, counts))
    cands.sort()
    _, idx, wr, counts = cands[0]
    skills = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2]]
    print(f"dataset LLMKT wrong-direction rate {overall:.3f}; chosen dialogue {idx} (rate {wr:.3f}, {len(lm[idx]['labels'])} turns), skills {skills} practised {[counts[k] for k in skills]} times")
    r0 = lm[idx]; L = len(r0["labels"]); labels = r0["labels"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    colors = ["#c1121f", "#3b6ea5"]
    for ax, (tag, title) in zip(axes, PANELS):
        r = recs[tag][idx]; S = r["states"]
        for c, k in zip(colors, skills):
            xs = [t for t in range(L) if not np.isnan(S[t][k])]
            ax.plot([x + 1 for x in xs], [S[t][k] for t in xs], "-o", color=c, ms=3, lw=1.3, label=f"skill {k}")
            for t in range(L):
                if k in r["kcs"][t]:
                    ax.scatter([t + 1], [1.02 if labels[t] == 1 else -0.02], marker="^" if labels[t] == 1 else "v", color=c, s=18, clip_on=False)
        ax.set_title(title, fontsize=8); ax.set_ylim(-0.05, 1.05); ax.set_xlabel("labelled turn"); ax.grid(alpha=0.3)
        ax.set_xticks(range(1, L + 1))
    axes[0].set_ylabel("estimated P(correct on skill)"); axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.text(0.5, -0.02, "triangles: the turn practised the skill; up = answered correctly, down = incorrectly", ha="center", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figs", "trajectory.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "figs", "trajectory.png"), dpi=150, bbox_inches="tight")
    for tag, _ in PANELS:
        print(f"  {tag:12s} wrong-direction on this dialogue: {wrong_rate(recs[tag][idx])[0]:.2f}")


if __name__ == "__main__":
    main()
