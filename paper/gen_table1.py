"""Table 1: dataset statistics, computed from the annotated CSVs through the repository's own loader.
    cd wp1/dialogue-kt && ../.venv/bin/python ../paper/gen_table1.py   -> paper/tables/datasets.tex
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath("."))
from dialogue_kt.data_loading import load_annotated_data, load_kc_dict
from dialogue_kt.kt_data_loading import apply_annotations

HERE = os.path.dirname(os.path.abspath(__file__))


def make_args(dataset):
    return argparse.Namespace(dataset=dataset, tag_src="atc", split_by_subject=False, typical_cutoff=1)


def stats(df, kc_dict):
    n_dlg = n_turns = n_pred = n_correct = 0; kcs_per_turn = []; skills = set()
    for _, s in df.iterrows():
        d = apply_annotations(s)
        if not d: continue
        turns = [t for t in d if t["correct"] is not None and t["kcs"]]
        if len(turns) < 2: continue
        n_dlg += 1; n_turns += len(turns); n_pred += len(turns) - 1
        n_correct += sum(int(t["correct"]) for t in turns); kcs_per_turn += [len(t["kcs"]) for t in turns]
        skills |= {kc_dict[k] for t in turns for k in t["kcs"]}
    return dict(dialogues=n_dlg, turns=n_turns, predicted=n_pred, skills=len(skills), kpt=np.mean(kcs_per_turn), correct=n_correct / n_turns)


rows = []
for dataset, label in [("comta", "CoMTA"), ("mathdial", "MathDial")]:
    args = make_args(dataset); kc = load_kc_dict(args)
    tr, va, te = load_annotated_data(args, 1)
    if dataset == "comta":
        full = pd.concat([tr, va, te]); s = stats(full, kc); split = "5-fold CV over dialogues (65/15/20)"
        rows.append((label, "real students, LLM tutor", s, split))
    else:
        s_tr = stats(pd.concat([tr, va]), kc); s_te = stats(te, kc)
        rows.append((label + " train", "GPT-3.5 students, human tutors", s_tr, "fixed split"))
        rows.append((label + " test", "", s_te, ""))
lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\caption{Datasets after the original authors' filters (dialogues with at least two labelled turns; turns with at least one skill). Predicted turns exclude the first labelled turn of each dialogue.}", "\\label{tab:data}",
         "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{lrrrrrr}", "\\toprule", "Dataset & Dial. & Turns & Pred. & Skills & Sk/turn & \\% corr \\\\", "\\midrule"]
for label, _, s, _ in rows:
    lines.append(f"{label} & {s['dialogues']:,} & {s['turns']:,} & {s['predicted']:,} & {s['skills']} & {s['kpt']:.1f} & {100*s['correct']:.0f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
open(os.path.join(HERE, "tables", "datasets.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
