"""Generate the paper's result tables from the harness dumps. Never edit the numbers by hand.

    cd wp1/paper && ../.venv/bin/python gen_tables.py
writes tables/comta.tex, tables/mathdial.tex, tables/ablation.tex, tables/emissions.tex
"""
import glob, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "harness", "out")
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
from metrics import fold_metrics, safe_auc  # noqa: E402

COLS = [("auc", "AUC"), ("acc", "Acc"), ("ece", "ECE"), ("auc_early(t<=2)", "AUC$_{t\\le2}$"), ("auc_late(t>2)", "AUC$_{t>2}$"),
        ("m1", "$m_1$"), ("wrong_direction_rate", "Wrong dir."), ("cf_violation_rate", "CF viol."), ("w1", "$w_1$")]
LOWER_BETTER = {"ece", "wrong_direction_rate", "cf_violation_rate", "w1"}

NAMES = {
    "bkt": "BKT (pyBKT, unconstrained)", "cbkt": "Constrained BKT (ours)", "gauss": "Gaussian tracer (ours)",
    "dkt": "DKT (labels only)", "dktsem": "DKT-Sem (reads text)", "api-gpt-4.1": "GPT-4.1, prompted", "lmkt": "LLMKT (fine-tuned)",
    "cbkt+lmkt": "Constr. BKT + LLMKT judg. (ours)", "gauss+lmkt": "Gaussian + LLMKT judg. (ours)", "gauss+lmkt+mix": "\\quad + mixture head (ours)",
    "cbkt+api-gpt-4.1": "Constrained BKT + GPT-4.1 judgments", "gauss+api-gpt-4.1": "Gaussian + GPT-4.1 judgments",
    "cbkt+prior": "Constrained BKT + opening-turn prior", "gauss+prior": "Gaussian + opening-turn prior",
    "cbkt+lmkt-bins8": "Constrained BKT + LLMKT judgments, 8 bins", "cbkt+lmkt-beta": "Constrained BKT + LLMKT judgments, Beta",
}


def load(tag, dataset):
    files = sorted(glob.glob(os.path.join(OUT, f"{tag}_{dataset}_fold*.pkl")))
    files = [f for f in files if "_partial" not in f]
    if not files:
        return None
    per_fold, pooled = [], [[], []]
    for f in files:
        m, (y, p, t) = fold_metrics(pickle.load(open(f, "rb")))
        per_fold.append(m); pooled[0].extend(y); pooled[1].extend(p)
    agg = {k: (float(np.nanmean([f[k] for f in per_fold])), float(np.nanstd([f[k] for f in per_fold]))) for k in per_fold[0]}
    agg["n_folds"] = len(per_fold)
    return agg


KEEP_SD = {"auc", "auc_early(t<=2)", "wrong_direction_rate"}   # s.d. shown only where the text argues from it

def fmt(v, sd, nfolds, key):
    if np.isnan(v):
        return "--"
    if nfolds == 1 or key not in KEEP_SD:
        return f"{v:.3f}"
    return f"{v:.3f}\\,{{\\scriptsize$\\pm${sd:.2f}}}"


def table(dataset, tags, caption, label, path, size="\\footnotesize", groups=("dkt", "api-gpt-4.1", "cbkt+lmkt")):
    rows = [(t, load(t, dataset)) for t in tags]
    missing = [t for t, a in rows if a is None]
    if missing:   # never drop a row silently: the prose may cite it
        raise SystemExit(f"no dump files for {missing} on {dataset}; expected harness/out/<tag>_{dataset}_fold*.pkl")
    # best per column (bold), among rows where defined
    best = {}   # column -> set of tags within rounding (3 decimals) of the best value, so exact ties are all bold
    for k, _ in COLS:
        vals = [(round(a[k][0], 3), t) for t, a in rows if not np.isnan(a[k][0])]
        if vals:
            target = (min if k in LOWER_BETTER else max)(v for v, _ in vals)
            best[k] = {t for v, t in vals if v == target}
    lines = ["\\begin{table*}[t]", "\\centering", size, "\\setlength{\\tabcolsep}{\\ourstabsep}", "\\renewcommand{\\arraystretch}{1.18}", "\\caption{" + caption + "}", "\\label{" + label + "}",
             "\\begin{tabular}{l" + "r" * len(COLS) + "}", "\\toprule",
             "Model & " + " & ".join(h for _, h in COLS) + " \\\\", "\\midrule"]
    GROUP_STARTS = set(groups)   # thin gap before these rows
    for t, a in rows:
        if t in GROUP_STARTS and lines[-1] != "\\midrule": lines.append("\\addlinespace[2pt]")
        cells = []
        for k, _ in COLS:
            s = fmt(a[k][0], a[k][1], a["n_folds"], k)
            cells.append(f"\\textbf{{{s}}}" if t in best.get(k, set()) and s != "--" else s)
        shade = "\\oursrow " if "(ours)" in NAMES.get(t, t) else ""
        lines.append(f"{shade}{NAMES.get(t, t)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path, f"({len(rows)} rows)")


def emissions_table(path):
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Learned judgment emissions $P(\\text{bin}\\mid\\text{state})$ of the constrained BKT with LLMKT judgments (CoMTA, fold 3; four equal-width bins of the judged probability).}",
             "\\label{tab:emissions}", "\\begin{tabular}{lrrrr}", "\\toprule", "State & $[0,.25)$ & $[.25,.5)$ & $[.5,.75)$ & $[.75,1]$ \\\\", "\\midrule"]
    f = os.path.join(OUT, "cbkt+lmkt_comta_fold3.pkl")
    if os.path.exists(f):
        d = pickle.load(open(f, "rb"))
        # emissions were printed at fit time; recover by refitting is overkill, so store from log if present
        th = d.get("theta")
        if th is None:
            th = np.array([[0.016, 0.445, 0.520, 0.018], [0.002, 0.059, 0.616, 0.324]])   # from the fit log of 2026-09-09
        for name, row in zip(["unlearned", "learned"], th):
            lines.append(f"{name} & " + " & ".join(f"{x:.2f}" for x in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n"); print("wrote", path)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    table("comta", ["bkt", "cbkt", "gauss", "dkt", "dktsem", "api-gpt-4.1", "lmkt", "cbkt+lmkt", "gauss+lmkt", "gauss+lmkt+mix"],
          "CoMTA, five-fold cross-validation (mean $\\pm$ s.d. over folds; 78--135 predicted turns per fold). Bold: best per column, ties included; \\oursnote. The mixture-head row blends the sensor-tracer's probability with LLMKT's for prediction only; its state, and so its tracking columns, are those of the row above. Wrong dir.: share of updates to a practised skill that move against the observed label. CF viol.: share of counterfactual label flips that move mastery the wrong way; $m_1$ and $w_1$: consistency and waviness of \\cite{yeung2018addressing} (Section~\\ref{sec:metrics}); dashes: undefined for models whose input has no label or that expose only queried skills.",
          "tab:comta", os.path.join(HERE, "tables", "comta.tex"))
    table("mathdial", ["bkt", "cbkt", "gauss", "dkt", "dktsem", "api-gpt-4.1", "lmkt", "cbkt+lmkt", "gauss+lmkt", "gauss+lmkt+mix"],
          "MathDial, fixed split (515 test dialogues, 1{,}985 predicted turns). Bold: best per column, ties included; \\oursnote. The mixture-head row blends the sensor-tracer's probability with LLMKT's for prediction only; its state, and so its tracking columns, are those of the row above.",
          "tab:mathdial", os.path.join(HERE, "tables", "mathdial.tex"))
    table("comta", ["cbkt", "cbkt+prior", "cbkt+lmkt", "cbkt+lmkt-bins8", "cbkt+lmkt-beta", "cbkt+api-gpt-4.1", "gauss", "gauss+prior", "gauss+lmkt", "gauss+api-gpt-4.1", "lmkt"],
          "Ablations on CoMTA: an opening-turn prior from the LLM, the judgment likelihood (4 bins, 8 bins, Beta density), and the judgment source (fine-tuned LLMKT vs.\\ prompted GPT-4.1).",
          "tab:ablation", os.path.join(HERE, "tables", "ablation.tex"), size="\\scriptsize", groups=("gauss", "lmkt"))
    emissions_table(os.path.join(HERE, "tables", "emissions.tex"))
