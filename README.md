# Sensor-tracer: LLM judgments as a noisy observation channel of a transparent knowledge tracer

Code, state dumps and table/figure generators for the paper *Good Predictor, Poor Tracker: Measuring the Learner-State
Dynamics of LLM Knowledge Tracers, and a Transparent Alternative* (under review). Anonymised for review.

What is here:

| path | contents |
|---|---|
| `harness/` | the evaluation protocol (`metrics.py`, `bootstrap.py`), the tracers (`tracers.py`: constrained BKT and Gaussian tracer, each with an optional LLM-judgment channel), and one `dump_*.py` script per model family that writes per-skill state trajectories in a common format |
| `harness/out/` | the state dumps behind every number in the paper (92 files, one per model x dataset x fold). Tables and figures regenerate from these alone, without any model run |
| `harness/priors/`, `harness/head/` | cached LLM outputs for the two negative ablations (text-informed prior; learned head over question features) |
| `paper/` | `gen_tables.py`, `gen_table1.py`, `fig_by_turn.py`, `fig_trajectory.py`: regenerate every table and figure from `harness/out/` |
| `dialogue-kt.patch` | our changes to the original authors' repository (resumable cross-validation with timing, float32 casts in the LLMKT loss, gradient checkpointing, an inference-tensor fix) |
| `docs/colab_llmkt.md` | step-by-step Colab guide for the one GPU job (fine-tuning LLMKT), with the fixes we needed |

Not included, and why: the datasets (obtain them from the original authors' repository, see below; CoMTA's licence forbids
redistribution), the API-call caches (they contain dialogue text), the LLMKT LoRA adapters (about 200 MB per fold; retrain
with the Colab guide, about 2 GPU-hours per dataset), and the Python virtual environment.

## 1. Setup (about 10 minutes)

```bash
git clone https://github.com/umass-ml4ed/dialogue-kt.git          # the original authors' code and annotated data
cd dialogue-kt && git checkout c61f335 && git apply ../dialogue-kt.patch && cd ..
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# pykt-toolkit ships a stray `from turtle import forward`; comment that line out:
sed -i 's/^from turtle import forward/# from turtle import forward/' .venv/lib/python3.12/site-packages/pykt/models/*.py 2>/dev/null
```

The annotated CoMTA and MathDial dialogues (`data/annotated/*.csv`) come with the `dialogue-kt` repository. All harness
commands below are run **from inside `dialogue-kt/`** with the environment active, because the scripts import the
authors' loaders.

API keys, when a step needs one, are read from environment variables (`OPENAI_API_KEY`, `HF_TOKEN`); nothing is
stored in this repository.

## 2. Reproduce the tables and figures from the dumps (about 1 minute, no GPU, no API)

```bash
cd paper
python gen_table1.py          # Table 1 (run from dialogue-kt/: cd ../dialogue-kt && python ../paper/gen_table1.py)
python gen_tables.py          # Tables 2, 3, 5 and the emission table; fails loudly if a dump is missing
python fig_by_turn.py         # Figure 4
python fig_trajectory.py      # Figure 3
cd ../harness
python metrics.py out/gauss+lmkt_comta_fold*.pkl          # any dump: AUC, ECE, direction, counterfactual, waviness
python bootstrap.py comta lmkt gauss+lmkt gauss cbkt dktsem   # paired bootstrap CIs (Section 6)
python mix_head.py mathdial gauss+lmkt lmkt               # the mixture head (Section 6)
```

## 3. Regenerate the dumps (optional)

Times are for one CPU core unless stated. Every script is resumable and skips finished folds.

```bash
cd dialogue-kt
python ../harness/dump_bkt.py --dataset comta --folds 5                 # pyBKT, unconstrained: ~2 min
python ../harness/dump_tracer.py --model cbkt  --dataset comta --folds 5   # constrained BKT: 15-40 s per fold
python ../harness/dump_tracer.py --model gauss --dataset comta --folds 5   # Gaussian tracer: same
python ../harness/dump_dktsem.py --dataset comta --model_type dkt-sem      # neural text baseline: minutes on CPU
python ../harness/dump_dktsem.py --dataset comta --model_type dkt
python ../harness/dump_api_llm.py --dataset comta --model gpt-4.1          # prompted arm: ~$1.5 (CoMTA), ~$16 (MathDial)
# LLMKT: train on a GPU with docs/colab_llmkt.md, then
python ../harness/dump_lmkt.py --dataset comta --folds 5
# two-channel tracers: the LLM dumps supply the judgment channel
python ../harness/dump_tracer.py --model gauss --dataset comta --folds 5 --judgments "../harness/out/lmkt_comta_fold*.pkl" --tag lmkt
python ../harness/dump_tracer.py --model cbkt  --dataset comta --folds 5 --judgments "../harness/out/lmkt_comta_fold*.pkl" --tag lmkt
python ../harness/dump_tracer.py --model cbkt  --dataset comta --folds 5 --judgments "../harness/out/lmkt_comta_fold*.pkl" --tag lmkt --jmodel bins8
python ../harness/mix_head.py comta gauss+lmkt lmkt                        # writes gauss+lmkt+mix dumps
```
Replace `comta --folds 5` by `mathdial --folds 1` for the fixed MathDial split.

## 4. Dump format

Each `out/<model>_<dataset>_fold<k>.pkl` is a dict with `model`, `dataset`, `fold`, `num_kcs` and `records`, one record
per test dialogue: `dialogue_idx`, `labels` (correctness per labelled turn), `kcs` (skill ids per turn), `states`
(turns x skills array of the model's state after each turn; NaN where a model exposes no value), `states_cf` (the same
after flipping each turn's label, for the counterfactual test; absent for LLM tracers), and optionally `head_preds`
(`{turn: probability}` when a prediction head is used; the tracking metrics still read `states`).

## 5. Licence

MIT (see `LICENSE`) for the code in `harness/` and `paper/`. The dumps are derived from the annotated datasets of
Scarlatos, Baker and Lan (LAK 2025); the underlying dialogues remain under their original licences.
