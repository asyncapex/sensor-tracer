# Colab guide: LLMKT on a GPU

Purpose: train the paper's LLM tracer (LLMKT, Llama-3.1-8B-Instruct + LoRA) on CoMTA and MathDial, then dump its per-skill mastery queries so the local harness can score it next to our tracers. Everything below runs in a Colab Pro notebook, in order. Each cell is one step; the run is resumable at fold level, and checkpoints and results live on Drive so a disconnect costs at most one fold.

**Runtime to pick.** Runtime, Change runtime type, GPU: **A100** (bf16, no quantization, fastest) or **L4** (24 GB; also bf16). On a T4 (16 GB) add `--quantize True` to every training and dump command and expect roughly three times the time.

**Expected times** (paper setup: 5 epochs, batch 1 with 64-step accumulation, LoRA rank 16):

| Run | A100 | L4 | T4 (quantized) |
|---|---|---|---|
| CoMTA, 5 folds, training | ~1.5 h total | ~3 h | ~8 h |
| CoMTA, 5 folds, harness dump | ~15 min | ~30 min | ~1 h |
| MathDial, one split, training | ~4 to 6 h | ~10 h | not advisable |
| MathDial dump | ~1 h | ~2 h | – |

These are estimates from the paper's compute description and the data sizes; the first fold tells you the real number, and the loop prints an ETA after every fold.

**Before you start (one-time).**
1. The commands below use `unsloth/Meta-Llama-3.1-8B-Instruct`, an ungated mirror of the identical weights and tokenizer, so no license approval is needed. If you prefer Meta's official repo, request access at huggingface.co/meta-llama/Llama-3.1-8B-Instruct and drop the `--base_model` argument once approved.
2. Create a Hugging Face access token (read) at huggingface.co/settings/tokens.
3. In Colab, open the key icon (Secrets) on the left and add `HF_TOKEN` with that token. Enable "notebook access".
4. Upload `the bundle (zip of `harness/` and `dialogue-kt.patch`)` (in `wp1/` next to this vault) to your Drive at `MyDrive/wp1/`. It contains our patches to the repository and the `harness/` folder.

---

## Cell 1 · Mount Drive and check the GPU

```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os
WORK = '/content/drive/MyDrive/wp1'
os.makedirs(WORK, exist_ok=True)
```

## Cell 2 · Clone the repository, apply our patches, link results to Drive

The patch adds fold-level resumability and timing to `crossval()`, the `.clone()` fix for newer sentence-transformers, and float32 casts in the LLMKT loss (newer torch refuses a bfloat16/float32 mix in BCELoss). `saved_models/` and `results/` become symlinks into Drive so they survive a disconnect.

```python
%cd /content
!git clone --depth 1 https://github.com/umass-ml4ed/dialogue-kt.git
%cd /content/dialogue-kt
!unzip -o -q {WORK}/the bundle (zip of `harness/` and `dialogue-kt.patch`) -d /content/bundle
!git apply /content/bundle/dialogue-kt.patch && echo "patch applied"
!cp -r /content/bundle/harness /content/harness
for d in ('saved_models', 'results'):
    !rm -rf {d}; mkdir -p {WORK}/{d}; ln -s {WORK}/{d} {d}
!ls -la
```

## Cell 2b · Only if your session was built from a bundle older than 9 Sep 2026, 11:00

Newer torch refuses a bfloat16/float32 mix in BCELoss. The current bundle's patch already contains this fix; on an older session apply it by hand and check that four lines are listed:

```python
%cd /content/dialogue-kt
!sed -i 's/torch.softmax(logits, dim=2)/torch.softmax(logits.float(), dim=2)/; s/torch.softmax(logits, dim=1)/torch.softmax(logits.float(), dim=1)/; s/loss = torch.nn.BCELoss()(corr_probs, batch\["labels"\])/loss = torch.nn.BCELoss()(corr_probs.float(), batch["labels"])/' dialogue_kt/training.py
!grep -n "softmax(logits.float()\|corr_probs.float()" dialogue_kt/training.py
```

## Cell 2c · Only if your session was built from a bundle older than 9 Sep 2026, 12:00

The repository enables gradient checkpointing only when quantizing; in bf16 a long packed dialogue runs a 40 GB A100 out of memory at the end of the first epoch. The current bundle's patch fixes this; on an older session apply it by hand:

```python
%cd /content/dialogue-kt
import pathlib
p = pathlib.Path('dialogue_kt/models/lm.py'); s = p.read_text()
old = "            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)\n"
new = old + ("        elif use_gradient_checkpointing:\n"
             "            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})\n"
             "            model.enable_input_require_grads()\n")
assert old in s and 'enable_input_require_grads' not in s
p.write_text(s.replace(old, new)); print('gradient checkpointing enabled for bf16')
import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## Cell 3 · Install dependencies

Colab's preinstalled torch and CUDA are fine. The repository's pins are from 2024; install the current versions plus bitsandbytes and PEFT.

```python
!pip uninstall -y -q torchao   # Colab preinstalls an old torchao that current PEFT rejects; nothing here needs it
!pip install -q "transformers>=4.44" "peft>=0.12" bitsandbytes accelerate sentence-transformers pykt-toolkit pyBKT==1.4.1 "scikit-learn==1.5.2" krippendorff tqdm   # pyBKT is imported by the repo at load time; the scikit-learn pin is for pyBKT (the umap/hdbscan warning is harmless)
# pykt imports turtle (tkinter) by mistake; neutralize the line
import pykt, pathlib, re
q = pathlib.Path(pykt.__file__).parent / 'models' / 'qdkt.py'
q.write_text(re.sub(r'^from turtle import forward$', '# removed turtle import', q.read_text(), flags=re.M))
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   # less fragmentation on long sequences
```

## Cell 4 · Log in to Hugging Face

```python
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get('HF_TOKEN'))
```

## Cell 5 · Smoke test (about 5 minutes)

One fold, one epoch, ten dialogues. Confirms the model downloads (about 16 GB, cached in the session), the attention-mask packing works, and the GPU has room. Delete its outputs afterwards so they are not mistaken for real folds.

```python
!python -m dialogue_kt.main train --dataset comta --model_type lmkt --model_name smoke_lmkt --epochs 1 --debug --base_model unsloth/Meta-Llama-3.1-8B-Instruct
!rm -rf saved_models/smoke_lmkt* results/*smoke_lmkt*
```

If you are on a T4 and this fails with an out-of-memory error, append `--quantize True` here and to every command below.

## Cell 6 · Train LLMKT on CoMTA, 5-fold (the long one)

Resumable: rerunning the same cell after a disconnect skips finished folds.

```python
!python -m dialogue_kt.main train --dataset comta --crossval --model_type lmkt --model_name lmkt_comta --base_model unsloth/Meta-Llama-3.1-8B-Instruct
!cat results/metrics_crossval_lmkt_comta.txt
```

Target from the paper: accuracy 58.0, AUC 65.8, F1 60.7 (their Table 2, CoMTA).

## Cell 7 · Dump LLMKT's mastery queries for the harness

Queries each test turn for the union of its own skills and the previous turn's skills, so the harness can compute prediction and direction metrics. Output goes to `harness/out/lmkt_comta_fold{1..5}.pkl` and is copied to Drive.

```python
%cd /content/dialogue-kt
!python /content/harness/dump_lmkt.py --model_name lmkt_comta --folds 1 2 3 4 5 --base_model unsloth/Meta-Llama-3.1-8B-Instruct
!mkdir -p {WORK}/harness_out && cp /content/harness/out/lmkt_comta_fold*.pkl {WORK}/harness_out/
!ls -la {WORK}/harness_out
```

## Cell 8 · MathDial (second Colab session)

MathDial uses one fixed train/test split (2,253 train dialogues after the loader's "typical" filter, 595 test), so this is a single long training run rather than five folds. The repository saves it without a fold suffix (`lmkt_mathdial`), which the dump script now handles. Run it in a **fresh session** with an **A100**: Cells 1 to 4 (the current bundle already contains every fix, so Cells 2b and 2c are not needed), skip Cell 5, then this cell.

Expected time on the A100: about 5 epochs × 45 min ≈ 4 h of training (the CoMTA rate of 1.5 dialogues/s, 11,000 training turns), plus about 10 min of testing and 15 min for the dump, roughly 50 compute units. The loop prints train and validation loss per epoch so you can see progress; there is no per-epoch checkpoint, so a disconnect restarts the run.

```python
%cd /content/dialogue-kt
!python -m dialogue_kt.main train --dataset mathdial --model_type lmkt --model_name lmkt_mathdial --base_model unsloth/Meta-Llama-3.1-8B-Instruct
!cat results/metrics_lmkt_mathdial.txt
```

Target from the paper (MathDial, Table 2): accuracy 68.4, AUC 76.7, F1 62.2.

**Early stopping by hand.** With 8,400 training turns the model overfits after the first epoch (validation loss rises from epoch 2 on). The loop keeps the best-validation adapter, so once validation loss has risen for two consecutive epochs you can interrupt the cell (the stop button) and evaluate the saved adapter directly, saving a couple of hours:

```python
!python -m dialogue_kt.main test --dataset mathdial --model_type lmkt --model_name lmkt_mathdial --base_model unsloth/Meta-Llama-3.1-8B-Instruct
!cat results/metrics_lmkt_mathdial.txt
```

Then the harness dump (the dump script's `--folds 1` names the single split "fold 1"; it should report 595 test dialogues minus the few with fewer than two labeled turns):

```python
!python /content/harness/dump_lmkt.py --dataset mathdial --model_name lmkt_mathdial --folds 1 --base_model unsloth/Meta-Llama-3.1-8B-Instruct
!mkdir -p {WORK}/harness_out && cp /content/harness/out/lmkt_mathdial_fold1.pkl {WORK}/harness_out/
!ls -la {WORK}/harness_out
```

Download `lmkt_mathdial_fold1.pkl` into `wp1/harness/out/` as before, then disconnect the runtime.

## Cell 9 · Bring the results home

Download the `harness_out` folder from Drive (or sync it) into `wp1/harness/out/` on your machine, then run locally:

```bash
cd wp1/harness
../.venv/bin/python metrics.py "out/*_comta_fold*.pkl"
```

That produces the five-model table: pyBKT, constrained BKT, Gaussian tracer, DKT-Sem, LLMKT (plus the API-prompted arm if you ran it).

---

## If something goes wrong

- **`OSError: You are trying to access a gated repo`**: the license was not accepted with the same account as the token, or the token lacks read scope.
- **CUDA out of memory** during training: first make sure Cell 2c's gradient-checkpointing fix is in (it is in the current bundle); then restart the runtime so nothing else holds GPU memory; as a last resort use `--quantize True`.
- **Session disconnected mid-fold**: rerun Cell 2 (fast), Cells 3 and 4, then Cell 6. Finished folds are skipped; the interrupted fold restarts from scratch (there is no epoch-level checkpoint).
- **Different numbers from the paper by a few points**: expected; folds hold 78 to 135 predicted turns.
