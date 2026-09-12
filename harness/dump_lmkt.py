"""Dump LLMKT (fine-tuned Llama, LoRA) per-skill mastery queries in the harness format.

GPU required. Run from the dialogue-kt directory after `train --model_type lmkt`:
    python ../harness/dump_lmkt.py --model_name lmkt_comta --folds 1 2 3 4 5 [--quantize True]

For each test dialogue with labeled turns i = 0..L-1 (in dialogue order):
  * for i < L-1: prompt = dialogue up to the teacher's utterance of labeled turn i+1 (student reply hidden),
    queried for KC set kcs[i+1] U kcs[i]; states[i][k] = P("True" | k) = "state after observing turn i".
    states[i][kcs[i+1]] is exactly LLMKT's next-turn prediction; states[i][kcs[i]] gives the direction metrics.
  * for i = L-1: prompt = the full dialogue, queried for kcs[L-1].
  Unqueried entries are NaN. A text model has no label to flip, so states_cf is all NaN.
Resumable per fold.
"""
import argparse, os, pickle, sys, time
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from dialogue_kt.data_loading import load_annotated_data, load_kc_dict  # noqa: E402
from dialogue_kt.kt_data_loading import apply_annotations  # noqa: E402
from dialogue_kt.kt_data_loading import LMKTCollatorPacked  # noqa: E402
from dialogue_kt.models.lm import get_model  # noqa: E402
from dialogue_kt.prompting import get_true_false_tokens, kt_system_prompt, kt_user_prompt  # noqa: E402
from dialogue_kt.utils import device, get_checkpoint_path  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def make_args(dataset, model_name, base_model, quantize):
    return argparse.Namespace(dataset=dataset, model_type="lmkt", model_name=model_name, base_model=base_model, quantize=quantize,
                              agg="mean-ar", tag_src="atc", split_by_subject=False, typical_cutoff=1, inc_first_label=False,
                              testonval=False, debug=False, pack_kcs=True, prompt_inc_labels=False)


def build_prompt(tokenizer, sample, dialogue, turn_idx, kc_texts, args):
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": kt_system_prompt(args)},
        {"role": "user", "content": kt_user_prompt(sample, dialogue, turn_idx, None, args)},
    ], tokenize=False)
    conts = [tokenizer.apply_chat_template([{"role": "user", "content": kc}, {"role": "assistant", "content": "\n"}], tokenize=False) for kc in kc_texts]
    conts = [" " + c.split("user<|end_header_id|>\n\n")[1] for c in conts]
    return prompt + "".join(conts)


@torch.no_grad()
def query(model, collator, true_token, false_token, prompt, kc_texts):
    batch = collator([{"prompt": prompt, "kcs": kc_texts, "label": 0}])
    attention_mask = batch["attention_mask"]; min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype; attention_mask[attention_mask == 1] = 0
    out = model(input_ids=batch["input_ids"], attention_mask=attention_mask.type(model.dtype), position_ids=batch["position_ids"])
    logits = out.logits[0, batch["last_idxs"][0]]
    logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
    return torch.softmax(logits.float(), dim=1)[:, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comta")
    ap.add_argument("--model_name", default="lmkt_comta")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--quantize", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_args(a.dataset, a.model_name, a.base_model, a.quantize)
    kc_dict = load_kc_dict(args); K = len(kc_dict)
    for fold in a.folds:
        out = os.path.join(OUT_DIR, f"lmkt_{a.dataset}_fold{fold}.pkl")
        if os.path.exists(out):
            print(f"fold {fold}: exists, skipping"); continue
        t0 = time.time()
        ck = f"{a.model_name}_{fold}" if os.path.isdir(get_checkpoint_path(f"{a.model_name}_{fold}")) else a.model_name  # MathDial: single split, no fold suffix
        print(f"fold {fold}: loading adapters {ck}")
        model, tokenizer = get_model(a.base_model, True, model_name=ck, quantize=a.quantize)
        model.eval()
        collator = LMKTCollatorPacked(tokenizer)
        true_token, false_token = get_true_false_tokens(tokenizer)
        _, _, test_df = load_annotated_data(args, fold)
        records = []
        for idx, sample in tqdm(list(test_df.iterrows()), desc=f"fold {fold}"):
            dialogue = apply_annotations(sample)
            if not dialogue:
                continue
            turns = [t for t in dialogue if t["correct"] is not None and t["kcs"]]
            L = len(turns)
            if L < 2:
                continue
            labels = np.array([int(t["correct"]) for t in turns]); kcs = [[kc_dict[k] for k in t["kcs"]] for t in turns]
            states = np.full((L, K), np.nan); states_cf = np.full((L, K), np.nan)
            for i in range(L):
                if i < L - 1:
                    kc_texts = list(dict.fromkeys(turns[i + 1]["kcs"] + turns[i]["kcs"]))
                    prompt = build_prompt(tokenizer, sample, dialogue, turns[i + 1]["turn"], kc_texts, args)
                else:
                    kc_texts = list(dict.fromkeys(turns[i]["kcs"]))
                    prompt = build_prompt(tokenizer, sample, dialogue, None, kc_texts, args)
                probs = query(model, collator, true_token, false_token, prompt, kc_texts)
                for kc, p in zip(kc_texts, probs):
                    states[i][kc_dict[kc]] = p
            records.append({"dialogue_idx": idx, "labels": labels, "kcs": kcs, "states": states, "states_cf": states_cf})
        with open(out, "wb") as f:
            pickle.dump({"model": "lmkt", "dataset": a.dataset, "fold": fold, "num_kcs": K, "records": records, "base_model": a.base_model}, f)
        print(f"fold {fold}: {len(records)} dialogues -> {out} ({time.time()-t0:.0f}s)")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
