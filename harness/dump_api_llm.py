"""LLM-standalone arm without a GPU: prompt an OpenAI chat model with the repository's own KT prompt
(system + dialogue up to the teacher's question + one knowledge component), read P("True") from the
first-token logprobs. Same query pattern as dump_lmkt.py, so the harness treats both identically.

Run from the dialogue-kt directory:
    OPENAI_API_KEY=... ../.venv/bin/python ../harness/dump_api_llm.py --model gpt-4.1-mini --folds 1 2 3 4 5
Every (prompt, model) response is cached in harness/cache/, so reruns are free and the run is resumable
mid-fold. Prints token usage and an estimated cost per fold.
"""
import argparse, hashlib, json, math, os, pickle, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from dialogue_kt.data_loading import load_annotated_data, load_kc_dict  # noqa: E402
from dialogue_kt.kt_data_loading import apply_annotations  # noqa: E402
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out"); CACHE_DIR = os.path.join(HERE, "cache")
PRICE_PER_M = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}  # USD per 1M (input, output); verify against current pricing


def make_args(dataset):
    return argparse.Namespace(dataset=dataset, model_type="lmkt", model_name=None, agg="mean-ar", tag_src="atc", split_by_subject=False,
                              typical_cutoff=1, inc_first_label=False, testonval=False, debug=False, pack_kcs=False, prompt_inc_labels=False)


class Cache:
    def __init__(self, model):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, f"api_{model}.jsonl"); self.d = {}
        self.lock = threading.Lock()
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line); self.d[r["key"]] = r
    def get(self, key): return self.d.get(key)
    def put(self, key, rec):
        rec["key"] = key
        with self.lock:
            self.d[key] = rec
            with open(self.path, "a") as f: f.write(json.dumps(rec) + "\n")


class NotCached(Exception):
    pass


def p_true(client, model, system, user, cache, cached_only=False):
    key = hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()
    hit = cache.get(key)
    if hit: return hit["p_true"], 0, 0
    if cached_only: raise NotCached()
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                               max_tokens=1, temperature=0, logprobs=True, top_logprobs=10)
            break
        except Exception as e:
            if attempt == 4: raise
            time.sleep(2 ** attempt)
    top = r.choices[0].logprobs.content[0].top_logprobs
    lp = {}
    for t in top:
        tok = t.token.strip().lower()
        if tok in ("true", "false") and tok not in lp: lp[tok] = t.logprob
    if "true" in lp and "false" in lp:
        p = 1 / (1 + math.exp(lp["false"] - lp["true"]))
    elif "true" in lp: p = math.exp(lp["true"])
    elif "false" in lp: p = 1 - math.exp(lp["false"])
    else: p = 0.5
    usage = r.usage
    cache.put(key, {"p_true": p, "top": {t.token: t.logprob for t in top}, "in": usage.prompt_tokens, "out": usage.completion_tokens})
    return p, usage.prompt_tokens, usage.completion_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comta")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=None, help="only the first N test dialogues per fold (smoke test)")
    ap.add_argument("--workers", type=int, default=6, help="concurrent API calls per turn")
    ap.add_argument("--cached_only", action="store_true", help="make no API calls: keep only dialogues whose prompts are all cached (partial baseline)")
    a = ap.parse_args()
    from openai import OpenAI
    client = None if a.cached_only else OpenAI()
    os.makedirs(OUT_DIR, exist_ok=True)
    args = make_args(a.dataset); kc_dict = load_kc_dict(args); K = len(kc_dict); cache = Cache(a.model)
    system = kt_system_prompt(args)
    tag = f"api-{a.model}"
    for fold in a.folds:
        out = os.path.join(OUT_DIR, f"{tag}_{a.dataset}_fold{fold}.pkl")
        if os.path.exists(out) and not a.limit:
            print(f"fold {fold}: exists, skipping"); continue
        t0 = time.time(); tok_in = tok_out = 0; n_calls = 0
        _, _, test_df = load_annotated_data(args, fold)
        rows = list(test_df.iterrows())[: a.limit] if a.limit else list(test_df.iterrows())
        records = []
        for idx, sample in tqdm(rows, desc=f"fold {fold}"):
            dialogue = apply_annotations(sample)
            if not dialogue: continue
            turns = [t for t in dialogue if t["correct"] is not None and t["kcs"]]
            L = len(turns)
            if L < 2: continue
            labels = np.array([int(t["correct"]) for t in turns]); kcs = [[kc_dict[k] for k in t["kcs"]] for t in turns]
            states = np.full((L, K), np.nan); states_cf = np.full((L, K), np.nan)
            try:
                for i in range(L):
                    if i < L - 1:
                        kc_texts = list(dict.fromkeys(turns[i + 1]["kcs"] + turns[i]["kcs"])); turn_idx = turns[i + 1]["turn"]
                    else:
                        kc_texts = list(dict.fromkeys(turns[i]["kcs"])); turn_idx = None
                    users = [kt_user_prompt(sample, dialogue, turn_idx, kc, args) for kc in kc_texts]
                    with ThreadPoolExecutor(max_workers=a.workers) as ex:
                        results = list(ex.map(lambda u: p_true(client, a.model, system, u, cache, a.cached_only), users))
                    for kc, (p, ti, to) in zip(kc_texts, results):
                        states[i][kc_dict[kc]] = p; tok_in += ti; tok_out += to; n_calls += int(ti > 0)
            except NotCached:
                continue
            records.append({"dialogue_idx": idx, "labels": labels, "kcs": kcs, "states": states, "states_cf": states_cf})
        price = PRICE_PER_M.get(a.model, (0, 0)); cost = tok_in / 1e6 * price[0] + tok_out / 1e6 * price[1]
        if not a.limit:
            if a.cached_only: out = out.replace(".pkl", "_partial.pkl")
            with open(out, "wb") as f:
                pickle.dump({"model": tag + ("-partial" if a.cached_only else ""), "dataset": a.dataset, "fold": fold, "num_kcs": K, "records": records}, f)
        print(f"fold {fold}: {len(records)} dialogues, {n_calls} uncached calls, {tok_in} in / {tok_out} out tokens, est. ${cost:.2f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
