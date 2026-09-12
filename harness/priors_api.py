"""Text-informed priors: for every dialogue, ask an LLM at the *opening* prompt (dialogue up to the teacher's
first assessed question, no student answer observed yet) whether the student has each skill that occurs
anywhere in the dialogue. Output: harness/priors/<model>_<dataset>.json  {dialogue_idx: {kc_id: P(True)}}.

Run from the dialogue-kt directory:
    OPENAI_API_KEY=... ../.venv/bin/python ../harness/priors_api.py --model gpt-4.1
Cached through the same cache as dump_api_llm.py, so reruns are free.
"""
import argparse, json, os, sys, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.data_loading import load_annotated_data, load_kc_dict  # noqa: E402
from dialogue_kt.kt_data_loading import apply_annotations  # noqa: E402
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt  # noqa: E402
from dump_api_llm import Cache, PRICE_PER_M, make_args, p_true  # noqa: E402

PRIOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "priors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comta")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(); os.makedirs(PRIOR_DIR, exist_ok=True)
    args = make_args(a.dataset); kc_dict = load_kc_dict(args); cache = Cache(a.model)
    system = kt_system_prompt(args)
    # every dialogue once: union of the fold-1 train/val/test splits is the whole dataset
    tr, va, te = load_annotated_data(args, 1)
    df = pd.concat([tr, va, te])
    out_path = os.path.join(PRIOR_DIR, f"api-{a.model}_{a.dataset}.json")
    priors = json.load(open(out_path)) if os.path.exists(out_path) else {}
    t0 = time.time(); tok_in = tok_out = 0; n_calls = 0
    for idx, sample in tqdm(list(df.iterrows()), desc="priors"):
        if str(idx) in priors:
            continue
        dialogue = apply_annotations(sample)
        if not dialogue:
            continue
        turns = [t for t in dialogue if t["correct"] is not None and t["kcs"]]
        if len(turns) < 2:
            continue
        all_kcs = list(dict.fromkeys(kc for t in turns for kc in t["kcs"]))
        first_turn = turns[0]["turn"]   # prompt stops after the teacher's question of the first assessed turn
        users = [kt_user_prompt(sample, dialogue, first_turn, kc, args) for kc in all_kcs]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(lambda u: p_true(client, a.model, system, u, cache), users))
        priors[str(idx)] = {str(kc_dict[kc]): p for kc, (p, _, _) in zip(all_kcs, results)}
        for _, ti, to in results:
            tok_in += ti; tok_out += to; n_calls += int(ti > 0)
        json.dump(priors, open(out_path, "w"))
    price = PRICE_PER_M.get(a.model, (0, 0)); cost = tok_in / 1e6 * price[0] + tok_out / 1e6 * price[1]
    print(f"{len(priors)} dialogues, {n_calls} uncached calls, {tok_in} in / {tok_out} out tokens, est. ${cost:.2f} ({time.time()-t0:.0f}s) -> {out_path}")


if __name__ == "__main__":
    main()
