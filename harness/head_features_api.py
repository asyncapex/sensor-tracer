"""Prediction-head inputs read by an LLM from the dialogue up to the tutor's question of the predicted turn
(the student's answer is never shown): per-skill relevance weights, question difficulty, all-skills-needed flag.

    OPENAI_API_KEY=... ../.venv/bin/python ../harness/head_features_api.py --dataset comta --model gpt-4.1
    OPENAI_API_KEY=... ../.venv/bin/python ../harness/head_features_api.py --dataset mathdial --model gpt-4.1
Output: harness/head/<model>_<dataset>.json  {dialogue_idx: {turn i: {"w": {kc_id: weight}, "difficulty": d, "all": bool}}}
for every labelled turn i >= 1 of every dialogue (train, val and test). Cached per prompt; resumable.
"""
import argparse, hashlib, json, os, sys, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.data_loading import load_annotated_data, load_kc_dict  # noqa: E402
from dialogue_kt.kt_data_loading import apply_annotations  # noqa: E402
from dialogue_kt.prompting import get_dialogue_text, get_mathdial_context  # noqa: E402
from dump_api_llm import Cache, PRICE_PER_M, make_args  # noqa: E402

HEAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head")
SYSTEM = """You are an experienced math teacher. You are given a tutoring dialogue up to the teacher's most recent question, and the list of knowledge components (skills) that question involves. Judge the question itself, not the student.
Return only a JSON object with three fields:
- "weights": an object mapping each skill's index (as given) to a number in [0,1]; the weights must sum to 1 and express how much answering this question correctly depends on each skill.
- "difficulty": a number in [0,1], how hard this question is for a typical student at this level (0 trivial, 1 very hard).
- "all": true if the student must have every listed skill to answer correctly, false if strength in one skill can compensate for weakness in another."""


def user_prompt(sample, dialogue, turn_idx, kcs, dataset):
    p = (get_mathdial_context(sample) + "\n\n") if dataset == "mathdial" else ""
    p += get_dialogue_text(dialogue, turn_idx=turn_idx)
    p += "\n\nSkills involved in the teacher's most recent question:\n" + "\n".join(f"{j}: {kc}" for j, kc in enumerate(kcs))
    return p


def ask(client, model, system, user, cache):
    key = hashlib.sha256(f"head\n{model}\n{system}\n{user}".encode()).hexdigest()
    hit = cache.get(key)
    if hit: return hit["obj"], 0, 0
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                               max_tokens=200, temperature=0, response_format={"type": "json_object"})
            obj = json.loads(r.choices[0].message.content); break
        except Exception:
            if attempt == 4: raise
            time.sleep(2 ** attempt)
    cache.put(key, {"obj": obj, "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens})
    return obj, r.usage.prompt_tokens, r.usage.completion_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comta"); ap.add_argument("--model", default="gpt-4.1"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(); os.makedirs(HEAD_DIR, exist_ok=True)
    args = make_args(a.dataset); kc_dict = load_kc_dict(args); cache = Cache("head-" + a.model)
    tr, va, te = load_annotated_data(args, 1); df = pd.concat([tr, va, te])
    out_path = os.path.join(HEAD_DIR, f"{a.model}_{a.dataset}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    t0 = time.time(); tok_in = tok_out = 0; n = 0
    for idx, sample in tqdm(list(df.iterrows()), desc=f"head features {a.dataset}"):
        if str(idx) in out: continue
        dialogue = apply_annotations(sample)
        if not dialogue: continue
        turns = [t for t in dialogue if t["correct"] is not None and t["kcs"]]
        if len(turns) < 2: continue
        jobs = [(i, turns[i]) for i in range(1, len(turns))]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(lambda it: ask(client, a.model, SYSTEM, user_prompt(sample, dialogue, it[1]["turn"], it[1]["kcs"], a.dataset), cache), jobs))
        rec = {}
        for (i, t), (obj, ti, to) in zip(jobs, res):
            kcs = t["kcs"]; w = obj.get("weights", {}) if isinstance(obj, dict) else {}
            ws = []
            for j, kc in enumerate(kcs):
                v = w.get(str(j), w.get(j, None)) if isinstance(w, dict) else None
                ws.append(float(v) if isinstance(v, (int, float)) else 1.0 / len(kcs))
            s = sum(ws) or 1.0; ws = [x / s for x in ws]
            d = obj.get("difficulty", 0.5) if isinstance(obj, dict) else 0.5
            rec[str(i)] = {"w": {str(kc_dict[kc]): round(x, 4) for kc, x in zip(kcs, ws)},
                           "difficulty": float(d) if isinstance(d, (int, float)) else 0.5, "all": bool(obj.get("all", False)) if isinstance(obj, dict) else False}
            tok_in += ti; tok_out += to; n += int(ti > 0)
        out[str(idx)] = rec
        json.dump(out, open(out_path, "w"))
    price = PRICE_PER_M.get(a.model, (0, 0)); cost = tok_in / 1e6 * price[0] + tok_out / 1e6 * price[1]
    print(f"{len(out)} dialogues, {n} uncached calls, {tok_in} in / {tok_out} out tokens, est. ${cost:.2f} ({time.time()-t0:.0f}s) -> {out_path}")


if __name__ == "__main__":
    main()
