"""Score a calibration campaign: B1 rank uniformity, B2 knowledge-boundary drift, F1 single-token JSD.

    python analyse.py --runs run.jsonl --scorer /path/to/qwen3-0.6b-hf --out results.json

Arms are named by tag. Tags starting "q8-" are the unchanged model (the null); "q4-" is the planted
downgrade; "bf16" is the same model at higher precision; "reference" holds B1's reference samples.
The scorer is a fixed function of text (summed log-probability of the answer under the published
bf16 weights, run on CPU); B1 is valid for any fixed scorer because provider and reference samples
are compared with the same function.
"""
import argparse
import itertools
import json
import re
from collections import defaultdict

import numpy as np
from scipy.stats import cramervonmises

RNG = np.random.default_rng(20261029)
ALPHA = 0.05
NS = [10, 25, 50, 100, 200]
BOOT = 1000


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------- B1: rank uniformity ----------

def score_open(records, scorer_dir, cache, batch=16):
    """Summed log-probability of each answer under the scorer, cached to `cache` (one float per line)."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(cache):
        with open(cache) as f:
            done = [float(x) for x in f]
        if len(done) == len(records):
            return done
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(scorer_dir)
    # float32: bf16 matmuls on this CPU ran several times slower in the first campaign.
    model = AutoModelForCausalLM.from_pretrained(scorer_dir, dtype=torch.float32).eval()
    pairs = []
    for r in records:
        prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        a_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        pairs.append((p_ids, a_ids))
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            seqs = [p + a for p, a in chunk]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), tok.pad_token_id or 0)
            mask = torch.zeros_like(ids)
            for k, s in enumerate(seqs):
                ids[k, :len(s)] = torch.tensor(s)
                mask[k, :len(s)] = 1
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            for k, (p, a) in enumerate(chunk):
                if not a:
                    scores.append(0.0)
                    continue
                # Output layer only at the positions that predict answer tokens.
                pos = torch.arange(len(p) - 1, len(p) + len(a) - 1)
                logp = torch.log_softmax(model.lm_head(hidden[k, pos]), -1)
                scores.append(float(logp[torch.arange(len(a)), torch.tensor(a)].sum()))
            if (i // batch) % 25 == 0:
                print(f"scored {len(scores)}/{len(pairs)}", flush=True)
    with open(cache, "w") as f:
        f.write("\n".join(map(str, scores)) + "\n")
    return scores


def ranks(provider_scores, ref_scores):
    """Randomised rank percentile in (0, 1); exactly uniform when provider and reference share a distribution."""
    ref = np.asarray(ref_scores)
    below = np.sum(ref < provider_scores)
    ties = np.sum(ref == provider_scores)
    return (below + RNG.uniform(0, ties + 1)) / (len(ref) + 1)


def cvm_p(u):
    return float(cramervonmises(u, "uniform").pvalue)


def rejection_rate(u_by_prompt, n):
    keys = list(u_by_prompt)
    hits = 0
    for _ in range(BOOT):
        pick = RNG.choice(len(keys), size=n, replace=False)
        hits += cvm_p(np.array([u_by_prompt[keys[i]] for i in pick])) < ALPHA
    return hits / BOOT


def b1(rows, scorer, cache):
    open_rows = [r for r in rows if r["set"] == "open"]
    scores = score_open(open_rows, scorer, cache)
    ref = defaultdict(list)
    prov = defaultdict(dict)
    for r, s in zip(open_rows, scores):
        if r["tag"] == "reference":
            ref[r["id"]].append(s)
        else:
            prov[r["tag"]][r["id"]] = s
    u = {tag: {pid: ranks(s, ref[pid]) for pid, s in d.items()} for tag, d in prov.items()}
    out = {"reference_samples_per_prompt": int(np.median([len(v) for v in ref.values()])), "arms": {}}
    for tag, d in sorted(u.items()):
        vals = np.array(list(d.values()))
        out["arms"][tag] = {"n": len(vals), "median_u": float(np.median(vals)), "cvm_p_all": cvm_p(vals),
                            "reject_rate_by_n": {n: rejection_rate(d, n) for n in NS if n <= len(vals)}}
    # Routing dilution: each prompt's answer comes from the downgrade with probability p.
    nulls = sorted(t for t in u if t.startswith("q8-day"))
    downs = sorted(t for t in u if t.startswith("q4-day"))
    dil = {}
    for p in [0.1, 0.3, 0.5, 1.0]:
        rates = {}
        for n in NS:
            hits = 0
            for b in range(BOOT):
                null_tag, down_tag = nulls[b % len(nulls)], downs[b % len(downs)]
                pids = RNG.choice(sorted(u[null_tag]), size=n, replace=False)
                vals = [u[down_tag][k] if RNG.random() < p else u[null_tag][k] for k in pids]
                hits += cvm_p(np.array(vals)) < ALPHA
            rates[n] = hits / BOOT
        dil[p] = rates
    out["dilution_reject_rate"] = dil
    return out


# ---------- B2: knowledge-boundary drift ----------

def norm(text):
    """First number in the answer if there is one, else its first word."""
    t = text.strip().lower()
    m = re.search(r"-?\d[\d,]*", t)
    if m:
        return m.group(0).replace(",", "")
    w = re.findall(r"[a-z]+", t)
    return w[0] if w else t


def boot_frac(flags):
    flags = np.asarray(flags, float)
    if len(flags) == 0:
        return [None, None, None]
    bs = RNG.choice(flags, size=(BOOT, len(flags))).mean(1)
    return [float(flags.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def b2(rows, baseline="q8-day1"):
    ans = defaultdict(dict)
    gold = {}
    for r in rows:
        if r["set"] == "boundary":
            ans[r["tag"]][r["id"]] = norm(r["text"])
            gold[r["id"]] = r["answer"].lower()
    base = ans[baseline]
    wrong = [k for k in base if base[k] != gold[k]]
    out = {"baseline": baseline, "items": len(base), "baseline_wrong": len(wrong), "arms": {}}
    for tag, a in sorted(ans.items()):
        out["arms"][tag] = {
            "accuracy": float(np.mean([a[k] == gold[k] for k in a])),
            "changed_all": boot_frac([a[k] != base[k] for k in base]),
            "changed_on_baseline_wrong": boot_frac([a[k] != base[k] for k in wrong]),
        }
    return out


# ---------- F1: single-token fingerprint ----------

def dist(top):
    p = {k: float(np.exp(v)) for k, v in top.items()}
    p["<other>"] = max(0.0, 1.0 - sum(p.values()))
    return p


def jsd(p, q):
    keys = set(p) | set(q)
    a = np.array([p.get(k, 0.0) for k in keys])
    b = np.array([q.get(k, 0.0) for k in keys])
    m = (a + b) / 2

    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def f1(rows):
    d = defaultdict(dict)
    for r in rows:
        if r["set"] == "tokens":
            d[r["tag"]][r["id"]] = dist(r["top"])
    tags = sorted(d)
    probes = sorted(set.intersection(*(set(d[t]) for t in tags)))
    family = lambda t: t.split("-")[0]
    pairs = {}
    for a, b in itertools.combinations(tags, 2):
        per = np.array([jsd(d[a][k], d[b][k]) for k in probes])
        pairs[f"{a}|{b}"] = {"within_model": family(a) == family(b), "mean_jsd": float(per.mean()),
                             "max_probe_jsd": float(per.max()), "per_probe": per.tolist()}
    within = [v["mean_jsd"] for v in pairs.values() if v["within_model"]]
    cross = [v["mean_jsd"] for v in pairs.values() if not v["within_model"]]
    # Probes needed: smallest k at which every cross-model pair's mean JSD over k random probes
    # exceeds the largest within-model mean over all probes, in 95% of draws.
    threshold = max(within) if within else 0.0
    need = {}
    for key, v in pairs.items():
        if v["within_model"]:
            continue
        per = np.array(v["per_probe"])
        for k in [1, 2, 3, 5, 8, 10, 20, 40]:
            if k > len(per):
                break
            draws = np.array([per[RNG.choice(len(per), k, replace=False)].mean() for _ in range(BOOT)])
            if np.mean(draws > threshold) >= 0.95:
                need[key] = k
                break
        else:
            need[key] = None
    return {"probes": len(probes), "within_max": threshold, "within": within, "cross_min": min(cross) if cross else None,
            "probes_needed": need, "pairs": {k: {x: y for x, y in v.items() if x != "per_probe"} for k, v in pairs.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.runs)
    res = {"b2": b2(rows), "f1": f1(rows), "b1": b1(rows, a.scorer, a.out + ".scores")}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"b2": res["b2"]["arms"], "f1_within_max": res["f1"]["within_max"],
                      "f1_cross_min": res["f1"]["cross_min"], "f1_probes_needed": res["f1"]["probes_needed"],
                      "b1": {t: {k: v for k, v in x.items()} for t, x in res["b1"]["arms"].items()},
                      "b1_dilution": res["b1"]["dilution_reject_rate"]}, indent=1))
