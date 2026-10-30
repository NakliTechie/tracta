"""Score calibration campaigns: B1 rank uniformity, B2 knowledge-boundary drift, F1 single-token fingerprint.

    python analyse.py --runs runs.jsonl.gz runs-engine2.jsonl --scorer /path/to/qwen3-0.6b-hf --out results.json

Tags name arms: "<model>-<variant>", where model is q8, q4 or bf16 and a variant containing "hf"
means the transformers engine (collect_hf.py); everything else is llama-server. "reference" holds
B1's reference bank.

Every rate is over fresh, independent audits. An audit is a disjoint chunk of n prompts from one
answer sample of one arm (prompt order fixed by a seeded shuffle), so no answer is used twice
within a rate. Intervals are exact (Clopper-Pearson). Rates are conditional on the fixed
200-prompt probe set and the frozen reference bank, which is the setting a user's tool runs in.

B1's scorer is the summed log-probability of the answer under the published weights (float32, CPU).
Any fixed scorer keeps the test valid, provided provider and reference answers come from the same
per-prompt distribution under the same template, tokenizer and sampling settings.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, cramervonmises

SEED = 20261029
# Each statistic restarts this generator from SEED, so adding rows for one statistic (say a second
# engine's B2 answers) never changes another's random draws.
RNG = np.random.default_rng(SEED)
ALPHA = 0.05
AUDIT_SIZES = [10, 25, 50]
DILUTION = [0.1, 0.3, 0.5, 1.0]


def load(paths):
    rows = []
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            rows += [json.loads(line) for line in f]
    return rows


def model_of(tag):
    return tag.split("-")[0]


def engine_of(tag):
    return "transformers" if "-hf" in tag else "llama.cpp"


def pair_kind(a, b):
    if model_of(a) != model_of(b):
        return "cross-model"
    return "same-engine" if engine_of(a) == engine_of(b) else "cross-engine"


def rate(hits, n):
    """Observed rate with an exact 95% interval."""
    if n == 0:
        return None
    ci = binomtest(hits, n).proportion_ci(method="exact")
    return {"rate": hits / n, "lo": ci.low, "hi": ci.high, "k": hits, "n": n}


# ---------- B1: rank uniformity ----------

def score_open(records, scorer_dir, cache):
    """Summed answer log-probability under the scorer. Cached by a hash of scorer, dtype and every (prompt, text)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import transformers

    # The key covers the scorer's files (weights, tokenizer, config), library versions, dtype and every
    # (prompt, text), so a changed scorer or answer set can never reuse old scores.
    files = hashlib.sha256()
    for f in sorted(os.listdir(scorer_dir)):
        if f.endswith((".safetensors", ".json", ".txt", ".model")):
            with open(os.path.join(scorer_dir, f), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    files.update(block)
    key = hashlib.sha256(json.dumps([files.hexdigest(), torch.__version__, transformers.__version__, "float32"] +
                                    [[r["prompt"], r["text"]] for r in records]).encode()).hexdigest()
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        if c["key"] == key:
            return c["scores"], c["boundary_mismatches"]
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(scorer_dir)
    model = AutoModelForCausalLM.from_pretrained(scorer_dir, dtype=torch.float32).eval()
    pairs, mismatches = [], 0
    for r in records:
        prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        a_ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        # The answer is scored as its own token sequence after the prompt; count where that differs
        # from tokenizing the whole conversation, which would shift the scored positions.
        mismatches += tok(prefix + r["text"], add_special_tokens=False)["input_ids"] != p_ids + a_ids
        pairs.append((p_ids, a_ids))
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), 16):
            chunk = pairs[i:i + 16]
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
            if (i // 16) % 50 == 0:
                print(f"scored {len(scores)}/{len(pairs)}", flush=True)
    with open(cache, "w") as f:
        json.dump({"key": key, "scores": scores, "boundary_mismatches": int(mismatches)}, f)
    return scores, int(mismatches)


def rank(score, ref_scores):
    """Randomised rank percentile in (0, 1); exactly uniform when provider and reference share a distribution."""
    ref = np.asarray(ref_scores)
    return (np.sum(ref < score) + RNG.uniform(0, np.sum(ref == score) + 1)) / (len(ref) + 1)


def cvm_p(u):
    return float(cramervonmises(u, "uniform").pvalue)


def audits(samples, order, n):
    """Disjoint audits of n prompts: for each answer sample j, consecutive chunks of the fixed prompt order."""
    out = []
    for j in sorted(samples):
        pids = [p for p in order if p in samples[j]]
        out += [[(j, p) for p in pids[i:i + n]] for i in range(0, len(pids) - n + 1, n)]
    return out


def b1(rows, scorer, cache):
    global RNG
    RNG = np.random.default_rng(SEED)
    open_rows = [r for r in rows if r["set"] == "open"]
    scores, mismatches = score_open(open_rows, scorer, cache)
    ref = defaultdict(list)
    u = defaultdict(lambda: defaultdict(dict))  # tag -> sample j -> prompt -> rank
    raw = defaultdict(lambda: defaultdict(dict))
    for r, s in zip(open_rows, scores):
        if r["tag"] == "reference":
            ref[r["id"]].append(s)
        else:
            raw[r["tag"]][r["j"]][r["id"]] = s
    for tag in raw:
        for j in raw[tag]:
            u[tag][j] = {p: rank(s, ref[p]) for p, s in raw[tag][j].items()}
    order = sorted(ref)
    RNG.shuffle(order)
    out = {"reference_samples_per_prompt": int(np.median([len(v) for v in ref.values()])),
           "scorer_boundary_mismatches": mismatches, "scored_answers": len(open_rows), "arms": {}}
    for tag in sorted(u):
        allu = np.array([x for j in u[tag] for x in u[tag][j].values()])
        arm = {"answers": len(allu), "median_rank": float(np.median(allu)), "reject": {}}
        for n in AUDIT_SIZES:
            a = audits(u[tag], order, n)
            hits = sum(cvm_p(np.array([u[tag][j][p] for j, p in audit])) < ALPHA for audit in a)
            arm["reject"][n] = rate(int(hits), len(a))
        out["arms"][tag] = arm
    # Dilution: pair each fresh audit of a changed model with a fresh audit of the unchanged one on
    # the same engine and prompts; each prompt's answer comes from the changed model with probability p.
    dil = {}
    for change, (null, down) in {"q4 on llama.cpp": ("q8-audit", "q4-audit"),
                                 "swap on llama.cpp": ("q8-audit", "swap-audit"),
                                 "q4 on transformers": ("q8-hfaudit", "q4-hfaudit")}.items():
        if null not in u or down not in u:
            continue
        engine_tag = change
        dil[engine_tag] = {}
        for p in DILUTION:
            dil[engine_tag][p] = {}
            for n in AUDIT_SIZES:
                a = [x for x in audits(u[down], order, n) if all(q in u[null].get(j, {}) for j, q in x)]
                hits = 0
                for audit in a:
                    vals = [u[down][j][q] if RNG.random() < p else u[null][j][q] for j, q in audit]
                    hits += cvm_p(np.array(vals)) < ALPHA
                dil[engine_tag][p][n] = rate(int(hits), len(a))
    out["dilution"] = dil
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


def b2(rows, baseline="q8-day1"):
    ans = defaultdict(dict)
    gold = {}
    for r in rows:
        if r["set"] == "boundary":
            ans[r["tag"]][r["id"]] = norm(r["text"])
            gold[r["id"]] = r["answer"].lower()
    base = ans[baseline]
    wrong = [k for k in base if base[k] != gold[k]]
    out = {"baseline": baseline, "items": len(base), "baseline_wrong": len(wrong), "arms": {}, "pairs": {}}
    for tag, a in sorted(ans.items()):
        out["arms"][tag] = {
            "accuracy": float(np.mean([a[k] == gold[k] for k in a])),
            "changed_all": rate(sum(a[k] != base[k] for k in base), len(base)),
            "changed_on_baseline_wrong": rate(sum(a[k] != base[k] for k in wrong), len(wrong)),
        }
    for x, y in itertools.combinations(sorted(ans), 2):
        out["pairs"][f"{x}|{y}"] = {"kind": pair_kind(x, y),
                                    "changed_all": rate(sum(ans[x][k] != ans[y][k] for k in base), len(base))}
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


def eer(same, cross):
    """Equal error rate of the rule 'call it a different model when the distance exceeds t'."""
    same, cross = np.sort(same), np.sort(cross)
    best = 1.0
    for t in np.unique(np.concatenate([same, cross, [-1.0]])):
        fpr = np.mean(same > t)
        fnr = np.mean(cross <= t)
        best = min(best, max(fpr, fnr))
    return float(best)


def f1(rows, draws=2000):
    global RNG
    RNG = np.random.default_rng(SEED)
    d = defaultdict(dict)
    for r in rows:
        if r["set"] == "tokens":
            d[r["tag"]][r["id"]] = dist(r["top"])
    tags = sorted(d)
    probes = sorted(set.intersection(*(set(d[t]) for t in tags)))
    per = {}
    pairs = {}
    for a, b in itertools.combinations(tags, 2):
        v = np.array([jsd(d[a][k], d[b][k]) for k in probes])
        per[(a, b)] = v
        pairs[f"{a}|{b}"] = {"kind": pair_kind(a, b), "mean_jsd": float(v.mean()), "max_probe_jsd": float(v.max())}
    same = [k for k in per if pair_kind(*k) != "cross-model"]
    cross = [k for k in per if pair_kind(*k) == "cross-model"]
    # The predeclared rule (hypotheses §7): equal error rate between same-model distances (any engine,
    # any serving setting) and cross-model distances, for a check that averages k random probes.
    # Reported per change being detected (q4 against q8, bf16 against q8, ...), because one pooled
    # rate mixes an easy change with one that sits inside engine noise.
    changes = defaultdict(list)
    for a, b in cross:
        changes["|".join(sorted({model_of(a), model_of(b)}))].append((a, b))
    pick = lambda keys, k: np.array([per[keys[RNG.integers(len(keys))]][RNG.choice(len(probes), k, replace=False)].mean()
                                     for _ in range(draws)])
    rule = {}
    if same:
        for change, keys in sorted(changes.items()):
            rule[change] = {k: eer(pick(same, k), pick(keys, k)) for k in [1, 3, 5, 10, 40] if k <= len(probes)}
    return {"probes": len(probes), "eer_by_change_and_probes": rule, "pairs": pairs,
            "max_same_model_mean_jsd": max(pairs[f"{a}|{b}"]["mean_jsd"] for a, b in same) if same else None,
            "min_cross_model_mean_jsd": min(pairs[f"{a}|{b}"]["mean_jsd"] for a, b in cross) if cross else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, nargs="+", help="one or more runs files (.jsonl or .jsonl.gz)")
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.runs)
    res = {"inputs": a.runs, "b2": b2(rows), "f1": f1(rows), "b1": b1(rows, a.scorer, a.out + ".scores.json")}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"b1": res["b1"]["arms"], "b1_dilution": res["b1"]["dilution"],
                      "f1_eer": res["f1"]["eer_by_change_and_probes"]}, indent=1))
