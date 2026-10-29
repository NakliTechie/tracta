"""Play the provider: serve one GGUF with llama-server on CPU and record its answers to the probe sets.

One call of this script is one "arm" (a model plus a serving configuration plus a seed base).
Answers are appended to one JSONL file, one line per request, so every arm of a campaign lands
in a single aggregate file.

    python collect.py --model Q8.gguf --tag q8-day1 --sets open,boundary,tokens --out run.jsonl

CPU only (--device none, no op offload), so no GPU turn is needed.
"""
import argparse
import json
import subprocess
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import probes

OPEN = dict(temperature=0.7, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, max_tokens=48)
BOUNDARY = dict(temperature=0.0, max_tokens=16)
TOKENS = dict(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, max_tokens=1, logprobs=True, top_logprobs=20)


def seed_for(base, pid, j):
    return zlib.crc32(f"{base}:{pid}:{j}".encode()) & 0x7FFFFFFF


def start_server(model, port, threads, parallel, log):
    cmd = ["llama-server", "-m", model, "--device", "none", "-ngl", "0", "--no-op-offload",
           "--host", "127.0.0.1", "--port", str(port), "--jinja", "-c", str(2048 * parallel),
           "--parallel", str(parallel), "-t", str(threads)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return proc
        except OSError:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        time.sleep(1)
    proc.kill()
    raise RuntimeError("llama-server did not become healthy")


def chat(port, prompt, params, seed):
    body = dict(params, seed=seed, messages=[{"role": "user", "content": prompt}],
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def jobs(sets, open_samples, seed_base):
    if "open" in sets:
        for p in probes.open_prompts():
            for j in range(open_samples):
                yield "open", p, j, OPEN, seed_for(seed_base, p["id"], j)
    if "boundary" in sets:
        for p in probes.boundary_prompts():
            yield "boundary", p, 0, BOUNDARY, seed_for(seed_base, p["id"], 0)
    if "tokens" in sets:
        for p in probes.token_probes():
            yield "tokens", p, 0, TOKENS, seed_for(seed_base, p["id"], 0)


def run(args):
    with open(args.log, "a") as log:
        server = start_server(args.model, args.port, args.threads, args.parallel, log)
    try:
        def one(job):
            kind, p, j, params, seed = job
            d = chat(args.port, p["prompt"], params, seed)
            c = d["choices"][0]
            rec = dict(tag=args.tag, model=args.model, build=d.get("system_fingerprint"), set=kind,
                       id=p["id"], j=j, seed=seed, prompt=p["prompt"], text=c["message"]["content"])
            if "answer" in p:
                rec["answer"] = p["answer"]
            if kind == "tokens":
                first = c["logprobs"]["content"][0]
                rec["top"] = {t["token"]: t["logprob"] for t in first["top_logprobs"]}
            return rec

        with ThreadPoolExecutor(args.concurrency) as pool, open(args.out, "a") as out:
            for n, rec in enumerate(pool.map(one, jobs(args.sets.split(","), args.open_samples, args.seed_base)), 1):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 200 == 0:
                    out.flush()
                    print(f"{args.tag}: {n} requests", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="open,boundary,tokens")
    ap.add_argument("--open-samples", type=int, default=1)
    ap.add_argument("--seed-base", default=None, help="defaults to the tag")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--port", type=int, default=18731)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    a.seed_base = a.seed_base or a.tag
    run(a)
