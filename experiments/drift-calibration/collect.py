"""Play the provider: serve one GGUF with llama-server on CPU and record its answers to the probe sets.

One call of this script is one "arm" (a model plus a serving configuration plus a seed base).
Answers are appended to one JSONL file, one line per request, so every arm of a campaign lands
in a single aggregate file.

    python collect.py --model Q8.gguf --tag q8-day1 --sets open,boundary,tokens --out run.jsonl

CPU only (--device none, no op offload), so no GPU turn is needed.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import probes

OPEN = dict(temperature=0.7, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, max_tokens=48)
BOUNDARY = dict(temperature=0.0, repeat_penalty=1.0, max_tokens=16)
TOKENS = dict(temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, max_tokens=1, logprobs=True,
              top_logprobs=20)


def seed_for(base, pid, j):
    return zlib.crc32(f"{base}:{pid}:{j}".encode()) & 0x7FFFFFFF


def pressure_level():
    """macOS memory pressure: 1 normal, 2 warn, 4 critical. Swap use alone is no signal, because
    macOS keeps swap allocated after the pressure has passed."""
    return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                              capture_output=True, text=True).stdout.strip() or 1)


def memory_gate(label, gate_log):
    """Before a memory-heavy step, wait while memory pressure is critical (another session may
    hold memory). Writes WAIT and GO lines to gate_log (gate.log beside the runs file) so the
    supervising agent can ask the other session to pause. Proceeds after 60 minutes, logging it."""
    def note(msg):
        with open(gate_log, "a") as f:
            f.write(f"{time.strftime('%H:%M')} {label} {msg}\n")
    waited = 0
    while pressure_level() >= 4 and waited < 3600:
        if waited == 0:
            note(f"WAIT pressure level {pressure_level()}")
        time.sleep(60)
        waited += 60
    note(f"GO pressure level {pressure_level()} after {waited} s")


def start_server(model, port, threads, parallel, log):
    # CPU by default, so no GPU turn is needed; TRACTA_DEVICE=MTL0 serves every layer on Metal instead.
    device = os.environ.get("TRACTA_DEVICE", "none")
    placement = ["--device", "none", "-ngl", "0", "--no-op-offload"] if device == "none" else ["--device", device, "-ngl", "999"]
    cmd = ["llama-server", "-m", model, *placement,
           "--host", "127.0.0.1", "--port", str(port), "--jinja", "-c", str(2048 * parallel),
           "--parallel", str(parallel), "-t", str(threads)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup (port in use?)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                healthy = json.load(r).get("status") == "ok"
            if healthy:
                # A server already on this port also answers /health; confirm it is ours and serves this file.
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as r:
                    served = json.load(r).get("model_path", "")
                if proc.poll() is not None or os.path.realpath(served) != os.path.realpath(model):
                    proc.kill()
                    raise RuntimeError(f"port {port} is served by another llama-server ({served})")
                return proc
        except OSError:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("llama-server did not become healthy")


def chat(port, prompt, params, seed):
    body = dict(params, seed=seed, messages=[{"role": "user", "content": prompt}],
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    # A shared, oversubscribed CPU can stall llama.cpp's spinning threads for minutes; wait long, retry once.
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except TimeoutError:
            if attempt == 2:
                raise


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
    memory_gate(args.tag, os.path.join(os.path.dirname(os.path.abspath(args.out)), "gate.log"))
    with open(args.log, "a") as log:
        server = start_server(args.model, args.port, args.threads, args.parallel, log)
    try:
        def one(job):
            kind, p, j, params, seed = job
            t0 = time.time()
            d = chat(args.port, p["prompt"], params, seed)
            c = d["choices"][0]
            rec = dict(tag=args.tag, model=args.model, build=d.get("system_fingerprint"), set=kind,
                       id=p["id"], j=j, seed=seed, prompt=p["prompt"], text=c["message"]["content"],
                       # Logged, not used: timings are not authoritative when the machine is shared.
                       wall_s=round(time.time() - t0, 4), timings=d.get("timings"))
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
    ap.add_argument("--port", type=int, default=int(os.environ.get("TRACTA_PORT", 18731)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    a.seed_base = a.seed_base or a.tag
    run(a)
