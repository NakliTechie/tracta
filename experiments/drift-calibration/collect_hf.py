"""Second engine: the same GGUF file, dequantised and run by transformers on CPU in float32.

Same probe sets, sampling settings and record format as collect.py, so its arms drop into the
same runs file. This is the noisy null: one model file, a different engine, which is what a
hosted provider looks like next to a local reference.

    python collect_hf.py --model-dir D --gguf model-Q8_0.gguf --tag q8-hfday1 --out run.jsonl

Prompt token IDs come from llama-server running the same file (chat template applied with
thinking off, special tokens parsed), so both engines see identical input tokens and only the
arithmetic differs. Loading a published GGUF's tokenizer in transformers is not enough: it drops
the chat template and splits added tokens such as <think> into text.

--batch > 1 runs open and boundary prompts in left-padded batches, which changes the arithmetic
the way server-side batching does. Sampling is seeded once per batch, so a sampled answer from a
batch of 8 can be reproduced only by rerunning its whole batch; greedy answers do not depend on it. Needs a gguf package that works with NumPy 2 (the project .venv).
"""
import argparse
import json
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probes
import os
import time

from collect import BOUNDARY, OPEN, memory_gate, seed_for, start_server


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def prompt_ids(args, prompts):
    """Token IDs of each prompt, formatted and tokenized by llama-server on the same GGUF."""
    with open(args.out + ".server.log", "a") as log:
        server = start_server(f"{args.model_dir}/{args.gguf}", args.port, 4, 1, log)
    try:
        ids = {}
        for p in prompts:
            text = post(args.port, "/apply-template", {"messages": [{"role": "user", "content": p}],
                                                       "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            # add_special adds the model's start token where its metadata asks for one (Gemma's <bos>,
            # none for Qwen), which is what llama-server does for a chat request.
            ids[p] = post(args.port, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
        return ids
    finally:
        server.terminate()
        server.wait(timeout=30)


# End-of-generation tokens, after llama.cpp's own list: a published GGUF's tokenizer in transformers
# knows only the file's eos token, so a chat turn would not stop at <end_of_turn> or <|eot_id|>.
EOG_TEXTS = ["<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "<|endoftext|>", "<eos>", "</s>"]


def eog_ids(tok):
    vocab = tok.get_vocab()
    return sorted({tok.eos_token_id} | {vocab[t] for t in EOG_TEXTS if t in vocab} - {None})


def generate(tok, model, id_lists, params, seed):
    width = max(map(len, id_lists))
    stop = eog_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(id_lists), width), pad)
    mask = torch.zeros_like(ids)
    for k, s in enumerate(id_lists):  # left padding
        ids[k, width - len(s):] = torch.tensor(s)
        mask[k, width - len(s):] = 1
    sample = params["temperature"] > 0
    # Explicit values override the model's generation_config (Qwen's is temperature 0.6, top-k 20, top-p 0.95).
    kw = dict(do_sample=True, temperature=params["temperature"], top_k=0, top_p=1.0) if sample else dict(do_sample=False)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=params["max_tokens"],
                             pad_token_id=pad, eos_token_id=stop, repetition_penalty=1.0, **kw)
    texts = []
    for o in out:
        new = o[width:].tolist()
        cut = next((i for i, t in enumerate(new) if t in stop), len(new))  # drop the stop token and padding
        texts.append(tok.decode(new[:cut], skip_special_tokens=True))
    return texts


def top_tokens(tok, model, id_list, k=20):
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([id_list])).logits[0, -1].float(), -1)
    top = logp.topk(k)
    return {tok.decode([int(i)]): float(v) for v, i in zip(top.values, top.indices)}


def run(args):
    torch.set_num_threads(args.threads)
    sets = args.sets.split(",")
    work = []
    if "open" in sets:
        work += [("open", OPEN, j, probes.open_prompts()) for j in range(args.open_samples)]
    if "boundary" in sets:
        work.append(("boundary", BOUNDARY, 0, probes.boundary_prompts()))
    token_items = probes.token_probes() if "tokens" in sets else []
    memory_gate(args.tag, os.path.join(os.path.dirname(os.path.abspath(args.out)), "gate.log"))
    ids = prompt_ids(args, {p["prompt"] for _, _, _, items in work for p in items} | {p["prompt"] for p in token_items})

    tok = AutoTokenizer.from_pretrained(args.model_dir, gguf_file=args.gguf)  # used for decoding only
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, gguf_file=args.gguf, dtype=torch.float32).eval()
    base = dict(tag=args.tag, model=f"{args.model_dir}/{args.gguf}", build="transformers-cpu-fp32")
    with open(args.out, "a") as out:
        def emit(rec):
            out.write(json.dumps(dict(base, **rec), ensure_ascii=False) + "\n")

        for kind, params, j, items in work:
            for i in range(0, len(items), args.batch):
                chunk = items[i:i + args.batch]
                seed = seed_for(args.tag, chunk[0]["id"], j)
                t0 = time.time()
                texts = generate(tok, model, [ids[p["prompt"]] for p in chunk], params, seed)
                wall = round((time.time() - t0) / len(chunk), 4)  # logged, not used
                for p, text in zip(chunk, texts):
                    rec = dict(set=kind, id=p["id"], j=j, seed=seed, prompt=p["prompt"], text=text, wall_s=wall)
                    if "answer" in p:
                        rec["answer"] = p["answer"]
                    emit(rec)
            out.flush()
            print(f"{args.tag}: {kind} sample {j} done", flush=True)
        for p in token_items:
            emit(dict(set="tokens", id=p["id"], j=0, seed=0, prompt=p["prompt"], text="",
                      top=top_tokens(tok, model, ids[p["prompt"]])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="open,boundary,tokens")
    ap.add_argument("--open-samples", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=18741)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
