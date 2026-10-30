"""tracta: check whether a model provider still serves the model it promised.

    tracta baseline --endpoint URL --model NAME --name myprovider [--key-env VAR]   # first run creates the watch
    tracta baseline --name myprovider                                              # later baseline runs
    tracta check --name myprovider                                                 # writes a report
    tracta status --name myprovider

Exit codes for `check`: 0 no change detected, 2 changed, 3 no verdict (fewer than two baseline runs).
"""
import argparse
import json
import sys
import time

from . import __version__
from .compare import evaluate
from .endpoint import Endpoint, EndpointError
from .questions import TOKEN_PROBES, boundary_items
from .report import write
from .store import DEFAULT_ROOT, Store


def collect(config, items, logprobs, rps, progress=True):
    ep = Endpoint(config["endpoint"], config["model"], config.get("key_env"), config.get("extra_body"), rps=rps)
    answers, prompts = {}, {}
    for k, item in enumerate(items, 1):
        answers[item["id"]], _ = ep.chat(item["prompt"])
        prompts[item["id"]] = item["prompt"]
        if progress and k % 50 == 0:
            print(f"  {k}/{len(items)} questions", file=sys.stderr, flush=True)
    tokens = None
    if logprobs:
        tokens = {}
        try:
            for i, p in enumerate(TOKEN_PROBES):
                _, top = ep.chat(p, max_tokens=1, top_logprobs=20)
                if top:
                    tokens[f"tok-{i:02d}"] = top
        except EndpointError as e:
            print(f"  fingerprint probes skipped: {e}", file=sys.stderr)
            tokens = None
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "answers": answers, "prompts": prompts,
            "tokens": tokens or None}


def items_for(config, limit):
    items = boundary_items(config["seed"])
    return items[:limit] if limit else items


def cmd_baseline(args):
    store = Store(args.root, args.name)
    if not store.exists():
        if not (args.endpoint and args.model):
            sys.exit(f"no watch named {args.name!r} yet: give --endpoint and --model to create it")
        extra = json.loads(args.extra_body) if args.extra_body else None
        config = store.create(args.endpoint, args.model, args.key_env, extra)
        print(f"created watch {args.name!r} in {store.dir}")
    else:
        config = store.config()
    items = items_for(config, args.items)
    print(f"baseline {len(store.runs('baseline')) + 1} for {config['model']} at {config['endpoint']}: "
          f"{len(items)} questions", file=sys.stderr)
    run = store.save_run("baseline", collect(config, items, args.logprobs, args.rps))
    print(f"saved baseline {run['number']}")
    if run["number"] == 1:
        print("record a second baseline (ideally on another day) before checking, so tracta can measure "
              "the provider's own noise")


def cmd_check(args):
    store = Store(args.root, args.name)
    if not store.exists():
        sys.exit(f"no watch named {args.name!r}: run `tracta baseline` first")
    config = store.config()
    baselines = store.runs("baseline")
    if not baselines:
        sys.exit("no baseline runs yet: run `tracta baseline` first")
    items = [i for i in boundary_items(config["seed"]) if i["id"] in baselines[0]["answers"]]
    with_tokens = bool(baselines[0].get("tokens"))
    print(f"check {len(store.runs('check')) + 1}: {len(items)} questions", file=sys.stderr)
    run = store.save_run("check", collect(config, items, with_tokens, args.rps))
    tested_before = sum(1 for p in sorted((store.dir / "reports").glob("check-*.json"))
                        if "p_value" in json.loads(p.read_text()))
    result = evaluate(baselines, run, run["number"], args.alpha, tested_before)
    path = write(store, config, result)
    print(f"{result['verdict']}: {result['changed']} of {result['items']} answers changed"
          + (f", provider noise {100 * result['null_rate']:.1f}%, p = {result['p_value']:.3g}" if "p_value" in result else ""))
    print(f"report: {path}")
    return {"changed": 2, "no change detected": 0}.get(result["verdict"], 3)


def cmd_status(args):
    store = Store(args.root, args.name)
    if not store.exists():
        sys.exit(f"no watch named {args.name!r}")
    config = store.config()
    print(json.dumps({"name": config["name"], "endpoint": config["endpoint"], "model": config["model"],
                      "baselines": len(store.runs("baseline")), "checks": len(store.runs("check")),
                      "folder": str(store.dir)}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracta", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=f"tracta {__version__}")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="folder for watches (default ~/.tracta)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="record answers on a day you trust")
    b.add_argument("--name", required=True)
    b.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. https://api.example.com/v1")
    b.add_argument("--model", help="model name the provider expects")
    b.add_argument("--key-env", help="environment variable that holds the API key")
    b.add_argument("--extra-body", help="JSON merged into every request body (provider-specific fields)")
    b.add_argument("--items", type=int, default=0, help="use only the first N questions (default all 436)")
    b.add_argument("--logprobs", action="store_true", help="also record the fingerprint probes (needs top_logprobs)")
    b.add_argument("--rps", type=float, default=2.0, help="requests per second (default 2)")
    c = sub.add_parser("check", help="compare today's answers with the baseline and write a report")
    c.add_argument("--name", required=True)
    c.add_argument("--alpha", type=float, default=0.05, help="total false-alarm budget over all checks (default 0.05)")
    c.add_argument("--rps", type=float, default=2.0)
    s = sub.add_parser("status", help="show a watch")
    s.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    try:
        code = {"baseline": cmd_baseline, "check": cmd_check, "status": cmd_status}[args.cmd](args)
    except EndpointError as e:
        sys.exit(f"tracta: {e}")
    sys.exit(code or 0)
