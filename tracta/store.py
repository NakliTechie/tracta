"""Local storage: one folder per watched provider, holding its config, baseline runs, checks and reports.

    <root>/<name>/config.json
    <root>/<name>/runs/baseline-1.json, baseline-2.json, ...
    <root>/<name>/runs/check-1.json, ...
    <root>/<name>/reports/check-1.md, check-1.json
"""
import json
import pathlib
import secrets
import time

DEFAULT_ROOT = pathlib.Path.home() / ".tracta"


class Store:
    def __init__(self, root, name):
        self.dir = pathlib.Path(root).expanduser() / name
        self.name = name

    @property
    def config_path(self):
        return self.dir / "config.json"

    def exists(self):
        return self.config_path.exists()

    def create(self, endpoint, model, key_env, extra_body):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "runs").mkdir(exist_ok=True)
        (self.dir / "reports").mkdir(exist_ok=True)
        config = {"name": self.name, "endpoint": endpoint, "model": model, "key_env": key_env,
                  "extra_body": extra_body or {}, "seed": secrets.randbits(32),
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self.config_path.write_text(json.dumps(config, indent=1) + "\n")
        return config

    def config(self):
        return json.loads(self.config_path.read_text())

    def runs(self, kind):
        files = sorted((self.dir / "runs").glob(f"{kind}-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        return [json.loads(p.read_text()) for p in files]

    def save_run(self, kind, run):
        number = len(self.runs(kind)) + 1
        run = dict(run, kind=kind, number=number)
        (self.dir / "runs" / f"{kind}-{number}.json").write_text(json.dumps(run, ensure_ascii=False) + "\n")
        return run

    def report_path(self, number, suffix):
        return self.dir / "reports" / f"check-{number}.{suffix}"
