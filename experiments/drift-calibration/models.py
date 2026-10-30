"""Pinned model files, so every result can be rerun against the same published bytes.

    python models.py record DIR...         # write models.json from files downloaded with `hf download --local-dir`
    python models.py fetch DEST            # download every file in models.json at its pinned revision, check SHA-256
    python models.py verify DEST           # check SHA-256 of files already in DEST

models.json lists, per file: name (the key results use), repo, file, revision (commit), sha256, size.
The revision comes from the hf download metadata; the SHA-256 is computed locally and must equal the
LFS hash Hugging Face publishes for that revision.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).parent
MANIFEST = HERE / "models.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def published_sha(repo, revision, file):
    url = f"https://huggingface.co/api/models/{repo}/paths-info/{revision}"
    req = urllib.request.Request(url, data=json.dumps({"paths": [file]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.load(r)[0]
    return info["lfs"]["oid"]


def record(dirs, repos):
    entries = []
    for d in map(pathlib.Path, dirs):
        for gguf in sorted(d.glob("*.gguf")):
            meta = d / ".cache/huggingface/download" / (gguf.name + ".metadata")
            revision = meta.read_text().splitlines()[0].strip()
            repo = repos[d.name]
            digest = sha256(gguf)
            if digest != published_sha(repo, revision, gguf.name):
                raise SystemExit(f"{gguf}: SHA-256 differs from the published LFS hash")
            entries.append({"name": f"{d.name}/{gguf.name}", "repo": repo, "file": gguf.name,
                            "revision": revision, "sha256": digest, "size": gguf.stat().st_size})
    MANIFEST.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"wrote {len(entries)} files to {MANIFEST}")


def fetch(dest):
    for e in json.loads(MANIFEST.read_text()):
        target = pathlib.Path(dest) / e["name"]
        if not target.exists():
            subprocess.run(["hf", "download", e["repo"], e["file"], "--revision", e["revision"],
                            "--local-dir", str(target.parent)], check=True)
    verify(dest)


def verify(dest):
    bad = [e["name"] for e in json.loads(MANIFEST.read_text())
           if sha256(pathlib.Path(dest) / e["name"]) != e["sha256"]]
    print("all files match models.json" if not bad else f"MISMATCH: {bad}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "record":
        # Folder name -> repo; the folders are the --local-dir targets used for the downloads.
        repos = json.loads((HERE / "model-repos.json").read_text())
        record(rest, repos)
    elif cmd == "fetch":
        fetch(rest[0])
    elif cmd == "verify":
        verify(rest[0])
