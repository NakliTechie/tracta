"""A minimal client for an OpenAI-compatible chat completions endpoint (standard library only)."""
import json
import os
import time
import urllib.error
import urllib.request


class EndpointError(RuntimeError):
    pass


class Endpoint:
    def __init__(self, url, model, key_env=None, extra_body=None, rps=2.0, timeout=120):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = os.environ.get(key_env) if key_env else None
        if key_env and not self.key:
            raise EndpointError(f"environment variable {key_env} is not set")
        self.extra_body = extra_body or {}
        self.gap = 1.0 / rps if rps > 0 else 0.0
        self.timeout = timeout
        self._last = 0.0

    def chat(self, prompt, max_tokens=16, top_logprobs=0):
        """Greedy answer to one user message. Returns (text, {token: logprob} or None)."""
        body = dict(self.extra_body, model=self.model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
        if top_logprobs:
            body.update(logprobs=True, top_logprobs=top_logprobs)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        for attempt in range(4):
            wait = self._last + self.gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                # Retry rate limits and server errors with backoff; anything else is the caller's problem.
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise EndpointError(f"HTTP {e.code} from {self.url}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 3:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise EndpointError(f"cannot reach {self.url}: {e}") from None
        choice = d["choices"][0]
        text = choice["message"].get("content") or ""
        top = None
        if top_logprobs:
            content = (choice.get("logprobs") or {}).get("content") or []
            if content:
                top = {t["token"]: t["logprob"] for t in content[0].get("top_logprobs", [])}
        return text, top
