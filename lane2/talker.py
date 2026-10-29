"""Raw-text talker client (v2 Part A §3.4-§3.7).

Transport is raw text on POST /v1/completions against the dedicated MLX
server. Chat-template endpoints and token-ID prompt payloads are
forbidden on this path. Streaming with usage reporting gives the
per-turn cached_tokens the §3.7 shortfall alarm consumes.

Boot: hash pins (tokenizer files, mlx-lm version) refuse start on
mismatch; the golden-prompt check (§3.6.3) verifies byte-stable caching
before the lane reports ready; two consecutive failures move the lane to
the in-process mlx_lm fallback (integration owns that switch).
"""
import asyncio
import hashlib
import json
import time

import aiohttp

MLX_LM_PIN = "0.32.2"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pins(model_dir, pins):
    """pins: {tokenizer.json: sha, tokenizer_config.json: sha,
    mlx_lm: version}. Returns list of mismatch strings (empty = ok)."""
    bad = []
    import os
    for name in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(model_dir, name)
        want = pins.get(name)
        if want is None:
            bad.append(f"pin missing for {name}")
            continue
        try:
            got = sha256_file(p)
        except FileNotFoundError:
            bad.append(f"{name} not found in {model_dir}")
            continue
        if got != want:
            bad.append(f"{name} sha mismatch")
    try:
        import mlx_lm
        ver = getattr(mlx_lm, "__version__", None)
        if ver is None:
            import importlib.metadata as md
            ver = md.version("mlx-lm")
        if ver != MLX_LM_PIN:
            bad.append(f"mlx_lm version {ver} != pin {MLX_LM_PIN}")
    except Exception as e:
        bad.append(f"mlx_lm version check failed: {e}")
    return bad


class TalkerError(Exception):
    pass


class TalkerClient:
    def __init__(self, base_url, model, emit=None, timeout_s=30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.emit = emit or (lambda k, **f: None)
        self.timeout_s = timeout_s
        self._session = None

    def _sess(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def stream(self, prompt_text, *, max_tokens=220, temperature=0.7,
                     top_p=0.8, seed=None):
        """Yields (delta_text, None) per token chunk and a final
        (None, usage_dict). Raises TalkerError on transport/HTTP failure
        before the first token; mid-stream failures raise after tokens."""
        payload = {
            "model": self.model, "prompt": prompt_text,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True},
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }
        if seed is not None:
            payload["seed"] = seed
        t0 = time.monotonic()
        first = True
        usage = None
        try:
            async with self._sess().post(
                    self.base_url + "/completions", json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=None, sock_read=self.timeout_s)) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    raise TalkerError(f"talker HTTP {resp.status}: {body}")
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get("usage"):
                        usage = ev["usage"]
                    choices = ev.get("choices") or []
                    if choices:
                        txt = choices[0].get("text") or ""
                        if txt:
                            if first:
                                first = False
                                self.emit(
                                    "talker_ttft",
                                    ms=round((time.monotonic() - t0) * 1000, 1))
                            yield txt, None
        except TalkerError:
            raise
        except Exception as e:
            raise TalkerError(f"{type(e).__name__}: {e}") from e
        yield None, (usage or {})

    async def complete(self, prompt_text, *, max_tokens=1, temperature=0.0):
        """Non-streaming one-shot (warm requests, golden check)."""
        payload = {
            "model": self.model, "prompt": prompt_text,
            "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, "stop": ["<|im_end|>", "<|endoftext|>"],
        }
        t0 = time.monotonic()
        async with self._sess().post(
                self.base_url + "/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as resp:
            if resp.status != 200:
                raise TalkerError(f"talker HTTP {resp.status}")
            body = await resp.json()
        dt = time.monotonic() - t0
        text = ""
        choices = body.get("choices") or []
        if choices:
            text = choices[0].get("text") or ""
        return {"text": text, "usage": body.get("usage") or {},
                "latency_s": dt}

    @staticmethod
    def cached_tokens(usage):
        if not usage:
            return None
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            return details["cached_tokens"]
        return usage.get("cached_tokens")

    async def golden_check(self, golden):
        """§3.6.3: (i) prompt token count equals the golden record,
        (ii) second identical request reports cached_tokens >= golden
        cacheable length and completes <= 0.20 s at max_tokens=1,
        (iii) greedy 8-token output byte-equals the golden output."""
        prompt = golden["prompt"]
        want_tokens = golden["prompt_tokens"]
        want_cacheable = golden["cacheable_tokens"]
        want_output = golden["output_8tok"]
        r1 = await self.complete(prompt, max_tokens=8, temperature=0.0)
        got_tokens = (r1["usage"] or {}).get("prompt_tokens")
        if got_tokens != want_tokens:
            return False, f"token count {got_tokens} != golden {want_tokens}"
        if r1["text"] != want_output:
            return False, "greedy output differs from golden record"
        r2 = await self.complete(prompt, max_tokens=1, temperature=0.0)
        cached = self.cached_tokens(r2["usage"])
        if cached is None or cached < want_cacheable:
            return False, f"cached {cached} < cacheable {want_cacheable}"
        if r2["latency_s"] > 0.20:
            return False, f"warm latency {r2['latency_s']:.3f}s > 0.20s"
        return True, "ok"
