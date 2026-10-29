"""Back-office executor: dispatch/cancel against the Hermes gateway
(worker path), plus a deterministic stub for benches.

Semantics carried from the production DelegationBridge:
- dispatch POSTs the worker template to {WORKER_BASE_URL}/chat/completions
  streaming, with the gateway key as Bearer.
- Phase boundary (Part G.2): any failure before the first streamed byte
  (transport error, HTTP >= 400 on submit, auth, schema) is phase='submit'
  -> DISPATCH_FAILED handling in the Arbiter. A failure after streaming
  began is phase='body' -> FAILED.
- The worker's reply must end with a "SPOKEN SUMMARY:" line; the summary
  is extracted here, typed facts too (screen/tagger).
- cancel(): the gateway has no cancel RPC; cancelling aborts the HTTP
  stream task. Reply mapping: CANCELLED (stream aborted), COMPLETED
  (result already fully received), PARTIAL is never claimed (completed
  steps are not observable on this worker path).
- capability_scope enforcement (Part E): the dispatch payload carries
  the job's task_class as capability_scope; READ_ONLY jobs get an
  explicit read-only instruction the worker must honor, and the stub
  refuses write-tool attempts under READ_ONLY scope (contract test).
"""
import asyncio
import json
import os
import re
import time

import aiohttp

from .tagger import extract_typed_values

DISPATCH_TEMPLATE = """You are the back office for a voice assistant. A user asked the assistant to handle the task below. Do the work now with your tools if needed.

TASK: {task}

Rules:
- Actually perform the task; do not merely describe how you would.
- Never claim something is done unless you did it.
- Your reply is processed by the system, not spoken directly. End your reply with exactly one final line in this format:
SPOKEN SUMMARY: <one short plain sentence the assistant can say out loud, max 25 words>
"""

READ_ONLY_SUFFIX = """

CAPABILITY SCOPE: READ_ONLY. You may read, search, and report only. Do NOT send, write, modify, delete, book, pay, or change anything. If the task requires a write action, stop and say so in the summary."""


def _summary_of(full):
    m = None
    for m in re.finditer(r"^SPOKEN SUMMARY:\s*(.+)$", full, re.M):
        pass
    if m:
        return m.group(1).strip()
    tail = full.strip().splitlines()
    return (tail[-1].strip()[:200] if tail else "Done.")


class GatewayExecutor:
    def __init__(self, base_url, model, api_key, emit):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.emit = emit
        self._tasks = {}  # job_id -> asyncio.Task
        self._results = {}  # job_id -> full text (for COMPLETED race)
        self._cb = None

    def bind(self, on_started, on_done, on_error):
        self._cb = (on_started, on_done, on_error)

    async def preflight(self):
        """B.7.3: verify the gateway credential before any DELEGATE."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                        self.base_url + "/models",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def dispatch(self, job_row):
        jid = job_row["job_id"]
        spec = json.loads(job_row["task_spec"] or "{}")
        task_text = spec.get("text", "")
        prompt = DISPATCH_TEMPLATE.format(task=task_text)
        if job_row["task_class"] == "READ_ONLY":
            prompt += READ_ONLY_SUFFIX
        task = asyncio.current_task()
        self._tasks[jid] = task
        started = False
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                        self.base_url + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": self.model, "stream": True,
                              "messages": [{"role": "user",
                                            "content": prompt}]},
                        timeout=aiohttp.ClientTimeout(total=None)) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:200]
                        retryable = resp.status in (429,) or resp.status >= 500
                        await self._cb[2](jid, RuntimeError(
                            f"gateway HTTP {resp.status}: {body}"),
                            "submit", http_status=resp.status,
                            retryable=retryable)
                        return
                    buf = []
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
                        delta = ev["choices"][0].get("delta", {})
                        piece = delta.get("content") or ""
                        if piece and not started:
                            started = True
                            await self._cb[0](jid)
                        buf.append(piece)
                    full = "".join(buf)
                    self._results[jid] = full
                    summary = _summary_of(full)
                    facts = extract_typed_values(full)
                    await self._cb[1](jid, summary, facts, full)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            phase = "body" if started else "submit"
            retryable = isinstance(
                e, (aiohttp.ClientConnectionError, asyncio.TimeoutError))
            await self._cb[2](jid, e, phase, http_status=None,
                              retryable=retryable)
        finally:
            self._tasks.pop(jid, None)

    async def cancel(self, job_id):
        if job_id in self._results:
            return "COMPLETED"
        task = self._tasks.get(job_id)
        if task is None:
            return "CANCELLED"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return "CANCELLED"

    def resume(self, job_id, answer):
        self.emit("executor_resume_unsupported", job_id=job_id)


class StubExecutor:
    """Bench/test executor: deterministic canary results, configurable
    delay; models the started/done/error callbacks and cancel."""

    def __init__(self, emit, delay_s=0.3, fail_first_n=0,
                 canary_facts=None):
        self.emit = emit
        self.delay_s = delay_s
        self.fail_first_n = fail_first_n
        self.canary_facts = canary_facts or []
        self._tasks = {}
        self._results = {}
        self._cb = None
        self.dispatch_count = 0
        self.write_attempts_refused = 0

    def bind(self, on_started, on_done, on_error):
        self._cb = (on_started, on_done, on_error)

    async def preflight(self):
        return True

    async def dispatch(self, job_row):
        jid = job_row["job_id"]
        self.dispatch_count += 1
        self._tasks[jid] = asyncio.current_task()
        spec = json.loads(job_row["task_spec"] or "{}")
        try:
            if self.dispatch_count <= self.fail_first_n:
                await self._cb[2](jid, RuntimeError("stub submit failure"),
                                  "submit", http_status=503, retryable=True)
                return
            delay = spec.get("stub_delay_s", self.delay_s)
            await asyncio.sleep(min(0.05, delay))
            await self._cb[0](jid)
            await asyncio.sleep(max(0.0, delay - 0.05))
            facts = [{"type": "IDENT", "value": f"CANARY-{jid[:6]}"}]
            facts += self.canary_facts
            summary = spec.get("stub_summary",
                               f"Finished {spec.get('obj', 'the task')}.")
            self._results[jid] = summary
            await self._cb[1](jid, summary, facts, summary)
        except asyncio.CancelledError:
            raise
        finally:
            self._tasks.pop(jid, None)

    async def cancel(self, job_id):
        if job_id in self._results:
            return "COMPLETED"
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return "CANCELLED"

    def resume(self, job_id, answer):
        pass

    def attempt_write(self, job_row):
        """Scope-enforcement contract hook: a write-tool attempt under a
        READ_ONLY scope is refused (Part E)."""
        if job_row["task_class"] == "READ_ONLY":
            self.write_attempts_refused += 1
            return False
        return True
