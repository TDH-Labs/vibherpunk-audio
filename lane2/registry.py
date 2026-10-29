"""JobRegistry — v2 §2.2 buffer with in-memory mirror (Part I.2 ruling).

Shape (I.2.2): Arbiter (single writer) -> write-ahead buffer (in-memory
FIFO, cap 4096) -> flusher thread -> SQLite (batched group commit) ->
in-memory mirror. The mirror contains exactly the durable state; reads
never touch SQLite on the hot path. Durability precedes acknowledgement
(publish-then-ack by construction: futures resolve only after the batch
that contains them has committed AND been published to the mirror).

Commit ordering (H1.40): a transition's future resolves only after its
journal row and the job row are durable in the same transaction
(F_FULLFSYNC via PRAGMA fullfsync + synchronous=FULL under WAL).

Idempotency: journal.write_id is the primary key; a retried write_id is
applied at most once (I.2.3 step 5). Recovery (I.2.6): on open, every job
row's version must equal MAX(journal.new_version) for that job, else the
process refuses to serve ("ABORTED_UNDURABLE").
"""
import asyncio
import json
import os
import secrets
import sqlite3
import threading
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(now_ms=None):
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    rand = secrets.randbits(80)
    val = (now_ms << 80) | rand
    out = []
    for i in range(26):
        out.append(_CROCKFORD[(val >> (5 * (25 - i))) & 31])
    return "".join(out)


def write_id():
    return "w_" + ulid()


NONTERMINAL = {
    "QUEUED", "BAND_HELD", "DRAFT", "DISPATCHING", "RUNNING",
    "NEEDS_INPUT", "CANCEL_REQUESTED", "HELD_DUPLICATE",
}
TERMINAL = {
    "DONE", "FAILED", "DISPATCH_FAILED", "CANCELLED", "MERGED",
    "DROPPED_SUBTHRESHOLD", "CONFIRM_EXPIRED", "CONFIRM_REJECTED",
    "SUPERSEDED", "SCOPE_KILLED", "ABORTED_UNDURABLE",
}
ALL_STATES = NONTERMINAL | TERMINAL

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA fullfsync=ON;
PRAGMA journal_size_limit=67108864;
CREATE TABLE IF NOT EXISTS jobs(
  job_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '', task_spec TEXT NOT NULL DEFAULT '{}',
  provenance TEXT NOT NULL DEFAULT 'RULE', p_delegate REAL,
  task_class TEXT NOT NULL DEFAULT 'SIDE_EFFECTING',
  origin_turn INTEGER NOT NULL DEFAULT 0, origin_seq INTEGER NOT NULL DEFAULT 0,
  entity_set TEXT NOT NULL DEFAULT '[]', spec_norm TEXT NOT NULL DEFAULT '',
  idempotency_key TEXT, parent_id TEXT, retry_of TEXT,
  version INTEGER NOT NULL, created_mono REAL NOT NULL, created_wall REAL NOT NULL,
  updated_mono REAL NOT NULL, updated_turn INTEGER NOT NULL DEFAULT 0,
  result_summary TEXT, result_facts TEXT, headline_screened TEXT,
  fail_reason TEXT, failure_code TEXT, eta_hint TEXT,
  announced INTEGER NOT NULL DEFAULT 0, announced_mono REAL,
  delivered_mono REAL, terminal_mono REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_session_status
  ON jobs(session_id, status, updated_mono DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_nonterminal
  ON jobs(session_id) WHERE status IN
  ('QUEUED','BAND_HELD','DRAFT','DISPATCHING','RUNNING','NEEDS_INPUT',
   'CANCEL_REQUESTED','HELD_DUPLICATE');
CREATE TABLE IF NOT EXISTS journal(
  write_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
  prior_version INTEGER NOT NULL, new_version INTEGER NOT NULL,
  turn_id INTEGER NOT NULL, op TEXT NOT NULL, delta TEXT NOT NULL DEFAULT '{}',
  ts_wall_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_journal_job ON journal(job_id, new_version);
"""

_JOB_COLS = [
    "job_id", "session_id", "status", "title", "task_spec", "provenance",
    "p_delegate", "task_class", "origin_turn", "origin_seq", "entity_set",
    "spec_norm", "idempotency_key", "parent_id", "retry_of", "version",
    "created_mono", "created_wall", "updated_mono", "updated_turn",
    "result_summary", "result_facts", "headline_screened", "fail_reason",
    "failure_code", "eta_hint", "announced", "announced_mono",
    "delivered_mono", "terminal_mono",
]


class BufferFullError(Exception):
    pass


class DurableWriteTimeout(Exception):
    pass


class RegistryFault(Exception):
    pass


class _Entry:
    __slots__ = ("write_id", "job_id", "from_v", "to_v", "row", "turn_id",
                 "op", "delta", "ack_required", "future", "loop",
                 "enqueued_mono")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class JobRegistry:
    """Single-writer registry. Async write API is used from the Arbiter's
    event loop; mirror reads are synchronous and lock-guarded."""

    def __init__(self, db_path, emit=None, batch_window_ms=8.0,
                 batch_max=32, buffer_cap=4096):
        self.db_path = db_path
        self.emit = emit or (lambda kind, **kw: None)
        self.batch_window = batch_window_ms / 1000.0
        self.batch_max = batch_max
        self.buffer_cap = buffer_cap
        self._buf = []  # list[_Entry], FIFO
        self._cond = threading.Condition()
        self._mirror = {}  # job_id -> dict row
        self._registry_version = 0
        self._durable_ids = set()  # write_ids committed to the journal
        self._pending_ids = set()  # write_ids currently in the buffer
        self._stop = False
        self._degraded = False
        self._last_commit_mono = None
        self.stats = {"commits": 0, "writes": 0, "batches": 0,
                      "commit_ms": [], "ack_ms": []}
        self._open_store()
        self._flusher = threading.Thread(target=self._flush_loop,
                                         name="registry-flusher", daemon=True)
        self._flusher.start()

    # ------------------------------------------------------------ store
    def _open_store(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.executescript(DDL)
        con.commit()
        # Recovery verification (I.2.6): row version == max journal version.
        bad = con.execute(
            "SELECT j.job_id FROM jobs j LEFT JOIN journal jr ON jr.job_id=j.job_id "
            "GROUP BY j.job_id HAVING j.version != COALESCE(MAX(jr.new_version), -1)"
        ).fetchall()
        if bad:
            con.close()
            raise RegistryFault(
                f"recovery verification failed for {len(bad)} jobs: {bad[:3]}")
        rows = con.execute(
            "SELECT " + ",".join(_JOB_COLS) + " FROM jobs").fetchall()
        for r in rows:
            row = dict(zip(_JOB_COLS, r))
            self._mirror[row["job_id"]] = row
        wids = con.execute("SELECT write_id FROM journal").fetchall()
        self._durable_ids = {w[0] for w in wids}
        self._con = con

    # ------------------------------------------------------------ writes
    def _enqueue(self, entry):
        with self._cond:
            if len(self._buf) >= self.buffer_cap:
                raise BufferFullError(f"registry buffer at cap {self.buffer_cap}")
            if entry.write_id in self._durable_ids:
                # Idempotent retry of an already-durable write: ack succeeds
                # immediately, nothing re-applied (I.2.3 step 5).
                if entry.future is not None and entry.loop is not None:
                    self._resolve(entry, None)
                return False
            if entry.write_id in self._pending_ids:
                raise ValueError(f"write_id {entry.write_id} already pending")
            self._buf.append(entry)
            self._pending_ids.add(entry.write_id)
            self._cond.notify()
            return True

    async def _submit(self, entry):
        self._enqueue(entry)
        if entry.ack_required:
            await entry.future
        return entry

    def _new_future(self):
        loop = asyncio.get_running_loop()
        return loop, loop.create_future()

    async def create_job(self, *, session_id, status, title="", task_spec=None,
                         provenance="RULE", p_delegate=None,
                         task_class="SIDE_EFFECTING", origin_turn=0,
                         origin_seq=0, entity_set=None, spec_norm="",
                         idempotency_key=None, parent_id=None, retry_of=None,
                         ack_required=True, wid=None):
        if status not in ALL_STATES:
            raise ValueError(f"bad status {status}")
        jid = ulid()
        now_m = time.monotonic()
        row = {
            "job_id": jid, "session_id": session_id, "status": status,
            "title": title, "task_spec": json.dumps(task_spec or {}),
            "provenance": provenance, "p_delegate": p_delegate,
            "task_class": task_class, "origin_turn": origin_turn,
            "origin_seq": origin_seq,
            "entity_set": json.dumps(entity_set or []),
            "spec_norm": spec_norm, "idempotency_key": idempotency_key,
            "parent_id": parent_id, "retry_of": retry_of, "version": 1,
            "created_mono": now_m, "created_wall": time.time(),
            "updated_mono": now_m, "updated_turn": origin_turn,
            "result_summary": None, "result_facts": None,
            "headline_screened": None, "fail_reason": None,
            "failure_code": None, "eta_hint": None, "announced": 0,
            "announced_mono": None, "delivered_mono": None,
            "terminal_mono": now_m if status in TERMINAL else None,
        }
        loop, fut = self._new_future() if ack_required else (None, None)
        entry = _Entry(write_id=wid or write_id(), job_id=jid, from_v=0,
                       to_v=1, row=row, turn_id=origin_turn, op="CREATE",
                       delta={"status": status}, ack_required=ack_required,
                       future=fut, loop=loop, enqueued_mono=now_m)
        await self._submit(entry)
        return jid

    async def transition(self, job_id, to_status, *, ack_required=False,
                         turn_id=0, wid=None, **fields):
        if to_status not in ALL_STATES:
            raise ValueError(f"bad status {to_status}")
        with self._cond:
            cur = self._pending_or_mirror(job_id)
        if cur is None:
            raise KeyError(f"unknown job {job_id}")
        now_m = time.monotonic()
        row = dict(cur)
        row["status"] = to_status
        row["version"] = int(cur["version"]) + 1
        row["updated_mono"] = now_m
        row["updated_turn"] = turn_id
        if to_status in TERMINAL and row.get("terminal_mono") is None:
            row["terminal_mono"] = now_m
        for k, v in fields.items():
            if k not in _JOB_COLS:
                raise ValueError(f"unknown job field {k}")
            row[k] = json.dumps(v) if k in ("task_spec", "entity_set",
                                            "result_facts") and not isinstance(v, str) else v
        loop, fut = self._new_future() if ack_required else (None, None)
        entry = _Entry(write_id=wid or write_id(), job_id=job_id,
                       from_v=int(cur["version"]), to_v=row["version"],
                       row=row, turn_id=turn_id, op="TRANSITION",
                       delta={"status": to_status, **{
                           k: fields[k] for k in fields}},
                       ack_required=ack_required, future=fut, loop=loop,
                       enqueued_mono=now_m)
        await self._submit(entry)
        return row["version"]

    def _pending_or_mirror(self, job_id):
        """Latest known row: newest buffered entry wins over the mirror."""
        for e in reversed(self._buf):
            if e.job_id == job_id:
                return e.row
        return self._mirror.get(job_id)

    # ------------------------------------------------------------ mirror reads
    def get(self, job_id):
        with self._cond:
            row = self._pending_or_mirror(job_id)
            return dict(row) if row else None

    def snapshot(self):
        """(registry_version, [rows]) — the only read speech planning uses."""
        with self._cond:
            rows = {}
            for jid, r in self._mirror.items():
                rows[jid] = r
            for e in self._buf:
                rows[e.job_id] = e.row
            return self._registry_version, [dict(r) for r in rows.values()]

    def mirror_version(self):
        with self._cond:
            return self._registry_version

    def nonterminal(self, session_id):
        _, rows = self.snapshot()
        return [r for r in rows
                if r["session_id"] == session_id and r["status"] in NONTERMINAL]

    def terminal_recent(self, session_id, within_s, now_mono=None):
        now_mono = now_mono or time.monotonic()
        _, rows = self.snapshot()
        out = []
        for r in rows:
            if r["session_id"] != session_id or r["status"] not in TERMINAL:
                continue
            tm = r.get("terminal_mono")
            if tm is not None and now_mono - tm <= within_s:
                out.append(r)
        return out

    def held_unannounced(self, session_id):
        _, rows = self.snapshot()
        out = [r for r in rows if r["session_id"] == session_id
               and r["status"] in TERMINAL and not r["announced"]
               and r["status"] not in ("DROPPED_SUBTHRESHOLD", "MERGED",
                                       "SUPERSEDED")]
        out.sort(key=lambda r: r.get("terminal_mono") or 0)
        return out

    def buffer_depth(self):
        with self._cond:
            return len(self._buf)

    def degraded(self):
        return self._degraded

    # ------------------------------------------------------------ flusher
    def _resolve(self, entry, exc):
        if entry.future is None or entry.loop is None:
            return

        def _do():
            if entry.future.done():
                return
            if exc is None:
                entry.future.set_result(True)
            else:
                entry.future.set_exception(exc)

        try:
            entry.loop.call_soon_threadsafe(_do)
        except RuntimeError:
            pass

    def _flush_loop(self):
        backoff = 0.005
        while not self._stop:
            with self._cond:
                while not self._buf and not self._stop:
                    self._cond.wait(timeout=0.25)
                    self._check_ack_timeouts_locked()
                if self._stop:
                    return
                # Batch: wait until window elapses or batch fills.
                deadline = self._buf[0].enqueued_mono + self.batch_window
                while len(self._buf) < self.batch_max and not self._stop:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        break
                    self._cond.wait(timeout=remain)
                batch = self._buf[: self.batch_max]
                del self._buf[: len(batch)]
            try:
                t0 = time.monotonic()
                self._commit_batch(batch)
                dt = time.monotonic() - t0
                self.stats["commits"] += 1
                self.stats["batches"] += 1
                self.stats["writes"] += len(batch)
                self.stats["commit_ms"].append(dt * 1000.0)
                backoff = 0.005
            except Exception as e:  # durability failure: requeue, back off
                with self._cond:
                    self._buf = batch + self._buf
                self.emit("registry_commit_error", error=f"{type(e).__name__}: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 4, 0.2)

    def _check_ack_timeouts_locked(self):
        now = time.monotonic()
        for e in list(self._buf):
            if e.ack_required and now - e.enqueued_mono > 2.0:
                self._buf.remove(e)
                self._pending_ids.discard(e.write_id)
                self._degraded = True
                self.emit("registry_ack_timeout", write_id=e.write_id,
                          job_id=e.job_id, age_ms=(now - e.enqueued_mono) * 1000)
                self._resolve(e, DurableWriteTimeout(
                    f"write {e.write_id} not durable in 2000 ms"))

    def _commit_batch(self, batch):
        con = self._con
        t0 = time.monotonic()
        con.execute("BEGIN IMMEDIATE")
        try:
            for e in batch:
                r = e.row
                con.execute(
                    "INSERT INTO jobs(" + ",".join(_JOB_COLS) + ") VALUES(" +
                    ",".join("?" * len(_JOB_COLS)) + ") "
                    "ON CONFLICT(job_id) DO UPDATE SET " +
                    ",".join(f"{c}=excluded.{c}" for c in _JOB_COLS if c != "job_id"),
                    [r[c] for c in _JOB_COLS],
                )
                con.execute(
                    "INSERT OR IGNORE INTO journal(write_id, job_id, "
                    "prior_version, new_version, turn_id, op, delta, ts_wall_ms) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (e.write_id, e.job_id, e.from_v, e.to_v, e.turn_id, e.op,
                     json.dumps(e.delta, default=str), int(time.time() * 1000)),
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        # Publish to mirror BEFORE resolving acks (publish-then-ack).
        with self._cond:
            for e in batch:
                self._mirror[e.job_id] = e.row
                self._durable_ids.add(e.write_id)
                self._pending_ids.discard(e.write_id)
            self._registry_version += 1
            self._last_commit_mono = time.monotonic()
        for e in batch:
            self.stats["ack_ms"].append(
                (time.monotonic() - e.enqueued_mono) * 1000.0)
            if e.ack_required:
                self._resolve(e, None)

    async def flush_barrier(self, timeout=10.0):
        """Resolve when every buffered write is durable (tests/shutdown)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.buffer_depth() == 0:
                return True
            await asyncio.sleep(0.005)
        return False

    # ------------------------------------------------------------ archival reads (bench)
    def point_read(self, job_id):
        cur = self._con.execute(
            "SELECT " + ",".join(_JOB_COLS) + " FROM jobs WHERE job_id=?",
            (job_id,))
        r = cur.fetchone()
        return dict(zip(_JOB_COLS, r)) if r else None

    def count_jobs(self):
        return self._con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._flusher.join(timeout=2.0)
        try:
            self._con.close()
        except Exception:
            pass
