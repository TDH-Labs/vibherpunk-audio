"""Monotonic event spine (v2 §2.2 v1, step 1 of the build order).

Every user turn is appended to a durable, append-only session event log in
real-arrival order. seq is assigned at the router tap from a monotonic
counter; all timestamps come from time.monotonic_ns() / time.monotonic().
The log is the sequencing authority; JobRegistry rows reference (seq, turn).

Durability: plain append + fsync per event is affordable (one write per
user turn, not per registry mutation) and keeps the spine crash-exact.
"""
import json
import os
import threading
import time


class EventSpine:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._turn = 0
        self._fh = open(path, "a", buffering=1)  # line buffered
        # Recover counters from an existing log (restart within a process
        # tree reuses the same session dir in benches).
        try:
            with open(path, "r") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    self._seq = max(self._seq, int(e.get("seq", 0)))
                    self._turn = max(self._turn, int(e.get("turn", 0)))
        except FileNotFoundError:
            pass

    def next_seq(self):
        with self._lock:
            self._seq += 1
            return self._seq

    def next_turn(self):
        with self._lock:
            self._turn += 1
            return self._turn

    @property
    def turn(self):
        return self._turn

    @property
    def seq(self):
        return self._seq

    def emit(self, kind, **fields):
        """Append an event. Returns the event dict."""
        ev = {
            "kind": kind,
            "seq": fields.pop("seq", None),
            "turn": fields.pop("turn", self._turn),
            "t_mono": time.monotonic(),
            "t_ns": time.monotonic_ns(),
            "t_wall": time.time(),
        }
        ev.update(fields)
        line = json.dumps(ev, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        return ev

    def close(self):
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
