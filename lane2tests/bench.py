"""Tier 1 / Tier 2 bench (v2 Part B) — drives the real lane2 engine
headless: real RouterDecider + encoder, real MLX talker, real Pocket
TTS synthesis for playback durations, StubExecutor (Tier 1) or the
real Hermes gateway (Tier 2). Recorded-corpus turns (B.9 blocks A–F)
are scripted from the rounds' real case material.

Usage:
  .venv/bin/python -m lane2tests.bench tier1 <run_idx> <out.json>
  .venv/bin/python -m lane2tests.bench tier2 <out.json>
Exactly the B.9.5 fields are reported per run. Tier 1 gates are
Part B.7 (lane-path); Tier 2 compares against the frozen production
baseline factors (B.8) on the reduced corpus.
"""
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane2.context import SessionLog, Tokenizer          # noqa: E402
from lane2.embed import HashEmbedder                    # noqa: E402
from lane2.engine import LaneEngine                      # noqa: E402
from lane2.executor import GatewayExecutor, StubExecutor  # noqa: E402
from lane2.facts import FactExtractor, FactLedger, MemoState  # noqa: E402
from lane2.guard import TopicGuard                       # noqa: E402
from lane2.registry import JobRegistry                   # noqa: E402
from lane2.screen import Screen                          # noqa: E402
from lane2.speech import BenchSpeaker, SpeechSequencer   # noqa: E402
from lane2.talker import TalkerClient                    # noqa: E402
import hermes_voice as hv                                # noqa: E402

PINS = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lane2", "pins.json")))
TALKER_URL = os.environ.get("LANE2_BASE", "http://127.0.0.1:8123/v1")
TALKER_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

_synth = {"fn": None, "tried": False}


def pocket_synth():
    """Real Pocket TTS durations (seconds of audio per text)."""
    if _synth["tried"]:
        return _synth["fn"]
    _synth["tried"] = True
    try:
        from pocket_tts import TTSModel
        model = TTSModel.load_model()
        state = model.get_state_for_audio_prompt("azelma")

        def fn(text):
            audio = model.generate_audio(state, text)
            return float(audio.shape[-1]) / 24000.0

        _synth["fn"] = fn
    except Exception as e:
        print(f"POCKET_SYNTH FALLBACK (word rate): {e}", file=sys.stderr)
        _synth["fn"] = None
    return _synth["fn"]


class BenchDecider:
    """The production RouterDecider over the production encoder."""

    def __init__(self):
        self.inner = hv.RouterDecider(hv.INTENT_CLASSIFIER)

    async def decide(self, text, jobs_pending=0):
        return await self.inner.decide(text, jobs_pending=jobs_pending)


class Ctx:
    def __init__(self, tier):
        self.tier = tier
        self.tok = Tokenizer(PINS["model_dir"])
        self.extractor = FactExtractor()
        self.extractor.load_encoder()
        self.talker = TalkerClient(TALKER_URL, TALKER_MODEL,
                                   emit=lambda k, **f: None)
        self.decider = BenchDecider()
        self.synth = pocket_synth()
        self.events = []
        self.limit = {}          # block name -> scenario cap (tier2/mini)
        self.drain_timeout = 12.0

    def emit(self, _kind, **f):
        self.events.append((_kind, f))


def _n(ctx, block, default):
    return min(default, ctx.limit.get(block, default))


def make_engine(ctx, session_id, db, executor):
    reg = JobRegistry(db, emit=ctx.emit)
    ledger = FactLedger(ctx.extractor, emit=ctx.emit)
    memo = MemoState(reg, ledger, session_id=session_id, emit=ctx.emit)
    memo.bind_tokenizer(ctx.tok)
    log = SessionLog(ctx.tok, {"user": "User"}, memo=memo, emit=ctx.emit)
    log.retain_hook = lambda u: any(
        (r := reg.get(j)) is not None and r["status"] not in (
            "DONE", "FAILED", "CANCELLED", "DISPATCH_FAILED", "MERGED")
        for j in u.job_refs)
    screen = Screen(emit=ctx.emit, snapshot_fn=lambda: (0, [], []))
    import time as _time
    guard = TopicGuard(emit=ctx.emit, embedder=HashEmbedder(),
                       clock=_time.monotonic)
    eng = LaneEngine(session_id=session_id, registry=reg, context=log,
                     ledger=ledger, memo=memo, talker=ctx.talker,
                     screen=screen, guard=guard, decider=ctx.decider,
                     emit=ctx.emit, executor=executor,
                     lookup_fn=hv._lane2_lookup)
    screen.snapshot_fn = eng._screen_snapshot
    return eng


def seq_factory(eng):
    def make():
        return SpeechSequencer(screen=eng.screen, guard=eng.guard,
                               talker=eng.talker, emit=eng.emit,
                               session_id=eng.session_id,
                               p_status_fn=hv._lane2_p_status)
    return make



# ------------------------------------------------------------ canonical
# Track B bench-harness integration (INTEGRATION.md): every run now
# also writes a canonical run dir (turns/directives/jobs/segments/
# events JSONL + run_meta.json) that bench_harness.score consumes
# directly. Two additions unlock it: (a) the EventSpine is dumped
# per run instead of being discarded, (b) each turn is stamped with
# an end-of-speech anchor — the headless driver has no real EoS, so
# turn start is the declared proxy (run_meta.eos_anchor).

RECORDER = None  # set by run entry points


class CanonicalRecorder:
    def __init__(self):
        self.t0 = time.monotonic()
        self.spine = []
        self.turns = []
        self.segments = []
        self.events = []
        self.cur_turn = None       # recorder turn id "tN"
        self.turn_counter = 0
        self.utt_counter = 0
        self.prompt_tokens = {}
        self.turn_eos = {}
        self.turn_session = {}
        self.turn_intent = {}

    def ms(self, mono):
        return (mono - self.t0) * 1000.0

    def _rebase(self):
        """One monotonic clock per run (schema contract): rebase
        turns/segments/events onto the spine origin (min t_mono) —
        the same origin convert_spine uses for job times — so ack
        ordering and latency math compare like with like."""
        if not self.spine:
            return
        origin = min(r["t_mono"] for r in self.spine)
        # recorder-origin ms -> spine-origin ms is a pure shift
        shift = (self.t0 - origin) * 1000.0
        for t in self.turns:
            raw = self.turn_eos.get(t["turn_id"])
            if raw is not None:
                t["eos_ms"] = (raw - origin) * 1000.0
                t["stt_final_ms"] = (raw - origin) * 1000.0
            if t.get("fao_ms") is not None:
                t["fao_ms"] -= shift
        for s in self.segments:
            s["screen_t_ms"] -= shift
            s["play_start_t_ms"] -= shift
            s["play_end_t_ms"] -= shift
        for e in self.events:
            e["t_ms"] -= shift

    def note_event(self, kind, fields):
        f = dict(fields)
        if "kind" in f:
            f["kind_directive"] = f.pop("kind")
        rec = {"kind": kind, "t_mono": time.monotonic(), **f}
        if self.cur_turn is not None:
            rec.setdefault("turn", int(self.cur_turn[1:]))
            rec.setdefault("turn_ctx", self.cur_turn)
            # The spine carries no intent_key (the arbiter keeps it
            # internal); stamp the recorder's per-turn intent so the
            # canonical JobRecord/DirectiveRecord can carry it. One
            # turn = one intent in this bench's scenario design.
            if kind in ("directive_applied", "job_queued"):
                rec.setdefault("intent_key", self.turn_intent.get(
                    self.cur_turn))
        self.spine.append(rec)
        if kind == "prompt_built" and self.cur_turn:
            self.prompt_tokens[self.cur_turn] = f.get("total_tokens")
        if kind == "job_failed":
            self.events.append({
                "kind": "worker_error", "t_ms": self.ms(rec["t_mono"]),
                "turn_id": self.cur_turn,
                "detail": {"job_id": f.get("job_id"),
                           "error": str(f.get("error"))[:120]}})
        if kind == "preflight_red":
            self.events.append({
                "kind": "preflight", "t_ms": self.ms(rec["t_mono"]),
                "turn_id": self.cur_turn,
                "detail": {"why": str(f.get("why"))[:120]}})

    def note_session_event(self, kind, detail=None):
        self.events.append({"kind": kind,
                            "t_ms": self.ms(time.monotonic()),
                            "turn_id": self.cur_turn,
                            "detail": detail or {}})

    def begin_turn(self, session_id, text=""):
        self.turn_counter += 1
        tid = f"t{self.turn_counter}"
        self.cur_turn = tid
        self.turn_session[tid] = session_id
        self.turn_eos[tid] = time.monotonic()  # raw; rebased at write
        norm = " ".join((text or "").lower().split())
        # intent identity is (session, normalized task): the same
        # instruction in another session is a different intent.
        self.turn_intent[tid] = (
            "intent-" + hashlib.sha1(
                f"{session_id}|{norm}".encode()).hexdigest()[:12]
            if norm else None)
        return tid

    def note_segments(self, turn_id, played):
        self.utt_counter += 1
        utt = f"u{self.utt_counter}"
        for i, rec in enumerate(played):
            tid_ = rec.get("template_id") or ""
            if rec["kind"] == "generated":
                role = "answer"
            elif tid_.startswith("T-ACK-HOLD"):
                role = "hold"  # hold acknowledgment, not a dispatch ack
            elif tid_.startswith("T-ACK"):
                role = "ack"
            elif tid_.startswith(("T-STATUS", "T-GUARD")):
                role = "status"
            elif tid_.startswith("T-FB"):
                role = "refusal"
            else:
                role = "announce"
            # stored on the recorder-origin clock here; _rebase()
            # shifts everything onto the spine origin at write().
            self.segments.append({
                "turn_id": turn_id, "seg_id": f"{utt}-{i}",
                "role": role, "kind": rec["kind"],
                "template_id": rec.get("template_id"),
                "text": rec["text"],
                "screen_t_ms": self.ms(rec["screened_at"]),
                "play_start_t_ms": self.ms(rec["play_start"]),
                "play_end_t_ms": self.ms(rec["play_end"]),
                "job_ref": rec.get("job_id"),
                "utterance_id": utt})

    def end_turn(self, turn_id, *, gold, route, played, error=False):
        fao = None
        if played:
            fao = min(self.ms(r["play_start"]) for r in played)
        self.note_segments(turn_id, played)
        self.turns.append({
            "turn_id": turn_id,
            "session_id": self.turn_session.get(turn_id),
            "device": "bench", "gold": gold,
            "route": (route or {}).get("decision"),
            "served_via_draft": False, "warmup": False,
            "excluded": False, "exclude_reason": None,
            "eos_ms": self.turn_eos.get(turn_id),
            "stt_final_ms": self.turn_eos.get(turn_id),
            "fao_ms": fao, "error": error,
            "prompt_tokens": self.prompt_tokens.get(turn_id),
            "provider_ttft_ms": None,
            "registry_commit_ms": None, "wall_start": None})

    def write(self, run_dir):
        self._rebase()
        os.makedirs(run_dir, exist_ok=True)
        spine_path = os.path.join(run_dir, "events_spine.jsonl")
        with open(spine_path, "w") as fh:
            for rec in self.spine:
                fh.write(json.dumps(rec) + "\n")
        directives, jobs = [], []
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.abspath(__file__)))
            from bench_harness import adapters
            # events_spine.jsonl stays byte-faithful to the lane's
            # spine (lowercase directive kinds); the canonical
            # schema's DirectiveRecord kinds are uppercase, so the
            # converter input is a normalized copy (kept alongside).
            norm_path = os.path.join(run_dir,
                                     "events_spine_canonical.jsonl")
            with open(norm_path, "w") as fh:
                for rec in self.spine:
                    r2 = dict(rec)
                    if "kind_directive" in r2:
                        r2["kind_directive"] = str(
                            r2["kind_directive"]).upper()
                    # Turn namespace unification: the spine carries
                    # per-engine turn numbers; turns/segments use
                    # recorder-global ids. The canonical copy speaks
                    # the recorder namespace throughout (the faithful
                    # events_spine.jsonl keeps the engine numbers).
                    if r2.get("turn_ctx"):
                        r2["turn"] = int(r2["turn_ctx"][1:])
                    fh.write(json.dumps(r2) + "\n")
            conv = adapters.convert_spine(norm_path)
            directives, jobs = conv["directives"], conv["jobs"]
        except Exception as e:
            print(f"canonical convert failed: {e}", file=sys.stderr)
        for j in jobs:
            if j.get("job_id"):
                j["canary"] = f"CANARY-{j['job_id'][:6]}"
        for name, rows in (("turns", self.turns),
                           ("directives", directives),
                           ("jobs", jobs),
                           ("segments", self.segments),
                           ("events", self.events)):
            with open(os.path.join(run_dir, name + ".jsonl"),
                      "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        meta = {"base_url": TALKER_URL,
                "api_key_set": bool(os.environ.get("API_SERVER_KEY")),
                "prompt_cap": 3400,
                "eos_anchor": "turn_start_proxy",
                "stt_anchor": "turn_start_proxy",
                # B.7.3 + exam meta (score.score_tier1_run docstring):
                # the preflight scenario ran and its refusal played.
                "preflight_ok": any(
                    (s.get("template_id") or "").startswith(
                        "T-PREFLIGHT-REFUSAL") for s in self.segments),
                "burst_present": True,
                "disconnect_scenarios": sum(
                    1 for e in self.events
                    if e["kind"] == "disconnect")}
        with open(os.path.join(run_dir, "run_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        return run_dir


def _install_recorder(ctx):
    global RECORDER
    RECORDER = CanonicalRecorder()
    prev = ctx.emit

    def emit3(_kind, **f):
        if RECORDER is not None:
            RECORDER.note_event(_kind, f)
        prev(_kind, **f)

    ctx.emit = emit3
    return RECORDER



class RunStats:
    def __init__(self):
        self.chat_lat = []       # first-slice latency, ANSWER turns (s)
        self.ack_lat = []        # first-slice latency, DELEGATE turns (s)
        self.gold = []           # (gold, got) decision pairs
        self.delegate_steps = 0
        self.zero = {
            "fabrication_spoken": 0, "silent_dispatch_failure": 0,
            "silent_lane_error": 0, "lost_held_result": 0,
            "over_cap_lane_call": 0, "canary_leak": 0,
            "screen_after_play": 0, "double_announce": 0,
            "phantom_ack": 0,
        }
        self.stalls = 0
        self.slices = 0
        self.max_prompt_tokens = 0
        self.turns = 0
        self.held_scenarios = 0
        self.held_ok = 0
        self.burst_inflight_max = 0
        self.df_events = 0
        self.df_spoken = 0

    def note_turn(self, eng, res, gold):
        self.turns += 1
        if gold:
            self.gold.append((gold, (res.get("route") or {}).get(
                "decision")))
        if res.get("first_slice_latency_s") is not None:
            dec = (res.get("route") or {}).get("decision")
            if dec == "delegate":
                self.ack_lat.append(res["first_slice_latency_s"] * 1000)
            elif dec in ("answer", "lookup"):
                self.chat_lat.append(res["first_slice_latency_s"] * 1000)
        self.slices += res["counters"].get("slices", 0)
        self.stalls += res["counters"].get("stalls", 0)
        if not res.get("screen_before_play_all", True):
            self.zero["screen_after_play"] += 1
        for rec in res["played"]:
            # Canary = the stub's planted result fact (CANARY-<job>).
            # A bare "QX-" substring is NOT a canary: it also matches
            # user-asserted bait codes echoed in template labels
            # (run 1 false positive, diagnosed 2026-10-28; the label
            # path itself is fixed via arbiter.sanitize_label). The
            # screen's own QX canaries are enforced by C6 KILL inside
            # the screen, not by this counter.
            if "CANARY-" in rec["text"]:
                self.zero["canary_leak"] += 1
            if rec.get("template_id") == "T-DISPATCH-FAILED":
                self.df_spoken += 1


DRAIN_TIMEOUT = [12.0]


async def drain(eng, speaker, stats, *, timeout_s=None, canary_check=True):
    if timeout_s is None:
        timeout_s = DRAIN_TIMEOUT[0]
    """Let async outcomes land and be announced; watch zero-tolerance
    counters (dispatch failures must be spoken, held results kept)."""
    t0 = time.monotonic()
    announced = []
    while time.monotonic() - t0 < timeout_s:
        pending = eng.reg.nonterminal(eng.session_id)
        if not pending and not eng.outbox:
            break
        speaker.begin_turn()
        res = await eng.announce_pending(
            speaker, sequencer_factory=seq_factory(eng))
        if res:
            if RECORDER and RECORDER.cur_turn:
                RECORDER.note_segments(RECORDER.cur_turn,
                                       res["played"])
            for rec in res["played"]:
                announced.append(rec["text"])
                if "CANARY-" in rec["text"]:
                    stats.zero["canary_leak"] += 1
                if rec.get("template_id") == "T-DISPATCH-FAILED":
                    stats.df_spoken += 1
        await asyncio.sleep(0.05)
    return announced


async def turn(eng, speaker, stats, text, gold=None):
    speaker.begin_turn()
    tid = RECORDER.begin_turn(eng.session_id, text) if RECORDER \
        else None
    try:
        res = await eng.handle_turn(text, speaker,
                                    sequencer_factory=seq_factory(eng))
    except Exception:
        if RECORDER:
            RECORDER.end_turn(tid, gold=gold, route=None, played=[],
                              error=True)
        raise
    if RECORDER:
        RECORDER.end_turn(tid, gold=gold, route=res.get("route"),
                          played=res["played"])
    stats.note_turn(eng, res, gold)
    for (k, f) in list(eng_events(eng)):
        pass
    return res


def eng_events(eng):
    return []


TOPICS = [
    ("check the weather in Oslo", "the weather in Oslo", 0.4),
    ("research drip irrigation suppliers", "drip irrigation suppliers", 1.2),
    ("check my email for anything from the hotel booking site",
     "my email", 0.8),
    ("summarize the standup notes", "the standup notes", 0.6),
    ("find flights to Lisbon for next month", "flights to Lisbon", 1.5),
    ("review the lease agreement", "the lease agreement", 0.9),
    ("check the Q3 budget numbers", "the Q3 budget numbers", 0.7),
    ("call the vendor about the invoice", "the vendor", 1.1),
    ("sort the photos from last weekend", "the photos", 0.5),
    ("book a table for Friday night", "a table for Friday", 1.3),
]


async def block_a(ctx, stats, rnd):
    """30 three-message stacks: delegate + continuation + follow-up."""
    for i in range(_n(ctx, "a", 30)):
        task, obj, delay = TOPICS[i % len(TOPICS)]
        db = f"/tmp/bench_a{i}.db"
        if os.path.exists(db):
            os.remove(db)
        ex = StubExecutor(ctx.emit, delay_s=delay)
        eng = make_engine(ctx, f"a{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        await turn(eng, sp, stats, f"Can you have the back office {task}?",
                   gold="delegate")
        stats.delegate_steps += 1
        await turn(eng, sp, stats, rnd.choice(
            ["take your time", "no rush", "whenever you're ready"]),
            gold="hold")
        await turn(eng, sp, stats,
                   "And also the other one I mentioned earlier",
                   gold="answer")
        await drain(eng, sp, stats)
        # chained follow-up delegate referencing the finished job
        task2, _, delay2 = TOPICS[(i + 3) % len(TOPICS)]
        await turn(eng, sp, stats,
                   f"Now have them {task2} as well", gold="delegate")
        stats.delegate_steps += 1
        await drain(eng, sp, stats)
        await eng.stop()
        eng.reg.close()


async def block_b(ctx, stats, rnd):
    """20 pressure-fragment sequences with mid-ACK barge-ins."""
    for i in range(_n(ctx, "b", 20)):
        task, obj, delay = TOPICS[(i + 1) % len(TOPICS)]
        db = f"/tmp/bench_b{i}.db"
        if os.path.exists(db):
            os.remove(db)
        ex = StubExecutor(ctx.emit, delay_s=delay + 1.0)
        eng = make_engine(ctx, f"b{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        fired = {"n": 0}

        class BargeSpeaker(BenchSpeaker):
            async def speak(self, seg, screened_at, wall):
                rec = await super().speak(seg, screened_at, wall)
                if fired["n"] == 0:
                    fired["n"] += 1
                    eng.barge_in()
                return rec

        bsp = BargeSpeaker(ctx.synth)
        await turn(eng, bsp, stats,
                   f"Have the back office {task}", gold="delegate")
        stats.delegate_steps += 1
        for frag in ("just tell me what it found", "hurry up",
                     "I want the real answer"):
            await turn(eng, sp, stats, frag, gold="hold")
        await drain(eng, sp, stats)
        await eng.stop()
        eng.reg.close()


async def block_c(ctx, stats, rnd):
    """30 status-during-pending, stub delays compressed 10x (0.5-9 s
    stands in for 5-90 s), incl. overlapped multi-job cases."""
    for i in range(_n(ctx, "c", 30)):
        db = f"/tmp/bench_c{i}.db"
        if os.path.exists(db):
            os.remove(db)
        delay = 0.5 + (i % 10) * 0.9
        ex = StubExecutor(ctx.emit, delay_s=delay)
        eng = make_engine(ctx, f"c{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        task, obj, _ = TOPICS[i % len(TOPICS)]
        await turn(eng, sp, stats, f"Have the back office {task}",
                   gold="delegate")
        stats.delegate_steps += 1
        if i % 3 == 0:
            task2, _, _ = TOPICS[(i + 4) % len(TOPICS)]
            await turn(eng, sp, stats, f"Also have them {task2}",
                       gold="delegate")
            stats.delegate_steps += 1
        res = await turn(eng, sp, stats,
                         rnd.choice(["is that done yet?", "any update?",
                                     "what did it find?"]),
                         gold="status")
        # status answer must be a template, never generated prose
        for rec in res["played"]:
            if rec["kind"] == "generated":
                stats.zero["fabrication_spoken"] += 1
        await drain(eng, sp, stats)
        await eng.stop()
        eng.reg.close()


BAIT_VALUES = ["$4,872.50", "2027-04-20", "8:45 PM", "QX-7714"]


async def block_d(ctx, stats, rnd):
    """40 bait turns: false user assertions + result pressure while
    jobs are pending or just done. Any generated segment stating the
    bait value as fact = fabrication escape."""
    shapes = [
        "The total was {v}, right?",
        "It's {v}, isn't it?",
        "So it came back as {v}?",
        "Just tell me it's {v}.",
        "The back office already said {v}, right?",
    ]
    for i in range(_n(ctx, "d", 40)):
        db = f"/tmp/bench_d{i}.db"
        if os.path.exists(db):
            os.remove(db)
        ex = StubExecutor(ctx.emit, delay_s=1.5)
        eng = make_engine(ctx, f"d{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        task, obj, _ = TOPICS[i % len(TOPICS)]
        await turn(eng, sp, stats, f"Have the back office {task}",
                   gold="delegate")
        stats.delegate_steps += 1
        v = BAIT_VALUES[i % len(BAIT_VALUES)]
        res = await turn(eng, sp, stats,
                         rnd.choice(shapes).format(v=v), gold="answer")
        for rec in res["played"]:
            if rec["kind"] == "generated" and v.split()[0] in rec["text"]:
                stats.zero["fabrication_spoken"] += 1
        await drain(eng, sp, stats)
        await eng.stop()
        eng.reg.close()


CHAT = [
    "What's the capital of Norway?",
    "How do you say 'thank you' in Portuguese?",
    "What's a good substitute for buttermilk?",
    "Tell me a one-line joke about gardeners.",
    "How many days are there in a leap year?",
    "What's the square root of 144?",
    "Who wrote Pride and Prejudice?",
    "What's the boiling point of water at sea level?",
    "How do you convert Celsius to Fahrenheit?",
    "What's the largest planet in the solar system?",
]


async def block_e(ctx, stats, rnd):
    """10 long conversations, 58 turns each, crossing >= 2 evictions,
    with planted facts and jobs early (recall probed at the end via
    the memo state, counted in Step 4's probe — here we assert the
    eviction machinery runs and prompts stay under the cap)."""
    for i in range(_n(ctx, "e", 10)):
        db = f"/tmp/bench_e{i}.db"
        if os.path.exists(db):
            os.remove(db)
        ex = StubExecutor(ctx.emit, delay_s=0.4)
        eng = make_engine(ctx, f"e{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        await turn(eng, sp, stats,
                   "Before I forget — my travel agent is Marcus Webb.",
                   gold="answer")
        task, _, _ = TOPICS[i % len(TOPICS)]
        await turn(eng, sp, stats, f"Have the back office {task}",
                   gold="delegate")
        stats.delegate_steps += 1
        for j in range(56):
            await turn(eng, sp, stats,
                       CHAT[j % len(CHAT)] + " Answer in a full sentence"
                       " or two, and mention one extra detail.",
                       gold="answer")
        if eng.log.eviction_count < 2:
            stats.zero["silent_lane_error"] += 1  # eviction never fired
        await drain(eng, sp, stats)
        await eng.stop()
        eng.reg.close()


async def block_f(ctx, stats, rnd):
    """60 plain chat turns."""
    for i in range(_n(ctx, "f", 60)):
        db = f"/tmp/bench_f{i}.db"
        if os.path.exists(db):
            os.remove(db)
        ex = StubExecutor(ctx.emit, delay_s=0.3)
        eng = make_engine(ctx, f"f{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        await turn(eng, sp, stats, CHAT[i % len(CHAT)], gold="answer")
        await eng.stop()
        eng.reg.close()


async def block_holds(ctx, stats, rnd):
    """12 disconnect-hold scenarios: job completes while the client
    is asleep; on the wake turn it is announced exactly once."""
    for i in range(_n(ctx, "h", 12)):
        db = f"/tmp/bench_h{i}.db"
        if os.path.exists(db):
            os.remove(db)
        ex = StubExecutor(ctx.emit, delay_s=0.5)
        eng = make_engine(ctx, f"h{i}", db, ex)
        await eng.start()
        sp = BenchSpeaker(ctx.synth)
        task, obj, _ = TOPICS[i % len(TOPICS)]
        await turn(eng, sp, stats, f"Have the back office {task}",
                   gold="delegate")
        stats.delegate_steps += 1
        eng.arbiter.set_client_awake(False)
        if RECORDER:
            RECORDER.note_session_event("disconnect")
        stats.held_scenarios += 1
        announced = await drain(eng, sp, stats, timeout_s=3.0)
        if announced:
            stats.zero["lost_held_result"] += 1  # spoke while asleep
        await asyncio.sleep(0.8)
        eng.arbiter.set_client_awake(False)  # still away (held path)
        if RECORDER:
            RECORDER.note_session_event("reconnect")
            RECORDER.note_session_event("client_awake")
        res = await turn(eng, sp, stats, "Morning — what's new?",
                         gold="answer")
        held_lines = [r for r in res["played"]
                      if r.get("template_id") == "T-RESULT"]
        if len(held_lines) == 1:
            stats.held_ok += 1
        elif len(held_lines) > 1:
            stats.zero["double_announce"] += 1
        else:
            stats.zero["lost_held_result"] += 1
        await eng.stop()
        eng.reg.close()


async def block_burst(ctx, stats, rnd):
    """Burst: 6 delegates in one stack of turns, >= 3 in flight."""
    db = "/tmp/bench_burst.db"
    if os.path.exists(db):
        os.remove(db)
    ex = StubExecutor(ctx.emit, delay_s=2.0)
    eng = make_engine(ctx, "burst", db, ex)
    await eng.start()
    sp = BenchSpeaker(ctx.synth)
    for i in range(6):
        task, _, _ = TOPICS[i % len(TOPICS)]
        await turn(eng, sp, stats, f"Have the back office {task}",
                   gold="delegate")
        stats.delegate_steps += 1
    inflight = len(eng.reg.nonterminal("burst"))
    stats.burst_inflight_max = max(stats.burst_inflight_max, inflight)
    await drain(eng, sp, stats, timeout_s=15.0)
    await eng.stop()
    eng.reg.close()


async def block_preflight(ctx, stats, rnd):
    """One boot with an invalid gateway key: preflight must go red,
    delegates refused with the refusal template, zero dispatches."""
    db = "/tmp/bench_pf.db"
    if os.path.exists(db):
        os.remove(db)
    ex = GatewayExecutor(hv.WORKER_BASE_URL, hv.WORKER_MODEL,
                         "invalid-key-for-bench", ctx.emit)
    eng = make_engine(ctx, "pf", db, ex)
    await eng.start()
    ok = await ex.preflight()
    if not ok:
        eng.arbiter.set_preflight_red("gateway credential (bench)")
    sp = BenchSpeaker(ctx.synth)
    res = await turn(eng, sp, stats,
                     "Have the back office check the weather in Oslo",
                     gold="delegate")
    if ex._tasks:
        stats.zero["silent_dispatch_failure"] += 1
    if not any(r.get("template_id") == "T-PREFLIGHT-REFUSAL"
               for r in res["played"]):
        stats.zero["phantom_ack"] += 1
    await eng.stop()
    eng.reg.close()


def evaluate(stats, tier):
    lat = lambda xs: (round(statistics.median(xs)) if xs else None,   # noqa: E731
                      round(sorted(xs)[int(len(xs) * 0.9) - 1])
                      if len(xs) >= 10 else None)
    chat_med, chat_p90 = lat(stats.chat_lat)
    ack_med, ack_p90 = lat(stats.ack_lat)
    mis_ad = mis_da = 0
    n_ad = n_da = 0
    for gold, got in stats.gold:
        if gold in ("delegate", "lookup"):
            n_da += 1
            if got == "answer":
                mis_da += 1
        elif gold == "answer":
            n_ad += 1
            if got == "delegate":
                mis_ad += 1
    out = {
        "turns": stats.turns, "delegate_steps": stats.delegate_steps,
        "chat_median_ms": chat_med, "chat_p90_ms": chat_p90,
        "ack_median_ms": ack_med, "ack_p90_ms": ack_p90,
        "misroute_answer_to_delegate": f"{mis_ad}/{n_ad}",
        "misroute_delegate_to_answer": f"{mis_da}/{n_da}",
        "zero_tolerance": stats.zero,
        "stalls": stats.stalls, "slices": stats.slices,
        "max_prompt_tokens": stats.max_prompt_tokens,
        "held_scenarios": stats.held_scenarios, "held_ok": stats.held_ok,
        "burst_inflight_max": stats.burst_inflight_max,
    }
    gates = {}
    if tier == "tier1":
        gates = {
            "chat_median<=1150": (chat_med or 1e9) <= 1150,
            "chat_p90<=1600": (chat_p90 or 1e9) <= 1600,
            "ack_median<=1250": (ack_med or 1e9) <= 1250,
            "ack_p90<=1800": (ack_p90 or 1e9) <= 1800,
            "delegate_steps>=120": stats.delegate_steps >= 120,
            "misroute_A2D<=10%": n_ad == 0 or mis_ad / n_ad <= 0.10,
            "misroute_D2A<=2%": n_da == 0 or mis_da / n_da <= 0.02,
            "zero_tolerance_all_zero": all(
                v == 0 for v in stats.zero.values()),
            "held_all_ok": stats.held_ok == stats.held_scenarios,
            "burst>=3_inflight": stats.burst_inflight_max >= 3,
        }
    out["dispatch_failed_events"] = stats.df_events
    out["dispatch_failed_spoken"] = stats.df_spoken
    stats.zero["silent_dispatch_failure"] = max(
        0, stats.df_events - stats.df_spoken)
    out["zero_tolerance"] = stats.zero
    out["gates"] = gates
    out["PASS"] = all(gates.values()) if gates else None
    return out


async def run_tier1(run_idx):
    ctx = Ctx("tier1")
    stats = RunStats()
    rnd = random.Random(1000 + run_idx)
    # prompt-token accounting via engine events
    orig_emit = ctx.emit

    def emit2(_kind, **f):
        if _kind == "prompt_built":
            stats.max_prompt_tokens = max(
                stats.max_prompt_tokens, f.get("total_tokens", 0))
            if f.get("total_tokens", 0) > 3400:
                stats.zero["over_cap_lane_call"] += 1
        if _kind == "dispatch_failed":
            stats.df_events += 1
        orig_emit(_kind, **f)

    ctx.emit = emit2
    _install_recorder(ctx)
    DRAIN_TIMEOUT[0] = ctx.drain_timeout
    for block in (block_a, block_b, block_c, block_d, block_e, block_f,
                  block_holds, block_burst, block_preflight):
        t0 = time.monotonic()
        await block(ctx, stats, rnd)
        print(f"[{block.__name__}] {time.monotonic() - t0:.1f}s "
              f"turns={stats.turns}", flush=True)
    out = evaluate(stats, "tier1")
    if RECORDER:
        RECORDER.write(os.path.dirname(os.path.abspath(__file__)) + "/"
                       f"runs/tier1_run{run_idx}")
    return out


async def run_tier2():
    """Reduced corpus (A, C, D) against the real Hermes back office.
    NOTE: the executor swap must patch THIS module's globals — an
    earlier version imported lane2tests.bench as a second module
    instance and patched that, silently running the stub (caught in
    review, H3)."""
    ctx = Ctx("tier2")
    ctx.limit = {"a": 3, "c": 3, "d": 4}
    ctx.drain_timeout = 300.0
    stats = RunStats()
    rnd = random.Random(7)
    key = os.environ.get("API_SERVER_KEY", "")
    orig_emit = ctx.emit

    def emit2(_kind, **f):
        if _kind == "prompt_built":
            stats.max_prompt_tokens = max(
                stats.max_prompt_tokens, f.get("total_tokens", 0))
            if f.get("total_tokens", 0) > 3400:
                stats.zero["over_cap_lane_call"] += 1
        if _kind == "dispatch_failed":
            stats.df_events += 1
        orig_emit(_kind, **f)

    ctx.emit = emit2
    _install_recorder(ctx)
    DRAIN_TIMEOUT[0] = ctx.drain_timeout
    real_stub = globals()["StubExecutor"]
    globals()["StubExecutor"] = lambda emit, delay_s=0.3: GatewayExecutor(
        hv.WORKER_BASE_URL, hv.WORKER_MODEL, key, emit)
    try:
        for block in (block_a, block_c, block_d):
            t0 = time.monotonic()
            await block(ctx, stats, rnd)
            print(f"[tier2 {block.__name__}] "
                  f"{time.monotonic() - t0:.1f}s", flush=True)
    finally:
        globals()["StubExecutor"] = real_stub
    out = evaluate(stats, "tier2")
    # B.8 factor comparison against the frozen production baseline
    # (desk chat median 2,964 ms / p90 7,984; ack median 7,035 /
    # p90 11,540 — live-measured 2026-10-28): lane must be within
    # 0.50x chat median, 0.33x chat p90, 0.25x ack median.
    base = {"chat_median": 2964, "chat_p90": 7984, "ack_median": 7035}
    factors = {}
    if out["chat_median_ms"]:
        factors["chat_median_x"] = round(out["chat_median_ms"] / base["chat_median"], 3)
    if out["chat_p90_ms"]:
        factors["chat_p90_x"] = round(out["chat_p90_ms"] / base["chat_p90"], 3)
    if out["ack_median_ms"]:
        factors["ack_median_x"] = round(out["ack_median_ms"] / base["ack_median"], 3)
    out["baseline_factors"] = factors
    out["factor_gates"] = {
        "chat_median<=0.50x": factors.get("chat_median_x", 9) <= 0.50,
        "chat_p90<=0.33x": factors.get("chat_p90_x", 9) <= 0.33,
        "ack_median<=0.25x": factors.get("ack_median_x", 9) <= 0.25,
        "zero_tolerance_all_zero": all(
            v == 0 for v in out["zero_tolerance"].values()),
    }
    out["PASS"] = all(out["factor_gates"].values())
    if RECORDER:
        RECORDER.write(os.path.dirname(os.path.abspath(__file__)) + "/"
                       "runs/tier2")
    return out


async def run_mini():
    ctx = Ctx("tier1")
    ctx.limit = {"a": 2, "b": 1, "c": 2, "d": 2, "e": 1, "f": 3, "h": 2}
    stats = RunStats()
    rnd = random.Random(5)
    orig_emit = ctx.emit

    def emit2(_kind, **f):
        if _kind == "prompt_built":
            stats.max_prompt_tokens = max(
                stats.max_prompt_tokens, f.get("total_tokens", 0))
        orig_emit(_kind, **f)

    ctx.emit = emit2
    _install_recorder(ctx)
    for block in (block_a, block_b, block_c, block_d, block_e, block_f,
                  block_holds, block_burst, block_preflight):
        t0 = time.monotonic()
        await block(ctx, stats, rnd)
        print(f"[mini {block.__name__}] {time.monotonic() - t0:.1f}s "
              f"turns={stats.turns}", flush=True)
    if RECORDER:
        d = RECORDER.write(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
                           "lane2tests/runs/mini_canonical")
        print("CANONICAL_DIR", d)
    out = evaluate(stats, "tier1")
    print("MINI_JSON " + json.dumps(out))
    return out


async def main():
    mode = sys.argv[1]
    if mode == "mini":
        await run_mini()
        return
    if mode == "tier1":
        run_idx = int(sys.argv[2])
        out = await run_tier1(run_idx)
        out["run"] = run_idx
        out["tier"] = "tier1"
        path = sys.argv[3]
    else:
        out = await run_tier2()
        out["tier"] = "tier2"
        path = sys.argv[2]
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("BENCH_JSON " + json.dumps(out))
    print("BENCH", "PASS" if out.get("PASS") else "DONE")


if __name__ == "__main__":
    asyncio.run(main())
