"""SpeechPlan sequencer (v2 §2.5 + Part E) with sliced speech.

A turn's speech is a plan, not a string: template obligations from
the arbiter (§2.6 fixed phrasings) plus at most one talker generation.
The generation is spoken in SLICES — sentence-sized segments — each
generated and screened while the previous slice plays. Invariants:

- Nothing reaches the speaker except ScreenedSegments (screen-built
  or template-built); the type is the enforcement.
- Every slice's screen timestamp precedes its playback timestamp
  (recorded per segment; the bench asserts it mechanically).
- A slice that fails screening is never spoken: it is dropped,
  replaced by a template segment, or ends the generation.
- A worker result landing mid-answer is woven in as a template
  segment after the current slice finishes — never mid-sentence.
- Barge-in cancels the whole plan: playback flushes within 200 ms,
  unplayed slices are discarded (never resumed — §2.8).
"""
import asyncio
import re
import time

from . import templates

_SENT_END = re.compile(r"[.!?](?:[\"'”’)]\s+|\s+|$)")

# §2.5.2 priority classes: lower speaks sooner within a plan.
P_CANCEL_CONFIRM = 0
P_RESULT = 1
P_STATUS_RESP = 2
P_ACK = 3
P_CLARIFY = 4
P_HELD = 5
P_CHATTER = 6


def segment_priority(kind):
    return {
        "cancel_confirm": P_CANCEL_CONFIRM,
        "result": P_RESULT,
        "status": P_STATUS_RESP,
        "ack": P_ACK,
        "clarify": P_CLARIFY,
        "held": P_HELD,
        "chatter": P_CHATTER,
    }.get(kind, P_CHATTER)


class SentenceSplitter:
    """Incremental sentence slicer over a token stream."""

    def __init__(self, max_words=60):
        self.buf = ""
        self.max_words = max_words

    def feed(self, token):
        self.buf += token
        out = []
        while True:
            m = _SENT_END.search(self.buf)
            if m and m.start() > 0:
                sent = self.buf[:m.end()].strip()
                self.buf = self.buf[m.end():]
                if sent:
                    out.append(sent)
                continue
            if len(self.buf.split()) > self.max_words:
                # force a slice at the last clause boundary
                cut = max(self.buf.rfind(", "), self.buf.rfind("; "),
                          self.buf.rfind(" — "))
                if cut > 20:
                    out.append(self.buf[:cut + 1].strip())
                    self.buf = self.buf[cut + 1:]
                    continue
            break
        return out

    def flush(self):
        s = self.buf.strip()
        self.buf = ""
        return [s] if s else []


class BenchSpeaker:
    """Bench/test speaker: real synthesis when a synth fn is given
    (Pocket TTS); playback timeline on the same monotonic clock the
    sequencer uses for screen timestamps, so the screen-before-play
    invariant and slice-starvation gaps are measured, not asserted.
    Playback of slice n+1 cannot start before slice n ends."""

    def __init__(self, synth=None):
        self.synth = synth  # fn(text) -> audio seconds
        self.played = []
        self.busy_until = None
        self.flushed_at = None

    def begin_turn(self):
        """Bench protocol: a new user turn starts on a quiet line —
        clear accumulated virtual playback debt so per-turn
        first-slice latencies measure this turn only."""
        self.busy_until = None

    async def speak(self, seg, screened_at, wall):
        now = time.monotonic()
        start = now if self.busy_until is None else \
            max(now, self.busy_until)
        dur = self.synth(seg.text) if self.synth else \
            max(0.4, len(seg.text.split()) / 2.6)
        self.busy_until = start + dur
        rec = {"text": seg.text, "kind": seg.kind,
               "template_id": seg.template_id, "job_id": seg.job_id,
               "screened_at": screened_at, "play_start": start,
               "play_end": self.busy_until, "duration_s": dur,
               "screen_before_play": screened_at <= start + 1e-9}
        self.played.append(rec)
        return rec

    def flush(self):
        self.flushed_at = time.monotonic()


class SpeechSequencer:
    def __init__(self, *, screen, guard, talker, emit, session_id,
                 p_status_fn=None):
        self.screen = screen
        self.guard = guard
        self.talker = talker
        self.emit = emit
        self.session_id = session_id
        # Part D signal: per-sentence status probability from the
        # encoder (zero-shot status hypothesis); None disables it
        self.p_status_fn = p_status_fn
        self._cancelled = False
        self._pending_weave = []

    def cancel(self):
        self._cancelled = True

    def note_result_obligation(self, obligation):
        """A job result landed mid-answer: queue its template segment
        to be woven after the current slice (§2.8: never mid-sentence,
        never by interrupting)."""
        self._pending_weave.append(obligation)

    # -------------------------------------------------- helpers
    def _template_seg(self, ob):
        seg = self.screen.template_segment(
            ob["template_id"], ob.get("slots") or {},
            job_id=ob.get("job_id"))
        return seg

    def _screen_to_segments(self, action, payload, sentence):
        """Map a screen verdict to 0..n segments. FALLBACK/KILL are
        returned as control signals via the second tuple element."""
        if action == "PASS":
            from .screen import ScreenedSegment
            seg = ScreenedSegment(
                ScreenedSegment._KEY, payload, "generated",
                snapshot_version=self.screen.S0_version)
            return [seg], None
        if action == "DROP":
            return [], None
        if action in ("REPLACE", "FALLBACK"):
            data = payload
            segs = []
            items = data if (data and isinstance(data, tuple)
                             and data and isinstance(data[0], tuple)) \
                else (data,)
            for item in items:
                if item is None:
                    continue
                tid, slots = item
                segs.append(self.screen.template_segment(tid, slots))
            return segs, ("end" if action == "FALLBACK" else None)
        if action == "KILL":
            return [], "kill"
        return [], None

    # -------------------------------------------------- main
    async def run_turn(self, *, turn_id, obligations, generation,
                       speaker, jobs_snapshot_fn, p_status=None,
                       now_fn=time.monotonic):
        """obligations: [{kind, template_id, slots, job_id}] template
        segments in priority order (caller sorts). generation: None or
        {messages, profile} for the single talker generation.
        Returns a result dict with the playback record + counters."""
        t_start = now_fn()
        self._cancelled = False
        played_meta = []
        counters = {"slices": 0, "stalls": 0, "dropped": 0,
                    "replaced": 0, "kills": 0, "fallbacks": 0}
        first_at = None
        last_end = 0.0
        kill = False

        async def play(seg, screened_at):
            nonlocal first_at, last_end
            if self._cancelled:
                return
            rec = await speaker.speak(seg, screened_at, now_fn())
            played_meta.append(rec)
            if first_at is None:
                first_at = rec["play_start"]
            last_end = rec["play_end"]
            counters["slices"] += 1

        # template obligations first (priority order given by caller),
        # except chatter-class obligations, which follow the generation
        ordered = sorted(
            obligations, key=lambda o: segment_priority(o["kind"]))
        pre = [o for o in ordered if o["kind"] != "chatter"]
        post = [o for o in ordered if o["kind"] == "chatter"]
        for ob in pre:
            if self._cancelled:
                break
            await play(self._template_seg(ob), now_fn())

        # the one talker generation, sliced
        if generation is not None and not self._cancelled and not kill:
            self.guard.begin_generation(turn_id)
            splitter = SentenceSplitter()
            end_gen = False

            async def handle_sentence(sentence):
                nonlocal end_gen, kill
                if self._cancelled or end_gen:
                    return
                # topic guard first (ANSWER only; guard enforces)
                jobs, _facts = jobs_snapshot_fn()
                ps = p_status
                if self.p_status_fn is not None:
                    try:
                        ps = self.p_status_fn(sentence)
                    except Exception:
                        ps = None
                gv = self.guard.evaluate(
                    sentence=sentence, turn_id=turn_id,
                    turn_type=generation["profile"], jobs=jobs,
                    session_id=self.session_id, p_status=ps)
                if gv["action"] in ("DRAFT", "BLOCK"):
                    seg = self.screen.template_segment(
                        gv["template_id"], gv["slots"],
                        job_id=(gv["job"] or {}).get("job_id"))
                    await play(seg, now_fn())
                    counters["replaced"] += 1
                    return
                if gv["action"] == "DROP":
                    counters["dropped"] += 1
                    return
                t_scr = now_fn()
                action, payload = self.screen.screen_sentence(sentence)
                if action != "PASS":
                    counters["replaced" if action == "REPLACE"
                             else "dropped"] += 1
                segs, control = self._screen_to_segments(
                    action, payload, sentence)
                for seg in segs:
                    if self._cancelled:
                        return
                    if seg.kind == "generated":
                        # release-time snapshot revalidation (§2.7.2)
                        va, vp = self.screen.validate_release(
                            seg.text, seg.snapshot_version)
                        if va != "PASS":
                            sub, ctl2 = self._screen_to_segments(va, vp,
                                                                 seg.text)
                            if ctl2 == "kill":
                                control = "kill"
                            segs2 = sub
                            for s2 in segs2:
                                await play(s2, now_fn())
                            continue
                    await play(seg, t_scr)
                if control == "kill":
                    kill = True
                    counters["kills"] += 1
                    fb = self.screen.template_segment("T-FB-KILL", {})
                    await play(fb, now_fn())
                    end_gen = True
                elif control == "end":
                    counters["fallbacks"] += 1
                    end_gen = True
                # weave any result that landed during this slice
                while self._pending_weave and not self._cancelled:
                    ob = self._pending_weave.pop(0)
                    await play(self._template_seg(ob), now_fn())

            stream = self.talker.stream(generation["messages"])
            try:
                async for item in stream:
                    if self._cancelled or end_gen:
                        break
                    # TalkerClient yields (delta, None) chunks and a
                    # final (None, usage); plain async generators of
                    # str are also accepted (bench fakes).
                    token = item[0] if isinstance(item, tuple) else item
                    if not token:
                        continue
                    for sentence in splitter.feed(token):
                        await handle_sentence(sentence)
                        if end_gen or self._cancelled:
                            break
                if not end_gen and not self._cancelled:
                    for sentence in splitter.flush():
                        await handle_sentence(sentence)
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass

        for ob in post:
            if self._cancelled:
                break
            await play(self._template_seg(ob), now_fn())

        # stall accounting: gaps in the simulated playback timeline
        prev_end = None
        for rec in played_meta:
            if prev_end is not None and \
                    rec["play_start"] - prev_end > 0.4:
                counters["stalls"] += 1
            prev_end = rec["play_end"]
        return {
            "turn_id": turn_id,
            "first_slice_at": first_at,
            "first_slice_latency_s":
                (first_at - t_start) if first_at is not None else None,
            "played": played_meta,
            "counters": counters,
            "cancelled": self._cancelled,
            "kill": kill,
            "screen_before_play_all": all(
                r["screen_before_play"] for r in played_meta),
        }
