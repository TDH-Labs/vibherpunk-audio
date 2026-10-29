# Adversarial review — Vibherpunk Audio v2 conversation lane ("lane2")

Reviewer: independent adversarial reviewer (did not build this system).
Scope reviewed: staged copies in `work/src/` plus, for the four modules
missing from `work/src/` (events, registry, executor, talker), the older
package snapshot in `work/lane2src/lane2/` — see H7. Read-only review;
no Mac contact, nothing rerun.

## Overall verdict: **NO-SHIP**

Tier 1 bench JSONs do not exist in evidence yet, Tier 2 has not run, and
the harness that would produce them can pass vacuously (H3). Independently
of the bench, the code as staged breaks three of the six invariants in
ways that let unbacked job claims reach TTS (H1, H2), lets world-changing
actions dispatch without the closed-set confirm (H4), and breaks
held-result exactly-once in both directions (H5). Any one of H1–H7
blocks cutover under the SCOPE rule ("High findings block cutover").

---

## High findings

### H1 — Template segments bypass the §2.7 screen, and their slots carry unscreened untrusted text
Invariant 1/2 breach. `Screen.template_segment` is explicitly "never
screened at runtime" (`work/src/screen.py:738-741`); the build-time lint
(`lint_templates`, screen.py) only checks template *format strings* with
dummy slots (`{"eta": "in a while", ...}`). At runtime the slots are
filled with untrusted content and spoken verbatim:

- `T-RESULT` slots take `summary` straight from the worker's
  `SPOKEN SUMMARY:` line (`work/src/arbiter.py` `_on_done` → Obligation
  `T-RESULT`; rendered via templates.py `T-RESULT`). The worker is an
  LLM with tools; its summary can contain invented money/dates, bait
  values, outcome claims, or canary-shaped strings — none of C0–C6 ever
  sees it, because the segment is constructed as "template".
- Same path for `T-GUARD-DONE` (`work/src/guard.py` `_status_template_for`:
  `summary = outcome_line(job)`, i.e. truncated `result_summary`),
  `T-QUOTE` (`summary = quote_text`, screen.py `_screen_inner` C5 branch),
  and labels derived from raw user text (`_label()` in arbiter.py takes
  `spec.obj`/`spec.text` from the user's utterance).

So the one channel that *by design* speaks job outcomes is the channel
with no screening of the outcome text itself. "Templates are the only
voice of job state" is only safe if slot values are registry-typed
values; `result_summary` is free text from the back-office model.
**Blocker.**

### H2 — ScreenedSegment is forgeable, speakers don't enforce it, and the production speaker fabricates the invariant
Invariant 2 breach, three layers:

- `ScreenedSegment._KEY` is a public class attribute
  (`work/src/screen.py`, `ScreenedSegment` class); `speech.py:157`
  itself constructs a "screened" segment directly in
  `_screen_to_segments`. Any code in the process can mint one.
- Neither speaker checks the type: `BenchSpeaker.speak` and
  `_PipecatSpeaker.speak` (`work/src/hermes_voice.py:2392`) accept any
  object with `.text`. The docstring claim "the type is the enforcement"
  is false at the only boundary that matters.
- `_PipecatSpeaker.speak` hardcodes `"screen_before_play": True`
  (hermes_voice.py:2404) with a word-count-estimated duration — in
  production the invariant is asserted, not measured. In the bench
  speaker the check is `screened_at <= start` (speech.py:114), but
  template segments are passed `now_fn()` at the moment of play
  (speech.py, `await play(self._template_seg(ob), now_fn())`), so they
  trivially pass; and for generated segments `t_scr = now_fn()` is taken
  at speech.py:250 *before* `screen_sentence()` runs — the timestamp
  proves the screen started, not that it completed before playback.
- Fallback path: on any turn exception, `Lane2Processor._run_turn`
  pushes a raw `TTSSpeakFrame(text=LANE_DOWN_MESSAGE)` downstream
  (hermes_voice.py:2470-2473), bypassing the sequencer entirely. The
  string is a constant, but it proves the TTS path is reachable without
  a ScreenedSegment whenever an exception occurs — including the
  OverCap case in M7, i.e. precisely on long/contended turns.

**Blocker.**

### H3 — The Tier 1 gate evaluation can pass vacuously; Tier 2 has no gates at all
`work/src/bench.py`:

- **Stale zero-tolerance gate.** `evaluate()` builds `gates` — including
  `"zero_tolerance_all_zero"` (bench.py:548) — and only *afterwards*
  recomputes `stats.zero["silent_dispatch_failure"] = max(0, df_events -
  df_spoken)` (bench.py:555). `PASS = all(gates.values())` therefore
  uses the pre-recompute value. During a run that counter is only ever
  touched by `block_preflight` (bench.py:501); the authoritative
  reconciliation — `df_events` counted from arbiter events minus
  `df_spoken` — is applied at bench.py:555, after the gate has already
  been computed. A run
  can report `PASS: true` alongside a non-zero `silent_dispatch_failure`
  in its own JSON.
- **delegate_steps counts its own inputs.** Every block does
  `stats.delegate_steps += 1` immediately after issuing a scripted
  delegate turn (bench.py:243, 255, 285, 309, 314, 353, 398, 440, 475),
  regardless of whether the engine routed, admitted, or dispatched
  anything. The `delegate_steps>=120` gate (bench.py:545) measures the
  script's length, not the lane's behaviour.
- **p90 is understated.** `sorted(xs)[int(len(xs)*0.9)-1]` (bench.py:511)
  returns the ~85th percentile at n=20 and the 89th at best; the
  p90 gates (1600/1800 ms) are evaluated against a softer statistic.
- **Misroute gates are vacuous-true on empty denominators**
  (`n_ad == 0 or ...`, bench.py gates) — acceptable for a full run,
  dangerous combined with the scenario caps (`Ctx.limit`, `_n()`),
  which can shrink blocks silently.
- **Tier 2 cannot pass or fail.** `evaluate(stats, "tier2")` leaves
  `gates = {}` and `PASS = None`; no factor comparison against the
  frozen A1 baseline (≤0.50× chat median, ≤0.33× p90, ≤0.25× ack)
  exists anywhere in the harness. SCOPE requires both tiers; the
  harness structurally cannot deliver Tier 2 evidence. The Tier 2
  runner is also internally broken (`make_gateway_engine` defined and
  unused; `StubExecutor` monkeypatched with a lambda of different
  signature; `make_engine` "patched" to itself).

**Blocker** — the bench evidence the cutover decision rests on is not
trustworthy as written.

### H4 — World-changing actions dispatch with no closed-set confirm; the confirm that exists accepts stale, fabricated-confidence "yes"
Invariant 6 breach.

- `lane_entry()` admits immediately for `provenance == "RULE"` or
  `p >= 1.0` (`work/src/arbiter.py:75`), and the engine *manufactures*
  that confidence: `if provenance == "RULE" and decision == "delegate":
  p_del = 1.0` (`work/src/engine.py:224-225`). A rule-matched "send /
  pay / book / email …" therefore goes QUEUED → dispatched after the
  ~700 ms amend window with no confirm of any kind. The band/confirm
  machinery only fires for *uncertain encoder* delegates. Invariant 6
  as briefed ("no world-changing action without the closed-set
  confirm") is not implemented; what is implemented is "no *uncertain*
  action without a confirm".
- The real-gateway READ_ONLY safeguard is a prompt suffix
  (`work/lane2src/lane2/executor.py:43-46,89` — "Do NOT send, write,
  modify…"), i.e. an instruction to the worker LLM. The only technical
  enforcement, `StubExecutor.attempt_write` (executor.py:223), exists
  only in the bench stub and is never invoked by `GatewayExecutor`.
  Combined with the deterministic verb-list classifier
  (`classify_task_class`, arbiter.py), a misclassified or
  prompt-injected worker task can perform writes under a READ_ONLY job.
- Confirm liveness: the confirm timer starts only when the prompt is
  reported played (`confirm_prompt_spoken`, arbiter.py:508-515).
  `focused_confirm()` (arbiter.py:505) returns the open confirm
  regardless of `spoken_mono`. If the prompt obligation is never played
  (client asleep, announce loop error, barge-in), the confirm stays
  armed indefinitely, and the engine pre-empts routing for *any* later
  turn whose text is a closed-set value (engine.py `handle_turn`
  confirm branch). `closed_value()` assigns a constant `p = 0.95`
  (engine.py:57) — above `P_CLOSED_MIN = 0.80` by construction, not by
  measurement — and matches bare "ok"/"sure"/"right". A conversational
  "ok" an hour later can dispatch a held SIDE_EFFECTING job. That is
  consent without freshness, scope, or a real confidence signal.

**Blocker.**

### H5 — Held results are not exactly-once: they can be lost silently, and cancel-suppressed results can be announced
Invariant 3 breach, both directions:

- **Loss.** `LaneEngine._speak_plan` marks held obligations announced
  by iterating the *obligation list*, not the played records
  (engine.py:~440 and `announce_pending` ~493:
  `for o in obligations: if o["kind"] == "held": await
  arbiter.mark_announced(...)`). If playback is barged-in/cancelled
  before the held segment plays, or the speaker errors, the job is
  still written `announced=1, delivered_mono=now` — the result is
  never heard and will never be offered again. (Contrast:
  `confirm_prompt_spoken` correctly keys off actually-played segments.)
- **Resurrection.** `_on_done` for a CANCEL_REQUESTED job writes DONE
  with `delivered_mono` set but `announced` untouched, on the stated
  theory that "held-result logic never surfaces it"
  (arbiter.py:789-800). It does: `JobRegistry.held_unannounced`
  (`work/lane2src/lane2/registry.py:313-320`) filters on `announced`
  only and never checks `delivered_mono`. On the next wake turn the
  engine's held path speaks `T-RESULT` for a job the user cancelled —
  exactly the §2.8 outcome the code comment claims to prevent.

**Blocker.**

### H6 — Step 6's RECOVERY and RECHECK gates test code that production never calls; BAIT is fitted to its own signals
- `TopicGuard.recheck()` (guard.py:307) and
  `note_status_like_utterance()` (guard.py:129) have **no production
  caller**: grep over engine/speech/hermes_voice finds them only in
  `step6bench.py`. The D.4 RECOVERY 200/200 and RECHECK 100/100 results
  quoted in TODOS.md therefore certify dead code. In production there
  is no recovery suppression and no commit-time re-check: a guard
  replacement rendered early in a generation is played even if the job
  was delivered meanwhile.
- The BAIT set is 4 hand-written sentence shapes cycled ×100 with label
  substitution (`step6bench.py` guard_bench), and in 2 of 4 modes the
  bench *supplies* `p_status ∈ [0.86, 0.99]` — i.e. it hands the guard
  the very signal that makes it fire, then counts the firing.
  Production `p_status` quality was, per the bench docstring, "measured
  separately on 60 sentences"; that number is not in TODOS.md or the
  staged evidence. Production `_lane2_p_status` is a synchronous
  zero-shot classifier call per sentence (hermes_voice.py:2506ff) whose
  latency is excluded from the guard latency gate (step6bench passes
  `p_status` in precomputed and reports p95 0.16 ms).
- On the Step 6 deviation itself: the unique-referent rule for DRAFT
  and the deictic subject-position rule (guard.py `_reference`) are
  principled — they encode "don't guess which job" and are exactly
  what the NEAR gate needs. The `status_lex` off-by-one fix
  (`{1,40}` → `{0,40}`) is a genuine grammar bug fix. But the two
  "missing completion patterns" were added against the same bench
  shapes that then scored 400/400, with no held-out set; with BAIT at
  a perfect score and PLAIN/NEAR drawn from 10 and 3 unique sentences
  respectively, the bench cannot distinguish "grammar learned" from
  "grammar fitted". Shipped configuration does match the benched one
  (spec-default thresholds, `DRAFT_ONLY`, `HashEmbedder`,
  monotonic guard clock — guard.py `DEFAULT_THRESHOLDS`,
  hermes_voice.py `build_lane2`), which is to the builder's credit;
  the problem is what the bench measures, not config drift.

**Blocker** (dead-code gates + unmeasured production signal).

### H7 — The staged copy is incomplete and divergent, Tier 1/Tier 2 evidence is absent, and the talker server is not started by the run scripts
- `work/src/` — the copy this review was directed at — contains **no**
  `registry.py`, `events.py`, `executor.py`, or `talker.py`. Those
  exist only in `work/lane2src/lane2/`, an older snapshot whose
  `arbiter.py`/`context.py`/`tagger.py` *differ* from `work/src/`
  (e.g. the completion-after-cancel write and the dedupe
  earlier-directive rule exist only in `work/src/arbiter.py`). Which
  tree is on the Mac, and whether the Mac's registry is the one whose
  kill-drive numbers TODOS.md quotes, cannot be established from the
  staged evidence. A hash-verified review of "the built lane" is
  impossible against a divergent, partial staging.
- Tier 1: TODOS.md shows the three runs *launched* (~14:55 PDT), no
  results filed; the JSONs are not available. Tier 2: not run. SCOPE
  requires ≥360 delegate steps across three consecutive full Tier 1
  runs plus Tier 2 factors plus this review before flip. None of the
  three legs is complete.
- TODOS.md itself notes: "run scripts do not yet start the talker
  server for lane2 — cutover step must add it or verify an always-on
  server before flipping." Flipping `HERMES_LANE_BACKEND=lane2` today
  points every turn at `127.0.0.1:8123` with nothing guaranteed to be
  listening; every turn becomes the H2 fallback message.

**Blocker.**

---

## Medium findings

### M1 — Registry: two writers in practice, ack-timeout is dead code under load, and speech reads non-durable state
`work/lane2src/lane2/registry.py`: (a) `_check_ack_timeouts_locked` is
called only inside the `while not self._buf` wait (registry.py:347-353);
while the buffer is non-empty — i.e. under load, or while a poison batch
is being requeued with backoff — the 2 s ack timeout is never evaluated,
so a single failing batch head-of-line-blocks every acked write with no
`degraded` signal and the Arbiter's `_create(ack_required=True)` hangs
the turn. (b) `snapshot()`/`get()` merge the *uncommitted* buffer into
reads (registry.py:277-290), contradicting the module docstring ("The
mirror contains exactly the durable state"); most `transition()` calls
default `ack_required=False`, so speech planning and guard/screen
snapshots routinely act on state that a SIGKILL would erase — the
Step 1 kill-drive numbers cover acked writes, not this read path.
(c) The Arbiter's dispatch loop performs transitions outside the
Arbiter lock (`_dispatch_loop` → `_dispatch_gate` → `transition`) while
`_on_done`/`_on_started`/`_confirm_reply` take the lock; `transition()`
does a read-modify-write of the full row from `_pending_or_mirror`, and
`_commit_batch` upserts the full row — two interleaved transitions
computed from the same base version can silently overwrite each other's
fields (journal keeps both, recovery check compares only MAX version).

### M2 — Barge-in ≤200 ms is unmeasured and the sequencer never flushes the speaker
`SpeechSequencer.cancel()` only sets a flag; nothing calls
`speaker.flush()` anywhere in speech.py/engine.py.
`_PipecatSpeaker.flush()` is a no-op (`pass`, hermes_voice.py:~2407).
Bench block B fires the barge after the first `speak()` *returns* and
records no flush latency; there is no barge-in gate in `evaluate()` at
all. SCOPE's ≤200 ms requirement has no evidence, and in production the
already-pushed `TTSSpeakFrame` relies entirely on Pipecat's downstream
InterruptionFrame handling, which this code does not verify.

### M3 — Boot integrity (pins / golden check) is never run in production
`check_pins()` and `TalkerClient.golden_check()` exist only in
`work/lane2src/lane2/talker.py` (lines 31, 169) and are called nowhere.
`build_lane2` (hermes_voice.py:2539-2552) loads `pins.json` only to find
`model_dir`, constructs the TalkerClient, and fires preflight and
`engine.start()` as unawaited `asyncio.create_task`s — a delegate that
arrives before the preflight task lands is dispatched against an
unverified gateway, and a tokenizer/model swap that breaks the cache
contract (§3.6.3) is never detected at boot.

### M4 — Eviction extraction barrier is unwired; extraction can stall turns
`SessionLog.pre_evict_hook` is never assigned anywhere (context.py:148
defines it; no engine/integration code sets it), and
`FactLedger.wait_turn` / `MemoState.stub_only_turns` (facts.py:410, 477)
are dead in production. `extract_turn_async` is awaited at the end of
every turn with no timeout — a hung GLiNER pass stalls all subsequent
turns, and the "extraction timed out → stub-only" degradation the memo
design specifies cannot occur.

### M5 — Bench realism: 10× time compression and virtual playback with per-turn debt reset
Bench block C states it plainly: "stub delays compressed 10x (0.5–9 s
stands in for 5–90 s)" (bench.py block_c docstring), while the guard's
recency windows (`t_recent_s=180`, `t_delivered_s=60`), the dedupe
trailing windows (20 s / 120 s), and confirm windows run on the real
monotonic clock — compression systematically favours the lane.
`BenchSpeaker` playback is virtual (`busy_until` arithmetic, speech.py
BenchSpeaker) and `begin_turn()` zeroes the accumulated playback debt
at every turn *and every drain iteration* — so first-slice latency
never includes queueing behind the previous turn's speech, stalls are
computed on a timeline that is reset whenever it would hurt, and the
sliced-speech overlap claim (next slice generated *while* one plays)
is never exercised against real time: `speak()` returns immediately.
The canary counters are likewise vacuous: StubExecutor's
`CANARY-{jid}` facts never enter any spoken summary (summaries come
from `stub_summary`/"Finished {obj}."), and the engine's `QX-` canaries
are never placed in the talker prompt, so `canary_leak` cannot fire.

### M6 — OverCap collapses the turn; tail shedding silently deletes lookup evidence
`SessionLog.build_prompt` raises `OverCap` above 3,400 tokens
(context.py:324); `LaneEngine.handle_turn` does not catch it, so the
turn dies into the H2 raw-frame fallback. Before that, when the tail
exceeds 550 tokens the builder silently truncates to the mode directive
only (context.py:295), discarding lookup facts and the open-jobs ledger
with no event or counter — the talker then answers a LOOKUP turn with
no evidence and the screen absorbs the damage invisibly.

### M7 — Gateway executor: no total timeout, and OA resume is a silent no-op
`GatewayExecutor.dispatch` uses `ClientTimeout(total=None)`
(executor.py:101): a hung stream leaves a job RUNNING forever with no
arbiter watchdog. Worse, `_oa_reply` transitions a NEEDS_INPUT job back
to RUNNING and calls `executor.resume()`, whose gateway implementation
only emits `executor_resume_unsupported` (executor.py `resume`) — the
job is RUNNING, nothing is executing it, and it will never terminate,
never announce, and will hold its context unit pinned via the retain
hook forever.

### M8 — Talker prompt and screen allowlist contradict invariant 1's letter, and the date allowlist can't match
The engine injects an `[Open jobs]` ledger — `"{task_label}:
{state}"` for every QUEUED/RUNNING job — into the talker prompt
(engine.py:372-379; context.py:284-287). The brief's invariant says the
talker never sees job states; the design evidently intends this ledger,
but then job-state fabrication by the talker is prevented only by the
guard/screen net, not by information hiding — and the guard only runs
on ANSWER-profile sentences. Separately, the screen's HEADER allowlist
adds today's date as ISO `YYYY-MM-DD` and time as `HH:MM`
(screen.py `_build_allowlist`), while `canonical_typed` canonicalises
DATE surfaces to normalised surface strings (e.g. "october 28 2026") —
the two can never be equal, so legitimate date/time statements by the
talker are systematically C2/C4 casualties; and `_bind_task` binds any
sentence to the single live/DONE job, with `_facts_for` also admitting
task-less lookup facts as support for that job's outcome claims.

### M9 — HashEmbedder cosine against neural-embedding thresholds
`embed.py` is candid that hashed bag-of-words cosine is "weaker than a
neural embedder", yet the guard's `cos_block=0.82` / draft band
0.74–0.82 (guard.py `DEFAULT_THRESHOLDS`, the spec defaults) are applied
unchanged. For short job labels, shared-token cosine is near-binary;
the thresholds' meaning under this embedder is uncalibrated, and every
Step 6 number inherits that. (Shipped = benched, per H6; the concern is
that both are measuring an uncalibrated signal.)

---

## Low findings

- **L1 — C1 null-result phrases fire unconditionally** (screen.py
  `_c1`): "no issues", "all clear", "everything checks out" are
  dropped/replaced even in job-free small talk ("the inspection had no
  issues"), because the null-result branch does not require a bound
  task. Availability/annoyance, not safety.
- **L2 — Status turns pick `live[0]`** (engine.py `handle_turn` status
  branch): with several open jobs, a status question is answered for
  whichever row the snapshot returns first, with no referent check and
  no disambiguation template.
- **L3 — `MemoState.render()` mutates the ledger** (facts.py
  `render_with` sets `r.dropped = True` during a read): a render under
  a transiently large P1 permanently discards P2 facts that would fit
  later. Render should be pure; drop decisions belong to absorb/evict.
- **L4 — Production first-audio truth is deferred**: `_PipecatSpeaker`
  estimates duration from word count and the engine's
  `first_slice_latency_s` in production measures frame-push time, not
  audio; SCOPE's live-verification numbers must come from SoundTap,
  not from the engine result dict.

## What checked out (for the record)

- Rollback default is real: `LANE_BACKEND` defaults to `"relay"`
  (hermes_voice.py:94); the lane2 stage is constructed only on
  `== "lane2"` (hermes_voice.py:2621; mobile_server.py:202), with the
  old stage intact in the `else` branch. (Byte-identity of old paths
  against pre-build production cannot be verified from the staged copy
  alone — no baseline was staged.)
- Registry ack design (group commit, publish-to-mirror before resolving
  futures, journal `write_id` PK with INSERT OR IGNORE) is sound for
  acked writes; the Step 1 numbers in TODOS.md are consistent with it.
- Screen precedence (C6→C5→C1→C3a→C3b→C4→C2), regeneration disabled,
  and the guard's unique-referent/deictic fixes are principled designs;
  the failures above are in enforcement boundaries and evidence, not
  in the check taxonomy.
- Frozen header: A0/A1 spans are built once and concatenated, never
  re-rendered (context.py SessionLog); `expected_cacheable` correctly
  excludes the newest unit in the accounting (the bytes still include
  it — the reprefill cost is acknowledged in the meta, not hidden).

## Required before re-review

1. Screen (or structurally validate) template slot values — at minimum
   worker summaries, quotes, and labels — or restrict T-RESULT-class
   templates to typed registry values.
2. Make speakers enforce `isinstance(seg, ScreenedSegment)`, make the
   key actually private to the screen module, take the screen timestamp
   after screening completes, and remove/measure the production
   hardcoded `screen_before_play: True`.
3. Fix `bench.py evaluate()` ordering, count delegate steps from
   registry/dispatch events, fix the p90 statistic, and implement the
   Tier 2 factor gates. Then file three full Tier 1 runs + Tier 2.
4. Decide invariant 6 honestly: either confirm all SIDE_EFFECTING
   dispatches via the closed set, or amend the invariant in the spec —
   and give confirms an expiry that runs whether or not the prompt was
   played, with a measured (not constant) closed-set confidence.
5. Mark held results announced only for segments actually played; make
   `held_unannounced` honour `delivered_mono` (or set `announced` in
   the completion-after-cancel write).
6. Wire guard recovery + recheck into the production speech path or
   re-run Step 6 against the configuration that actually ships;
   publish the production `p_status` quality/latency numbers.
7. Stage one complete, non-divergent copy of the lane (all modules,
   one tree) and start/verify the talker server in the run scripts
   before any flag flip.
