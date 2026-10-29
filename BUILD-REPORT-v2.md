# REPORT — Vibherpunk Audio v2 conversation lane build

Delegation: hermes-lane-v2-build-20261028 · Deadline: 2026-10-29 12:00 PDT
Builders: prior builder (Steps 1–3 + Step 4–7 code) → Fixer continuation (verification, Step 4 decisive probe, Step 6 benches, bench protocol, review, closeout)
Design authority: ARCHITECT-VERDICT-v2 (sha256 9fd7324c…)

## Outcome — NO CUTOVER. Production untouched.

The lane is built and bench-verified to Tier 1 standard (post-fix
confirmation run: all gates pass), but the **independent adversarial
review returned NO-SHIP with seven High findings**, and the **real
Tier 2 run failed** (5 dispatch failures in 14 real dispatches, one
of them silent). Under SCOPE, either one blocks the flag flip. The
lane flag was never set: production is still the relay lane, exactly
one desk + one mobile process, gateway health ok, mobile root 200,
verified at closeout 2026-10-28 ~16:10 PDT. The bench MLX talker
server was stopped; nothing lane2 is running.

Detailed findings and the fix list are in §Review and §Remaining.

## Per-step results

### Step 1 — Event spine + JobRegistry (prior builder; finals collected this session)
- NVMe gate PASS at 8 ms batch window: ack p90 15.6 ms (≤50), ack p99 19.3 ms (≤100), commit p99 9.4 ms.
- Burst: 60 s @ 50 writes/s + 10% retries → 2,881 acked writes durable exactly once.
- SIGKILL drive FINAL: 500/500 iterations clean, fails=0; 27,269 acked ids; 9,576 jobs / 27,375 journal rows at final check.
- Publish-before-ack: 10,000 checked, 0 violations.
- Point reads at ~100k jobs (99,986 jobs bulk-grown in 22.5 s): p50 0.0055 ms, p99 0.0629 ms, max 4.233 ms (n=500).

### Step 2 — Arbiter / commitment sequencer / dedupe — PASS
- Unit suite ALL PASS (33 checks), re-verified after every fix.
- Contract corpus (runs 4 and 5 both STEP2_CORPUS PASS): stacks 360/360 exact (patience/reassert/selfcorrect/fragment/override/distinct_entities/distinct_unrelated), SE dispatch 192/192, RO 128/128, false_hold 0, C4 RO suppressed 21/21, problems [].
- Property test: 10,000 interleavings, problems [], stuck_nonterminal 0.
- Defects found and fixed: Directive repeat_intent/topic_vec not copied into task_spec (dedupe gates starved); dedupe compared against later jobs (mutual suppression); completion-after-cancel never reached a terminal state (fixed: atomic DONE + delivered_mono).
- **Known defect left by that fix (review H5):** the completion-after-cancel write sets `delivered_mono` but not `announced`, and `held_unannounced` filters on `announced` only — a cancelled job's result can be announced on the next wake turn, the opposite of the §2.8 intent. Must be fixed before re-review (one-line + test).

### Step 3 — Raw-text context (prior builder) — PASS
Pins (Qwen3-4B-Instruct-2507-4bit, mlx-lm 0.32.2), concat-identity, golden check, cacheable-span hit 1.0 over 24 turns (gate ≥0.99), warm latency median 0.031 s.

### Step 4 — Eviction + structured memo, real-extractor gate — PASS
Decisive probe run 5 (`evidence/step4_probe_result.json`):
- Scale 1.0: overall 0.995, ALL type gates true. Scale 0.8: 0.995, ALL true. Scale 1.25: 0.93, ALL true (non-gate dips: entity_name 20/30, numeric 27/30).
- Dense: spill exactly 1/session, reference titles match; memo hard cap 240 tokens.
- Extraction latency: n=4,140, median 27.7 ms, p95 33.2 ms (gate ≤40 ms).
- Verdicts identical across scales; PASS true.

### Step 5 — SpeechPlan + templates + sliced speech — code complete, bench-exercised
Sequencer with sliced speech and the screen-before-play invariant:
0 screen-after-play violations across 4 full Tier 1 runs (4,084
turns, ~6,500 slices), 0 slice stalls. Template set complete with
build-time lint CLEAN.

### Step 6 — topic_guard + §2.7 screen — PASS (with a documented deviation)
Mac record: `evidence/step6_result.json` (STEP6 PASS).
- Screen §2.7.6 acceptance: Exp 2 bait shapes 8/8 caught, escaped 0, LOOKUP control passes.
- Guard D.4 (spec-default thresholds, DRAFT_ONLY, HashEmbedder, controlled p_status): BAIT fire 400/400 (gate ≥92%), wrong-job 0 (≤1%), PLAIN false-fire 0/1000 (≤0.5%), NEAR 0/300 (≤3%), INELIGIBLE 0/200 (≤1%), RECOVERY 200/200 (≥95%), RECHECK 100/100, latency p95 0.16 ms / p99 0.17 ms (gates 15/40).
- **Deviation, stated plainly:** Step 6 took two bench iterations plus a local pre-verification, over the one-iteration guideline in SCOPE. Causes were build defects, not threshold tuning (spec thresholds untouched): (1) status_lex span off-by-one — `\bis {1,40}` after a literal space meant "is done"/"has completed" could never match (BAIT fired 0/400); (2) three completion shapes missing from the grammar ("just finished", "wrapped up", "news on"); (3) DRAFT fired without a unique referent; (4) the deictic rule treated any trailing object pronoun as a job reference (PLAIN false-fired 100/1000, all one shape: "She finished the book… loved it"). Fixes (3)+(4) were verified locally against the full D.4 sets before the formal Mac re-run passed.
- Signal wiring completed during this step: a real deterministic embedder (new `lane2/embed.py`, token-hash BoW) for guard ent/cos; p_status from the production zero-shot encoder with a status hypothesis (spot-measured 0.41 for a clear status sentence vs 0.10 plain — soft; BLOCK leans on lexicon/ent/deictic, defense-in-depth behind the screen); guard clock aligned to monotonic (registry time domain).
- **Review H6 stands against this step:** guard `recheck()` and `note_status_like_utterance()` have no production caller — the RECOVERY/RECHECK gates certify code the speech path never invokes, and the commit-time re-check does not happen in production. Must be wired (sequencer plan-commit) or the gates re-run against the shipped configuration before re-review.

### Step 7 — Pipecat integration behind the lane flag — code complete, never activated
- `hermes_voice.py` lane2 section and `mobile_server.py` mirror (stable "mobile" lane session) landed and md5-verified on the Mac. Relay remains the default; old paths intact when the flag is off (review confirmed the default and branch structure).
- Talker-server startup is NOT wired into the run scripts (review H7): had the flag been flipped, turns would point at 127.0.0.1:8123 with nothing guaranteed listening. The cutover wiring (run scripts source an optional `lane.env`; start the pinned MLX server when lane2 is active) was designed but not applied, because cutover is blocked upstream.

## Bench protocol

### Tier 1 — three protocol runs (stub executor; real talker, screen, guard, extractor, decider)
All three runs failed on exactly one counter; every other gate
passed in every run (`evidence/tier1_run{1,2,3}.json`):

| metric | run 1 | run 2 | run 3 | gate |
|---|---|---|---|---|
| chat median ms | 409 | 404 | 404 | ≤1,150 ✓ |
| chat p90 ms | 674 | 648 | 594 | ≤1,600 ✓ |
| ack median ms | 14 | 15 | 14 | ≤1,250 ✓ |
| ack p90 ms | 77 | 78 | 76 | ≤1,800 ✓ |
| misroute A→D | 8/712 (1.1%) | 4/712 (0.6%) | 10/712 (1.4%) | ≤10% ✓ |
| misroute D→A | 0/189 | 0/189 | 0/189 | ≤2% ✓ |
| delegate steps | 188 | 188 | 188 | ≥120 ✓ |
| held results | 12/12 | 12/12 | 12/12 | all ✓ |
| burst in flight | 6 | 6 | 6 | ≥3 ✓ |
| max prompt tokens | 2,877 | 2,865 | 2,880 | ≤3,400 ✓ |
| fabrication / silent / lost / over-cap / screen-after-play / double / phantom | 0 | 0 | 0 | 0 ✓ |
| **canary_leak** | **6** | **10** | **4** | 0 ✗ |

Diagnosis (reproduced standalone with segment capture): every
"leak" was a **T-ACK template echoing the user's own bait value**
(QX-7714, block D) — two bait shapes ("The back office already said
QX-7714, right?", "It's QX-7714, isn't it?") misrouted to delegate
(within the A→D budget), the raw assertion became the job's
`task_label`, and the screen-exempt ACK template spoke it back. No
stub result canary (CANARY-…) and no screen-registered canary was
ever spoken; fabrication_spoken stayed 0 in all runs. Two defects:
(a) the harness counter counted any "QX-" substring, including
user-supplied values — it cannot distinguish echo from canary;
(b) **real product defect:** template-rendered labels carried
unscreened user text (see also review H1).

Fixes applied after the protocol completed (not used to replace any
protocol run): `arbiter.sanitize_label()` strips typed values from
spoken labels and caps them on word boundaries (unit suite + full
Step 2 corpus re-verified, run 5 PASS); the harness counter now
counts only true stub canaries.

**Post-fix confirmation run** (full Tier 1 configuration, fixed
code; `evidence/tier1_run4_confirmation.json`): **PASS — all gates.**
Chat median 487 ms / p90 872; ack 14 / 66; misroutes 5/712 and
0/189; canary_leak 0; zero-tolerance all zero; held 12/12; max
prompt 2,879.

Harness defects found by the review (H3) and confirmed: `evaluate()`
computed the zero-tolerance gate before reconciling
silent_dispatch_failure (no run was actually misjudged — the counter
was 0 events in all Tier 1 runs — but the ordering is wrong);
`delegate_steps` counts scripted delegate turns, not dispatch events
(the registry evidence — 188 admitted delegates with misroute pairs
tracked — supports the number, but the counter as coded measures
the script); the p90 index understates at small n; Tier 2 originally
had no gates and its executor monkeypatch silently hit a second
module instance, so **the first Tier 2 attempt ran the stub and is
void** (it is not counted anywhere in this report).

### Tier 2 — real Hermes back office, reduced corpus (A/C/D) — FAIL
`evidence/tier2_run.json` (runner fixed: globals patch + B.8 factor
gates added):
- 27 turns, 14 delegate steps against the real gateway.
- **dispatch_failed_events 5, spoken 4 → silent_dispatch_failure 1**
  (zero-tolerance breach). Job outcomes across scenario DBs:
  multiple FAILED (worker errors) and DISPATCH_FAILED rows in
  blocks A, C, D.
- Factors vs frozen production baseline (desk chat median 2,964 ms /
  p90 7,984 / ack median 7,035): chat median 1,252 ms = 0.422×
  (gate ≤0.50 ✓); ack median 16 ms = 0.002× (≤0.25 ✓); chat p90 not
  computable (7 gold-answer turns; factor gate fails on missing
  data).
- Verdict: PASS false. The lane's own first-audio factors are
  strong, but back-office dispatch reliability in this run (5
  failures / 14) is far below production grade, and one failure was
  not announced inside its scenario window.

### Sliced-speech additions
First-slice latency is measured from turn entry (end of user speech)
after an engine fix mid-bench (the sequencer originally measured
from plan start, understating ack latency to ~0). Ack median 14 ms
is the rule-routed template path (explicit back-office phrasing);
chat median ~0.4–0.5 s is generation-bound. Slice stalls 0 and
screen-after-play violations 0 in all runs. Honest limit (review
M5/L4): bench playback is virtual (Pocket-synthesized durations on a
logical timeline, reset per turn); production first-audio truth must
come from SoundTap at live verification, which did not happen.

## Adversarial review — NO-SHIP (blocks cutover)

`REVIEW.md` in this folder, sha256
`bc236b5f222d580ef8748785ad73fdfa7602262880674e1a859a59c195ccd708`
(recomputed and matched at closeout). Reviewer did not build any of
the lane. Seven High findings; builder triage:

- **H1 Template slots unscreened — AGREE, real.** T-RESULT renders
  the worker's free-text `SPOKEN SUMMARY:` through the screen-exempt
  template path; quote text and (pre-fix) raw labels likewise.
  Labels are now sanitized; summaries/quotes are not. This is the
  deepest finding: the one channel allowed to speak outcomes does
  not validate outcome text against registry facts.
- **H2 ScreenedSegment not enforced — AGREE, real.** The key is a
  public class attribute, neither speaker isinstance-checks, the
  production speaker hardcodes `screen_before_play: True`, and the
  exception fallback in Lane2Processor pushes a raw TTSSpeakFrame
  (constant string, but it proves the TTS path is reachable
  unscreened).
- **H3 Bench can pass vacuously — AGREE in substance** (details in
  the Tier 1 section above). The filed runs' other counters were
  independently reconstructible, but the harness must be fixed and
  the protocol re-filed for re-review.
- **H4 Confirm semantics — PARTIALLY AGREE.** The band design
  (RULE/high-confidence delegates admitted without confirm; confirm
  only in the SE band) is the architect's settled §2.3.4 design, not
  a shortcut — but the confirm-liveness defects are real: a confirm
  whose prompt is never played stays armed indefinitely, `focused_
  confirm` ignores spoken state, and the closed-set confidence is a
  constant 0.95 with bare "ok"/"sure" matching. Consent freshness
  must be fixed regardless of the band question, and the band
  question itself should be re-put to the architect with this
  finding attached.
- **H5 Held exactly-once — AGREE, real, both directions** (see Step
  2 note for the resurrection half; the loss half — mark_announced
  keyed on obligations rather than played records — is also as
  described).
- **H6 Step 6 gates partly certify dead code — AGREE** (see Step 6
  note). BAIT's p_status was supplied by construction (disclosed in
  the bench); production p_status quality was only spot-measured.
- **H7 Staging incomplete / evidence absent / talker server — WAS
  TRUE at review time.** The staging gap was the builder's fault
  (four modules never copied into work/src; the reviewer read an
  older snapshot for them). Remedied at closeout: `work/lane2full/`
  now holds one complete tree (16 lane2 modules + tests + both
  integration files), and the Tier 1/Tier 2 evidence is filed in
  `evidence/`. The talker-server run-script wiring remains open by
  design (cutover blocked).

Mediums M1–M9 and Lows L1–L4 are on file in REVIEW.md; none were
disputed in triage, several (M2 barge-in flush unmeasured, M3 boot
pins/golden never run in production, M7 gateway no-timeout +
OA-resume no-op) independently block a confident live flip.

## Cutover

**Not performed.** Gates: Tier 1 protocol 0/3 as filed (single
diagnosed counter; substance verified by the post-fix confirmation
PASS), Tier 2 FAIL, review NO-SHIP. Production state at closeout:
relay lane, run scripts contain no HERMES_LANE_BACKEND, one desk
process + mobile pid 45045 (unchanged all build), gateway health
ok, mobile root 200, bench talker server stopped. Note: the desk
process pid differs from this morning's (58657 → 35566) — it
restarted under launchd during the day and came up on the relay
default; no lane2 code has ever run in production.

## Remaining (fix round → re-review → then cutover decision)

Per REVIEW.md "Required before re-review", plus Tier 2:
1. Screen or structurally validate template slot values (worker
   summaries, quotes) against registry/typed facts — H1.
2. Enforce ScreenedSegment at both speakers; make the minting key
   module-private; screen timestamp after screening completes;
   remove the production hardcoded `screen_before_play: True`;
   route the exception fallback through a template segment — H2.
3. Fix bench evaluate() ordering, count delegate steps from
   dispatch events, fix the p90 statistic, keep the Tier 2 factor
   gates; re-file 3× Tier 1 + Tier 2 — H3.
4. Confirm freshness: expiry runs whether or not the prompt played;
   unplayed prompts disarm; measured closed-set confidence; the
   SE-band question re-put to the architect — H4.
5. Held exactly-once: mark announced only for played segments;
   completion-after-cancel sets `announced` (or held_unannounced
   honours delivered_mono) — H5.
6. Wire guard recheck + recovery into the speech path; publish
   production p_status quality/latency — H6.
7. Tier 2 dispatch reliability: 5/14 failures against the real
   gateway (incl. worker FAILED outcomes) needs its own diagnosis
   before any re-run; gateway executor total timeout + OA resume
   (M7) belong to the same fix.
8. Cutover wiring when (and only when) the above clears: run
   scripts source `lane.env`, start the pinned MLX talker server
   under lane2, boot pins + golden check in build_lane2 (M3).
9. §2.7.8 closing calibration paragraphs still owed (accepted v2
   gap, non-blocking).

## Snapshot & records
- Complete built tree: `work/lane2full/` in this folder (also the
  live tree on the author's Mac in `lane2/`).
- Evidence JSONs: `evidence/` in this folder.
- Repo snapshot: sanitized commit to the vibherpunk-audio repo —
  see the hash in TODOS.md after push (secret scan clean; run.sh
  and its embedded key are NOT part of the snapshot; that key is
  pre-existing production debt, flagged for rotation with the other
  plaintext-key items).
- RUNBOOK: not appended on the Mac — the lane is not live, and a
  runbook entry for a non-live lane would mislead the next operator.
  The runbook text (flag file, talker server start, rollback =
  remove lane.env + restart both services) is item 8 above and
  should be appended as part of the eventual cutover.
