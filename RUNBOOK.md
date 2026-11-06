# Hermes Live Voice — RUNBOOK (Phase 1: two-tier brain)

A Pipecat front-end that lets you TALK to Hermes: a fast conversation lane
answers immediately, and anything needing real work is handed to the full
Hermes agent in the background — its result is spoken aloud when it lands.

    You ──"hey clanker" / "hey viber"──► wake gate opens ──► you talk
    mic → Whisper (local) → conversation lane (fast model, no tools of its
    own except delegating) → Pocket TTS (eve) → speakers
    Work requests → delegate_to_hermes → full Hermes agent (default
    profile, all tools/memory) → spoken summary injected when done

## Start / stop

It starts at login via launchd (`ai.hermes.voice`). Manual control:

```
# stop
launchctl bootout gui/$(id -u)/ai.hermes.voice
# start
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.voice.plist
# one-off run in a Terminal (best for the first mic-permission grant)
cd /Users/ai/workspace/hermes-voice && ./run.sh
```

If macOS asks for microphone permission, click **Allow**. If the log shows
"MIC SILENT" (see Logs), the launchd process has no mic permission yet —
run `./run.sh` once from Terminal, Allow, then restart the launchd agent.

## Using it

- Say **"hey clanker"** or **"hey viber"**, then talk. The gate stays
  open for follow-ups for ~20 seconds after the last activity (2 minutes
  while a Hermes job is running), then re-arms — say either phrase again.
- Chat and quick questions are answered by the lane itself, out loud.
- Ask for real work ("have Hermes check…", anything needing memory, files,
  tools, the web) and the lane hands it off, says so, and keeps talking
  with you. When Hermes finishes, its summary is spoken — interrupting it
  works like interrupting any reply: just start talking.
- Ask "is that done yet?" any time — the lane checks the real job state.
- The lane will NOT guess a result before Hermes reports back. If you push
  ("just tell me what it found"), it holds: still working means still
  working.

Known interaction: Hermes' own desktop wake-word listener listens for
"hey hermes" — its own phrase; it does NOT answer to clanker/viber, and
this front-end no longer listens for "hey hermes", so the two no longer
collide. (Hermes' listener was already disabled Oct 27 in its settings,
config `wake_word.enabled`.)

## Wake word (engine, phrases, rollback)

Since Oct 27, 2026 the desk gate runs the **sherpa-onnx** open-vocabulary
keyword spotter (`HERMES_WAKE_ENGINE=sherpa` in run.sh) listening for
**"hey clanker"** and **"hey viber"** — either phrase wakes it, and the
fire log line / `wake_fire` metrics event name the phrase that matched.
Sherpa needs no per-phrase trained model: phrases are typed and tokenized
at startup. Registered keywords: HEY CLANKER, HEY VIBER, and HEY VIBAR (a
phonetic tokenization alias for "viber" — both count as "hey viber").

- Model: `models/sherpa-kws/` in this project (zipformer KWS, ~19 MB),
  served by the `sherpa-onnx` package in `.venv`.
- Sensitivity: `HERMES_WAKE_SENSITIVITY` (default 0.6) maps to the
  spotter threshold as `0.05 + 0.4 * s` (= 0.29 at default). Raise it if
  the gate false-fires on room conversation; lower it if it misses you.
- Phrases: `HERMES_WAKE_PHRASES` (default `hey clanker,hey viber`).

**Viber is the weaker phrase.** In headless tests with synthesized
voices it fired 0–1 of 3 renditions (clanker: 3 of 3 with the phrase at
stream start, 2 of 3 with a running stream; negatives 0 of 6
throughout). Treat "hey clanker" as the reliable wake; a trained TFLite
model is the planned fix for Viber. Your real voice in the real room is
the deciding test (see PENDING-ADAM-TEST).

**Rollback** — restore the original trained-model gate ("hey hermes",
pyopen-wakeword + `hey_hermes.tflite`, untouched on disk): in `run.sh`
change `export HERMES_WAKE_ENGINE=sherpa` to
`export HERMES_WAKE_ENGINE=oww` (or delete the line — `oww` is the code
default), then restart:

```
launchctl bootout gui/$(id -u)/ai.hermes.voice
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.voice.plist
```

The log should then show `wake gate armed: engine=oww model=hey_hermes`.

## Putting it to sleep (spoken)

While a conversation window is open, say either phrase as your whole
utterance (a polite lead-in or "that's all, ..." in front is fine):

- **"go to sleep"**
- **"stop listening"**

Hermes answers "Going to sleep — say hey Clanker when you need me." and
the wake gate re-arms **immediately** — no waiting out the follow-up
window. Matching is deliberately conservative: a request that merely
contains the words ("stop listening to the radio and check the
weather", "tell my kid to go to sleep") does NOT put it to sleep, and
the phrase is never treated as conversation or handed to Hermes as a
task.

**Held results:** if a Hermes job finishes while the gate is asleep,
its result is NOT announced into the quiet room. It is held, and on
your next "hey clanker" (or "hey viber") it is announced first — "While you were asleep,
Hermes finished: ..." — before anything else, in completion order if
there are several. Holds and announcements are logged
(`held_result_held` / `held_result_announced` in the metrics log).

Sleep is the soft off. The hard off is the launchctl stop under
"Start / stop" above — that kills the bot outright, sleeping or not.

## Logs

- `logs/metrics.jsonl` — per-turn metrics (first-audio splits, lane token
  counts), dispatch/completion/failure events, wake events.
- `logs/launchd.err.log` / `logs/launchd.out.log` — the running bot's log
  (loguru writes to stderr, so most of it lands in the .err file).

## Knobs (environment, set in run.sh or the plist)

- `LANE_MODEL` — lane model (default `glm-5.3-flash`, the measured-fastest
  on the Zen Go route; `deepseek-v4.1-flash` is the slower fallback).
- `HERMES_WAKE=off` — disable the wake gate (always listening).
- `HERMES_WAKE_ENGINE` — `sherpa` (current, set in run.sh) or `oww`
  (rollback to the trained "hey hermes" model). See "Wake word" above.
- `HERMES_WAKE_PHRASES` / `HERMES_WAKE_SENSITIVITY` — sherpa phrases and
  sensitivity (defaults `hey clanker,hey viber` / 0.6).
- `HERMES_STT=faster` — faster-whisper instead of MLX Whisper (slower).
- `HERMES_VOICE` — Pocket voice (default `eve`).
- `HERMES_FOLLOWUP_SECS` — follow-up window before the gate re-arms.

## Phase 0 comparison

The Phase 0 single-tier bot (talk straight to the agent, trimmed voice
profile) is preserved: `./runbot_phase0.sh` (its runbook is
`RUNBOOK-phase0.md`). Stop the Phase 1 agent first — they share the mic
and speakers.

## PENDING-ADAM-TEST (live items the headless proofs cannot settle)

1. **Barge-in feel** — ask for a long story, talk over it mid-sentence.
   Headless simulation in Phase 0 said the stop path drains in ~120 ms
   after detection; the perceived figure is yours to judge.
2. **Echo case** — one full conversation on speakers, no headphones.
3. **First-audio feel** — headless median for chat turns measured
   1.9 s end-of-utterance → first sound (the ≤500 ms gate was NOT met;
   the bottleneck is the lane provider's time-to-first-token, which
   varies 0.5–3.5 s). Judge whether it feels conversational enough.
4. **Wake word at the desk** — say "hey clanker" and "hey viber" in
   the real room. Headless tests (synthesized voices) say clanker is the
   reliable one and viber is hit-or-miss; your voice is the real test.
   (The old "hey hermes" gate scored 0.967 on its test phrase — that
   engine is the `oww` rollback above.)

## Troubleshooting

- Bot process: `launchctl print gui/$(id -u)/ai.hermes.voice | grep -E "state|pid"`,
  `pgrep -fl hermes_voice`.
- Gateway down? The front-end says so out loud. Gateway health:
  `curl -s http://127.0.0.1:8642/health`.
- Nothing happens when you speak: check the gate actually fired
  (`wake_fire` in metrics.jsonl), then the mic warning above.

## MOBILE (phone) — the phone version of live voice

What it is: `mobile_server.py` serves a small PWA (`client/index.html`) from
FastAPI on 127.0.0.1:8917. The phone loads that page over HTTPS, taps "Tap to
talk", and the browser streams mic audio to the Mac over WebRTC (SmallWebRTC).
Same two-tier pipeline as the desk version (local MLX Whisper -> fast lane ->
Pocket TTS), so the Mac's speakers/mic are never opened. Wake word is OFF on
mobile: the connect tap is the gate.

Start / stop (launchd, survives reboot):

    launchctl print gui/$(id -u)/ai.hermes.voice-mobile | grep "state ="
    launchctl kickstart -k gui/$(id -u)/ai.hermes.voice-mobile     # restart
    launchctl bootout gui/$(id -u)/ai.hermes.voice-mobile          # stop

Phone URL (Tailscale HTTPS, tailnet only):

    https://ais-mac-studio.tail3b031e.ts.net:8443/?token=<contents of ~/.hermes/mobile-voice-token>

`tailscale serve` maps :8443 -> http://127.0.0.1:8917 (see `tailscale serve
status`). The token lives in `~/.hermes/mobile-voice-token` (0600); the page
stores it in localStorage and strips it from the URL. Wrong/absent token = 403.

### Gotchas proven the hard way

- **Run uvicorn with the asyncio loop, not uvloop.** With uvloop, an inbound
  offer's smallwebrtc fetch of the ICE-gathering batch can reuse a dead
  aiohttp connection on a second connect: `POST /api/offer` 500s, the page
  says "connected", and no audio ever arrives. `loop="asyncio"` fixed it.
- **One server process only.** A stray `run_mobile.sh` outside launchd holds
  port 8917 and the launchd job then crash-loops on "address already in use".
- **Headless Chrome's fake mic is silent on this host** (peak 0.000 regardless
  of WAV format/rate), so `tests/test_mobile_web.py` always times out at
  "first remote audio" here. That is a harness limit, not a server fault.

### Tests (run with `env -u PYTHONPATH`, the shell exports a py3.14 path)

- `tests/stt_synthetic.py` — pushes the test WAV straight into the mobile
  pipeline: proves STT hears phone-style input and the lane answers.
- `tests/say_probe.py` / `tests/record_reply.py` — live WebRTC session;
  `/api/_test_say` injects a text turn (loopback-only + token) so the TTS ->
  WebRTC -> phone `<audio>` leg is provable without a mic.
- `tests/transcribe.py <file>` — local Whisper on a recording, to confirm
  what the phone actually heard.
## Routed local lane (Phase 2, built 2026-10-27)

The conversation lane has two backends, selected by `HERMES_LANE_BACKEND`
in `run.sh` / `run_mobile.sh`:

- `relay` (DEFAULT, production) — the OpenCode Zen Go route
  (`glm-5.3-flash`), the original two-tool lane prompt.
- `local` — a fast non-reasoning model served on this machine by mlx-lm
  (`mlx-community/Qwen3-4B-Instruct-2507-4bit` on `127.0.0.1:8123`),
  running the routed prompt with three exits: answer now /
  `quick_lookup` (bounded ~600 ms search of the Hermes default
  profile's local memory files, sourced passages only; a miss escalates
  to delegation, never a guess) / delegate up (the unchanged
  `delegate_to_hermes` + `check_delegated_jobs`).

Routing mechanism for the local backend, `HERMES_ROUTING`:

- `inband` — the lane model picks the exit inside its single call.
- `encoder` — a hybrid decider (deterministic rules + a ModernBERT
  zero-shot encoder pass, ~40–90 ms warm, no generated tokens) fixes
  the exit on the final transcript and hands it to the lane as an
  authoritative system note, including a HOLD note for bait-shaped
  turns while a job is running. GLiNER span extraction pins names on
  partial transcripts for the anticipator (pre-run lookups and
  pre-drafted delegations during the user's turn; worker jobs fire
  only on the final transcript; speculation is never spoken).

**Status: NO-SHIP (2026-10-27 bench).** The local lane stays behind the
flag. Chat latency passes easily (median 0.40 s vs the relay's 1.32 s,
same harness) and barge-in is unchanged (155 ms vs 162 ms simulated),
but the delegation-ack median (1,508 ms vs the ≤1,250 ms gate) and the
100% discipline hard rules did not pass in any benched configuration —
the best local build still broke the lookup→escalate tool chain on
3 of 12 tool-required turns. See the delegation REPORT
(`hermes-routed-lane-20261027`) for the full table. Production remains
on `relay`.

Model server (only needed when testing the local backend):

```
cd /Users/ai/workspace/hermes-voice
.venv/bin/python -m mlx_lm server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --host 127.0.0.1 --port 8123
```

A launchd agent (`ai.hermes.lane-model`) is the intended permanent
form if the lane ever ships; it is NOT installed while the verdict is
NO-SHIP.

Rollback / switching: set `HERMES_LANE_BACKEND=relay` (or delete the
line — `relay` is the code default) in both run scripts and restart
both services:

```
launchctl kickstart -k gui/$(id -u)/ai.hermes.voice
launchctl kickstart -k gui/$(id -u)/ai.hermes.voice-mobile
```

Bench harnesses: `tests/test_routed_lane.py` (full routed suite),
`tests/runbench_baseline.sh` (relay comparator),
`tests/runbench_local.sh` (inband), `tests/runbench_local_enc.sh` /
`runbench_local_enc2.sh` (encoder), `tests/test_bargein_routed.py` +
`tests/runbarge_routed.sh` (barge-in regression),
`tests/bench_classifier.py` / `tests/bench_router.py` (decider
standalone), `tests/bench_local_lane.py` (raw-API candidate probe).
All runners extract keys from `~/.hermes/.env` at runtime.

## Cloud talker (HERMES_LANE_BACKEND=cloud) — round 2, 2026-10-28

Status: built and benched; **production stays on `relay`** (see the
delegation report `hermes-cloud-talker-20261028/REPORT.md` for the gate
table). The flag is the whole switch — no other config changes.

What `cloud` is: the routed configuration (routed system prompt, encoder
routing + GLiNER pins, anticipator, fused lookup) with the talker =
GLM-5.3-Flash on the OpenCode Zen Go subscription route — the same route
and key (`OPENCODE_GO_API_KEY` from `~/.hermes/.env`) the relay lane
already uses. Only conversation text leaves the machine; all audio
stays local, as before. Extra env: `HERMES_FUSED_LOOKUP=on` (the fused
tool) and `HERMES_LANE_CONTEXT_TOKEN_BUDGET` (default 2200).

Pieces:
- `CloudLaneLLMService` — routed calls on the Zen route. Context cap
  applied before every call (token budget on carried history; oversized
  worker summaries truncated; oldest turn-units dropped whole). If a
  routed call fails before its first chunk, the turn degrades to the
  relay configuration (production prompt, the two original tools,
  routed-only history sanitized out) and a `lane_fallback` event is
  recorded in the session metrics. Post-first-chunk failures are never
  retried (no double-speak); they reach the SpokenFailureAnnouncer.
  Note: the OpenAI SDK retries a failing call 3× before the fallback
  engages.
- Fused `lookup_or_delegate` (replaces `quick_lookup` when
  `HERMES_FUSED_LOOKUP=on`) — one tool: query + self-contained task.
  The handler runs the bounded lookup (or consumes the anticipator's
  speculative result); a hit returns the passages, a miss dispatches
  the task itself. A hit counts only if a candidate contains one of
  the query's discriminating content words (`_responsive_hit`) —
  `quick_lookup`'s `found` alone means "passages cleared a loose
  keyword score," which returned junk for e.g. "wife name". Built
  because the two-call lookup→escalate chain failed ~1/15 even on
  GLM with verbatim round-1 prompts (API diagnostic, delegation
  report §0).

To enable (both services): set `HERMES_LANE_BACKEND=cloud` and
`HERMES_FUSED_LOOKUP=on` in the services' environment (run.sh /
run_mobile.sh or the launchd plists), restart `ai.hermes.voice` and
`ai.hermes.voice-mobile`, verify exactly one process each and the
mobile page answering 200. Rollback: set `HERMES_LANE_BACKEND=relay`
(or remove it), restart — the relay path is byte-for-byte the
pre-round-2 behavior.

Known limits (measured 2026-10-28): the Zen route's time-to-first-byte
swings ~20× across the day (434 ms best window; 8–10 s medians in the
worst) — first-audio latency on `cloud` is weather-bound and was above
the 1,150/1,250 ms gates in the bench window, as was the relay
incumbent measured the same hour (chat 4,407 / ack 7,507). Watch
`lane_ttft_ms` and `lane_prompt_tokens` in the session metrics:
prompt tokens should stay ≲3.5k however long the conversation runs.
Bench: `tests/runbench_cloud.sh` (per-run `HERMES_VOICE_LOGDIR`);
fallback proof: `test_fallback.py` + the selective-failure proxy in
the delegation work folder.

## Local fused lane (HERMES_LANE_BACKEND=local + HERMES_FUSED_LOOKUP=on) — round 3, 2026-10-28

Status: built and benched; **production stays on `relay`** (see the
delegation report `hermes-local-fused-20261028/REPORT.md` for the gate
table). The flags are the whole switch — no other config changes.

What changed vs rounds 1–2: the fused `lookup_or_delegate` tool, its
sufficiency screen, the fused prompt and directive turned out to be
backend-agnostic already (registered for every routed backend), so
the local lane gets them by flag. The genuinely new code is the
context cap on the local backend: `LocalLaneLLMService` applies the
2,200-token history cap before each call, and the `ContextTrimmer`
response-end cap now covers all routed backends (was cloud-only).

Also added, then left in (inert off the routed backends): a decider
rule (`rule-pressure`) that routes short follow-ups pressing on a
running job ("take your time", "I want the real text from Hermes
itself") to HOLD while jobs are pending. Bench finding: it removed
result fabrication on delegate turns, but on multi-message steps the
newest-directive-only context trimming let a HOLD fragment override
an undispatched DELEGATE from the same step, dropping two
delegations. Do not enable this lane until directive arbitration
(one arbitrated directive per generation) replaces per-message
directives — see the report's "next move" section.

Known limits (measured 2026-10-28): chat first-audio 442 ms median
passes the 1,150 ms gate and beat the same-window relay incumbent
(4,733 ms) by ~10×; the ack gate (1,250 ms) is out of reach in the
delegate region (5.5–5.6 s on both builds) because the context cap
rewrites the history prefix every turn, defeating the MLX server's
prefix cache — delegate-region TTFT runs 2.5–5.5 s vs ~200 ms in the
chat region. Barge-in 162 ms median. Bench:
`tests/runbench_local_fused.sh` (per-run `HERMES_VOICE_LOGDIR`),
baseline `tests/runbench_localfused_base.sh`.

To enable (both services, only after the gates pass in a future
round): set `HERMES_LANE_BACKEND=local`, `HERMES_ROUTING=encoder`,
`HERMES_FUSED_LOOKUP=on`, start the MLX server (model
`mlx-community/Qwen3-4B-Instruct-2507-4bit` on 127.0.0.1:8123),
restart `ai.hermes.voice` and `ai.hermes.voice-mobile`, verify one
process each and mobile 200. Rollback: `HERMES_LANE_BACKEND=relay`
(or remove), restart.
