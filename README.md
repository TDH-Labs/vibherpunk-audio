# Vibherpunk Audio

A live voice front-end that gives any agent harness ears and a voice. You talk
out loud; a fast conversation lane answers immediately, and anything needing
real work is delegated to the full agent in the background — its result is
spoken aloud when it lands.

It is a standalone program, not part of any harness: it runs as its own
services and talks to the agent through the harness's ordinary API. The
worker contract is one seam — an HTTP endpoint that takes a self-contained
task and returns a result ending in a spoken summary — so the brain behind
the voice can be swapped without touching the pipeline. The reference worker
adapter in this snapshot targets a Hermes gateway; the lane is any
OpenAI-compatible streaming endpoint.

The design borrows from multi-token prediction and speculative decoding: the
fast lane starts the low-risk social part of the reply while the large agent
does the authoritative work in parallel. Unlike draft tokens, spoken words
cannot be discarded after the fact, so the lane never speculates about facts
or results. Architecturally, the voice model is the front door and large
agents are its hands — tools, memory, and actions. Perception workers can
serve as its eyes through the same delegation seam; no camera worker is
connected in this snapshot. Reachy Mini is the physical reference point for
that pattern: an embodied voice surface with a camera, expressive movement,
pluggable speech-to-speech backends, and background tools.

> **Snapshot scope:** this commit brings the snapshot current with the
> source deployment as of 2026-10-27: the desk system (Phases 0–1), the
> mobile transport (`mobile_server.py` + `client/` — WebRTC with a private
> TURN relay, service-worker and wake-lock fixes, and the PWA access-token
> panel), the spoken sleep command with held results, the Sherpa
> open-vocabulary wake spotter ("hey clanker" / "hey viber"; the trained
> model remains as a one-setting rollback), the Vibherpunk rename in the
> served client and the spoken lane prompt (the worker is "the back
> office"), and the Phase 2 routed local lane (below). Internal
> file/service names still carry the original `hermes-voice` working
> name; the product was renamed **Vibherpunk Audio** on 2026-10-27
> because it was never Hermes-specific.
>
> **Phase 2 — routed local lane (in this snapshot, NOT shipped):** a
> local fast model (MLX) behind `HERMES_LANE_BACKEND=relay|local`
> (default `relay`) with three exits — answer / `quick_lookup` (bounded
> local-memory search, sourced facts only) / delegate — routed either
> in-generation or by a hybrid encoder decider (`HERMES_ROUTING`), plus
> a transcript-layer anticipator. It benched chat first-audio at a
> 0.40 s median (relay: 1.32 s, same harness) but missed the
> delegation-ack gate (1.51 s vs ≤1.25 s) and the 100% discipline hard
> rules, so the flag stays on `relay` in the source deployment too.
> Full measurements: `RUNBOOK.md`, routed-lane section.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat): local mic/speaker
transport, Silero VAD + Smart Turn end-of-turn detection, local speech-to-text
(Whisper), streaming local text-to-speech (Kyutai Pocket TTS — the speech
server and voice catalog are Buzz's, extracted and run standalone), wake-word
gating (openWakeWord), and a leading-silence trimmer so replies start on the
first real sound instead of after the TTS model's baked-in pause.

## Architecture

```
mic → wake gate → Whisper STT → conversation lane → Pocket TTS → speakers
                                   │  (fast direct model call, small prompt,
                                   │   one tool: delegating)
                                   └─ delegate_to_hermes ─► full agent
                                       (gateway, all tools/memory, async)
                                       ─► spoken summary injected on completion
```

- **Tier 1 — conversation lane.** A direct streaming call to a fast model
  with a small, spoken-style system prompt. It answers chat itself and holds
  a hard discipline: it never states a result it hasn't been given, and a
  spoken promise without a real dispatch counts as a failure.
- **Tier 2 — the agent as worker.** Delegation is an async call to an
  OpenAI-compatible agent gateway (default profile: full tools + memory).
  On completion, the worker's own spoken summary is injected into the live
  conversation as an interruptible spoken turn.
- **Failure is spoken, never silent.** If the lane provider or the agent is
  unreachable, the front-end says so out loud and the conversation survives.
- `hermes_voice_phase0.py` is the earlier single-tier version (talk straight
  to the agent through a tool-trimmed profile), kept runnable for comparison.

## Measured performance (desk, October 2026)

| Stage | Time |
|---|---|
| Transcription (MLX Whisper small, warm) | 0.07–0.12 s |
| TTS first byte (Pocket, local) | 91–99 ms |
| Barge-in stop (simulated) | 155–162 ms |
| Chat reply, end of speech → first sound (relay lane) | ~1.3 s median (re-measured 2026-10-27) |
| Delegation acknowledgment (relay lane) | ~3.3 s median (re-measured) |
| Chat reply (local routed lane, not shipped) | 0.40 s median |
| Delegation acknowledgment (local routed lane, not shipped) | 1.51 s median |

For context: independent benchmarks put frontier speech-to-speech models at
0.8–1.2 s time-to-first-audio on real tasks, and most production voice agents
at 1.4–1.7 s median end-to-end. The local stages here (STT, TTS, barge-in)
run at best-in-class times at zero marginal cost; the remaining gap on the
relay route is the lane model's first token. The Phase 2 local lane closes
most of that gap on chat but is not shipped — it failed the discipline and
ack gates (see the snapshot-scope note above). Full reasoning and test
evidence: `RUNBOOK.md`.

## Running it

See `RUNBOOK.md` (Phase 1, the daily driver) — start/stop, wake behavior,
environment knobs, logs, and troubleshooting. `install.sh` sets up the
virtualenv.

## Requirements

- macOS (developed on Apple Silicon), Python 3.12, `portaudio` (for PyAudio)
- A local Pocket TTS setup (catalog voices) and a Whisper engine
  (MLX Whisper by default; faster-whisper selectable)
- An OpenAI-compatible agent gateway for the worker tier
  (`HERMES_WORKER_URL`, default `http://127.0.0.1:8642/v1`)
- A lane provider endpoint (`LANE_BASE_URL` / `LANE_MODEL`) and the API keys
  for both, supplied through the environment — `run.sh` extracts them from
  local `.env` files at runtime; no key is ever stored in this code.

## A note on this copy

This is a sanitized snapshot of a working personal deployment. Machine
specifics — absolute paths, the user's name in prompts and test fixtures,
keys — have been parameterized or replaced with placeholders (e.g.
`/path/to/hermes-voice` in the test scripts, "the user" in the lane prompt).
Behavior and measurements are unchanged. Audio fixtures (`tests/*.wav`)
from the original test runs are not included; the test scripts regenerate
what they need.

## The back office: you need an agent harness

Vibherpunk Audio is a voice front end, not an agent. The parts divide like this:

- **Whisper** hears (speech to text). **Pocket TTS** speaks. Neither does anything else.
- **The conversation lane** (default model: GLM-5.3-Flash) talks. It answers what
  can be answered on the spot and delegates everything that is real work.
- **OpenCode Go is only a model provider** — it rents access to the lane and
  worker models. It has no tools and cannot act on your computer.
- **Real work** — reading files, checking a calendar, searching memory, running
  commands, sending messages — is done by a back-office **agent harness** the
  lane delegates to.

The back office is reached through one OpenAI-shaped chat endpoint
(`HERMES_WORKER_URL`, default `http://127.0.0.1:8642/v1`). The reference build
uses the Hermes agent harness,
which runs on your computer with its own tools, memory, and skills, and reports
back when a delegated task is done; the result is then spoken aloud.

Any harness that exposes an equivalent endpoint can serve as the back office.
A bare model API cannot — it will reply with text but take no action, because
nothing behind it can touch your computer. Using a Claude- or
Antigravity-based agent as the back office would require a small bridge that
accepts the chat request, hands the task to that agent, and returns its
result; that bridge is not included in this repo.

Without a harness connected, the system still talks, but delegated requests
report that the back office is unavailable.

## License

Vibherpunk Audio is free for personal and other noncommercial use under the
PolyForm Noncommercial License 1.0.0 — see [LICENSE](LICENSE).

Commercial use of any kind is **not** covered by that license. That includes
using the software inside a business or in a product, even if the software
itself is given away for free. Commercial use requires a separate commercial
license from the copyright holder. To ask about one, open an issue on this repository.

## Current state — November 5, 2026

- Two-tier shape: a fast talking lane (DeepSeek V4.1 Flash via Open Design) up front, and the Hermes agent on the Mac Studio as the back office — shell, files, web, memory — reached through the delegate tool. Back-office work routes by effort through Hermes' delegation tiers on the Open Design provider: Flash for everyday work, DeepSeek V4 Pro for heavy work, Gemini 4 Argon when the speaker asks for the big brain.
- The face is now a red-and-white Australian Shepherd (client avatar and icons), and the voice is a cheerful Australian male clone (Pocket TTS, temperature 0.9), with a trailing-silence trimmer in the speech pipeline so replies don't trail dead air.
- The client is a single calm screen: big avatar, one plain status line, Talk / Mute / Volume, and everything technical tucked into a Details drawer. Playback runs through an in-page gain stage, so the phone's volume buttons govern what you hear.
- A phoneme/viseme cue track is generated per reply and emitted by the mobile server (viseme.py) for the animated-avatar work in progress; the live client still shows the still portrait.
