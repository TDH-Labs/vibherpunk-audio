# Vibherpunk Audio — install package for agents

**Reader: an AI agent installing this system for your human.** Every step
below is written to be executed by you, in order, with your human stopped
at exactly three points (§4). Nothing in this package is copied from the
reference machine's accounts: names, addresses, and values from the
reference build appear only as `{{PLACEHOLDERS}}` you substitute, or as
measured facts (model revisions, hashes, timings) that are the same for
every install.

Reference build harvested live on 2026-10-28 (Apple M4 Max Mac, macOS,
launchd). Where this package and the reference build differ, the package
says so — the differences are deliberate fixes, listed again in §8.

---

## 0. What this builds

A voice front end for a full agent (the "back office"), in two transports
on one pipeline:

- **Desk bot** (`hermes_voice.py`, launchd `ai.hermes.voice`): listens on
  the Mac's microphone, speaks through its speakers. Wake-word gated.
- **Mobile server** (`mobile_server.py`, launchd `ai.hermes.voice-mobile`):
  the same pipeline over WebRTC to a phone browser PWA. It binds
  `127.0.0.1:8917` only, is reached through Tailscale Serve, and never
  opens the Mac's mic or speakers.
- **TURN relay** (coturn, launchd `ai.hermes.turn`): relays phone audio
  when a direct WebRTC path fails. Tailnet-only.

The conversation pipeline:

```
audio in → wake-word gate → Whisper STT → Silero VAD + Smart Turn
→ safety screen (ModernBERT intent + GLiNER entity spans)
→ conversation lane (fast streaming model, remote)
→ Pocket TTS (leading-silence trimmed) → audio out
```

**The two tiers.** The *lane* is a small, fast, non-reasoning model with a
disciplined voice prompt. It answers conversation immediately and holds
exactly two tools. Everything substantive — memory, files, calendar,
messages, web, multi-step work — is delegated. The *back office* is a full
Hermes agent profile with its own tools and memory.

**Worker contract (the one seam).** The back office is any HTTP endpoint
with this shape:

- `POST {HERMES_WORKER_URL}/chat/completions` (OpenAI-shaped;
  default `http://127.0.0.1:8642/v1`, the Hermes gateway's API server),
  header `Authorization: Bearer <API_SERVER_KEY>`, body model
  `HERMES_WORKER_MODEL` (default `hermes-agent`).
- **In:** one self-contained task as a single user message. The worker
  does not hear the conversation; the lane's `delegate_to_hermes` tool
  packages everything the task needs into that message.
- **Out:** the worker's final message, whose content is (or ends with) a
  short spoken summary. Dispatch is asynchronous — the lane keeps talking
  — and when the job completes, the bot injects that summary as a spoken
  turn. The lane's second tool, `check_delegated_jobs`, reads job state
  the bot owns.
- **Failure:** if dispatch or the job fails, the lane speaks the failure
  line in §7. There is no dead air.

**Safety screen.** Before the lane acts on a transcript, two small local
encoder models examine it: a ModernBERT zero-shot classifier (intent:
answer / lookup / delegate) and GLiNER (entity spans the lane must not
garble — names, places). In this shipping configuration (`relay`) the
screen's verdict rides with the turn as an advisory note and the lane
decides; the components fail soft. A stricter fail-closed screened-speech
backend exists in the tree (`lane2/`, env `HERMES_LANE_BACKEND`) —
**it is not part of this build** (shelved by the reference owner on
2026-10-28) and this package does not install it.

**Speech rules the lane lives by** (from its system prompt, shipped in
`hermes_voice.py`): the worker is always "the back office" in speech,
never the framework's name; the lane never states a result it has not
been given; a dispatch is announced in one short process-only sentence;
results are announced in the worker's own summary when they land.

---

## 1. Prerequisites — the human provides these

You cannot obtain any of these for your human. If one is missing, stop
and say which (§4 lists the exact pauses).

1. **An Apple-silicon Mac** that stays on and awake, with Homebrew and
   `uv` installed. Reference: M4 Max, 36 GB unified memory. The local
   models total ~4.3 GB on disk (see `models-manifest.json`); budget at
   least 16 GB of unified memory for the two processes plus Hermes.
2. **A Tailscale account**, with Tailscale installed and signed in on the
   Mac, and the human's phone on the same tailnet.
3. **Hermes installed and running** (Nous Research's Hermes agent) with a
   profile the bot may use as its back office, and the gateway's
   API-server platform enabled (§3.3). Hermes installation and account
   setup are the human's, done beforehand.
4. **Model access for lane + worker**: an OpenCode Go subscription
   (the reference route for both: lane `glm-5.3-flash`, worker profile
   on `deepseek-v4.1-flash`). Any OpenAI-shaped endpoint can substitute
   via the env knobs in §3.5, but the reference timings and behavior
   were measured on this route.
5. **Disk**: ~10 GB free for the venv, model cache, and logs.

---

## 2. OBTAIN — downloads, pinned

Work through this list exactly. `models-manifest.json` in this repo is
the machine-readable twin of §2.2.

### 2.1 The app code

Clone this repository to the install directory (call it `{{APP_DIR}}`):

```
git clone <this repo> {{APP_DIR}}
```

Layout you should now have: `hermes_voice.py`, `mobile_server.py`,
`client/` (the PWA), `models/` (wake-word assets), `install.sh`,
`run.sh`, `run_mobile.sh`, `tests/`, the RUNBOOKs.

### 2.2 Models — exact repos, exact revisions

Download each with the revision pinned (`snapshot_download(repo,
revision=<sha>)` or the HF CLI equivalent). Do not track a moving HEAD —
the reference cache was swept once and restored unpinned; it survived
only because every repo's head happened to be old. Pin.

| Role | Repo | Revision | Principal file SHA-256 | Size |
|---|---|---|---|---|
| Ears (STT) | `mlx-community/whisper-small-mlx` | `45f3915923c7a79a5a5b5a7d909d39aeb0e5630e` | `weights.npz` `55b6674c…a86b09f5` | ~459 MB |
| Voice (TTS) | `kyutai/pocket-tts-without-voice-cloning` | library-pinned (see note) | `model.safetensors` `be9c6b48…bbb4abb0a` | ~217 MB |
| Screen (spans) | `urchade/gliner_medium-v2.1` | `40ec419335d09393f298636f471328b722c6da9e` | `model.safetensors` `922214c0…a3c023` | ~1.5 GB |
| Screen (intent) | `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` | `d421c4545a438fd006fb43f8b981c5d908faa1e1` | `model.safetensors` `5af4dac8…b44b46` | ~2.1 GB |

(Full hashes in `models-manifest.json`. Whisper, GLiNER, and ModernBERT
were content-hash-verified file-by-file against the hub on 2026-10-28.)

- **Pocket TTS note:** you do not download this one by hand. The
  `pocket-tts` pip package (2.1.0 in the reference) fetches the model
  and the voice embeddings itself, at revisions hardcoded in the
  package (model `d29db797…`, embeddings `e041936c…`). The voice is a
  catalog name inside that package — there is no voice file to obtain.
- **Wake-word spotter model** (required, separate source): the
  sherpa-onnx KWS zipformer, `sherpa-onnx-kws-zipformer-gigaspeech-
  3.3M-2024-01-01` (~19 MB: encoder/decoder/joiner ONNX, `bpe.model`,
  `tokens.txt`). Reference source: the ModelScope repo
  `pkufool/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`
  (also distributed with sherpa-onnx by k2-fsa). Place the files in
  `{{APP_DIR}}/models/sherpa-kws/`. This repo's `models/sherpa-kws/`
  already carries them. Ignore the `keywords.txt` inside — it is the
  stock demo list; §3.6 explains why it is never used.
- **Not part of this build** (in the manifest for completeness only):
  the trained `hey_viber.tflite` (only for the alternate `oww` wake
  engine), `mlx-community/Qwen3-4B-Instruct-2507-4bit` (bench),
  `LiquidAI/LFM3-Audio-2B` (shelved experiment). Do not download them.

### 2.3 Runtime dependencies

```
brew install portaudio coturn
```

Then the Python environment (Python 3.12, venv at `{{APP_DIR}}/.venv`).
The repo's `install.sh` creates the venv and installs the Pipecat line:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "pipecat-ai[pocket-tts,whisper,silero,local-smart-turn-v3,openai]" pyaudio
```

**That line alone does not reproduce the reference environment** — the
screen and wake components arrive outside it. Install the measured set
(the reference venv's actual versions, 2026-10-28):

```
uv pip install --python .venv/bin/python \
  pipecat-ai==1.14.0 pocket-tts==2.1.0 sherpa-onnx==2.0.1 \
  gliner==0.2.24 transformers==5.21.0 torch==2.14.1 \
  mlx-whisper==0.4.3 pyopen-wakeword==1.1.0 \
  fastapi==0.162.0 uvicorn==0.54.0 huggingface-hub==2.3.0
```

(Several of these also arrive as transitive dependencies of the Pipecat
extras; the pinned line is the ground truth — if a version conflicts,
the reference venv wins and you note the divergence in your install log.)

First run of each model downloads/compiles; warm the cache in §5 step 7
so the first real conversation is not the first load.


---

## 3. CREATE — everything the install generates

Nothing in this section is downloaded. Each artifact is generated on the
install machine, by you, with the exact method given. **No generated
secret is ever printed** — not to the terminal log, not to chat. Secrets
go into their files; the human receives exactly one secret-bearing
artifact, the access link in §3.7, handed over at the very end.

### 3.1 Substitution table — decide these first

| Placeholder | What it is | Reference build |
|---|---|---|
| `{{APP_DIR}}` | Absolute path of the cloned repo | the owner's workspace path |
| `{{HOME}}` | The installing user's home | — |
| `{{OWNER_NAME}}` | The human's first name — optional, for display strings only; the shipped lane prompt is owner-neutral and never uses it | the owner's first name |
| `{{ASSISTANT_NAME}}` | Spoken + displayed name of the assistant | `Vibherpunk` |
| `{{VOICE}}` | Pocket TTS catalog voice name | `azelma` (code default `eve`) |
| `{{WAKE_PHRASES}}` | Comma-separated wake phrases | `hey clanker,hey viber` |
| `{{WORKER_MODEL}}` | Hermes profile model for the back office | `deepseek-v4.1-flash` |
| `{{WORKER_PROVIDER}}` | Hermes provider name for that model | `opencode-go` |
| `{{WORKER_BASE_URL}}` | That provider's endpoint | `https://opencode.ai/zen/go/v1` |
| `{{LANE_MODEL}}` | Lane model id (env `LANE_MODEL` if changed) | `glm-5.3-flash` |
| `{{TAILNET_HOST}}` | This Mac's Tailscale DNS name | `<machine>.<tailnet>.ts.net` |
| `{{TURN_HOST_IP}}` | This Mac's Tailscale IPv4 (`tailscale ip -4`) | a `100.x.x.x` address |
| `{{TURN_REALM}}` | Any stable realm string you choose | the machine's name |

### 3.2 `~/.hermes/.env` — two keys

From `templates/hermes.env.example`, permissions `0600`:

- `API_SERVER_KEY` — **generate**: `openssl rand -hex 32`. This is the
  bearer key for the Hermes gateway's API server; the voice bot
  presents it on every worker call.
- `OPENCODE_GO_API_KEY` — **the human's**, from their own subscription
  account. Obtaining it is human step §4.3; you receive the value only
  as "it is in the file now" — if the human pastes it to you instead,
  write it to the file and do not repeat it anywhere.

### 3.3 Hermes profile configuration

Merge `templates/hermes-config.fragment.yaml` into the back-office
profile's `~/.hermes/config.yaml`. It sets: the worker model/provider,
`platforms.api_server.enabled: true`, the Pocket TTS provider block
(**`voice_compatible: true` is load-bearing** — without it Pocket is
silently unavailable to the voice app), `wake_word.enabled: false`,
and ISO-code STT language. Start (or restart) Hermes **only if the
human's setup expects you to** — the gateway is the human's agent, not
part of this install; if it is already running, a config change needs
their word before any restart.

### 3.4 Mobile access token

The mobile server's shared secret. Method (exactly the reference's —
`templates/run_mobile.sh` also self-creates it on first run):

```
umask 077
head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > ~/.hermes/mobile-voice-token
chmod 600 ~/.hermes/mobile-voice-token
```

24 random bytes, hex-encoded: a 48-character file, mode `0600`. The
server reads it via env `MOBILE_ACCESS_TOKEN`; it gates `POST/PATCH
/api/offer` and `/api/session`. **Never print the value.** The human
meets it only inside the access link (§3.7). Rotation = regenerate the
file and restart `ai.hermes.voice-mobile`; every phone then needs the
new link once.

### 3.5 TURN shared secret + coturn config

- Generate: `openssl rand -base64 32` → write to
  `~/.hermes/turn-auth-secret`, `chmod 600`. The mobile server reads
  this file and mints time-limited TURN credentials (HMAC over an
  expiry timestamp) for the phone client.
- Install `templates/turnserver.conf` at `~/.hermes/turnserver.conf`
  (`chmod 600`), with the **same value** as `static-auth-secret`, plus
  `{{TAILNET_IPV4}}`/`{{TAILNET_IPV6}}`, `{{TURN_REALM}}`,
  `{{MACHINE_NAME}}`. Listening port 3478. Peer access stays restricted
  to the tailnet range (`allowed-peer-ip`) — this relay is never
  exposed to the public internet.

### 3.6 Run scripts, service files, personalization

- Install `templates/run.sh` and `templates/run_mobile.sh` as
  `{{APP_DIR}}/run.sh` / `run_mobile.sh` (executable), substituted.
  They export the two keys from `~/.hermes/.env` (one source — the
  reference build inlines its OpenCode key in `run.sh`; that is a
  defect, do not reproduce it), set `HERMES_VOICE={{VOICE}}` and
  `HERMES_WAKE_ENGINE=sherpa`, and exec the bots from the venv.
- Install the three plists from `templates/` into
  `~/Library/LaunchAgents/`, substituted, and create the log dirs
  `{{APP_DIR}}/logs/` and `{{APP_DIR}}/logs-mobile/`.
- **Leave the lane prompt exactly as shipped.** In `hermes_voice.py`,
  `LANE_SYSTEM_PROMPT` (and its `LANE_SYSTEM_PROMPT_ROUTED` sibling)
  address "the user" and never name the owner. Do not edit a name
  into them, and do not alter any behavioral rule in them — those
  rules are the product. (The author's own deployment carries a
  locally personalized copy of these prompts; that personalization
  is a private local variant, not part of this distribution, and no
  install step reproduces it.)
- **Wake phrases** ride in env `HERMES_WAKE_PHRASES` (default
  `hey clanker,hey viber`). The sherpa gate tokenizes the phrases
  itself at startup and writes its own keywords file — that is why
  the distribution `keywords.txt` is never used. Detection threshold
  is derived: `0.05 + 0.4 × HERMES_WAKE_SENSITIVITY` (default
  sensitivity 0.6 → threshold 0.29). If you coin a phrase whose
  spelling misleads the tokenizer, the code shows the pattern for
  registering a phonetic variant (`SHERPA_PHRASE_VARIANTS`).
- **Tailscale Serve** (the phone's path in):
  `tailscale serve --bg --https=8443 http://127.0.0.1:8917`
  Verify with `tailscale serve status`: an entry
  `https://{{TAILNET_HOST}}:8443 → 127.0.0.1:8917`, tailnet only.

### 3.7 The access link — the deliverable

When §6 verification passes, hand your human exactly this, once:

```
https://{{TAILNET_HOST}}:8443/?token=<contents of ~/.hermes/mobile-voice-token>
```

The page stores the token in the phone's local storage and strips it
from the URL; the human taps the link, adds the PWA to the home
screen, and never needs it again unless the token rotates. Do not put
this link in any log, file, or message other than the handoff itself.


---

## 4. Human-only steps — three hard stops

Pause the build at each of these. Do not attempt to route around them;
each is impossible for an agent by design of the platform or the
account, not by lack of trying.

1. **macOS microphone permission.** The desk bot's host process needs
   the microphone. TCC permission cannot be granted by any program —
   the human grants it in the macOS prompt (or System Settings →
   Privacy & Security → Microphone) the first time the bot opens the
   mic. Until they do, the desk bot hears nothing and no error will
   tell you so plainly.
2. **Tailscale sign-in.** On the Mac, and the phone joining the same
   tailnet. Account sign-in is the human's. (Serve configuration in
   §3.6 is yours; the account is theirs.)
3. **Provider account logins.** The OpenCode Go subscription sign-in
   that produces `OPENCODE_GO_API_KEY` — and, if the back-office
   Hermes profile needs its own provider auth, that login too. You
   handle keys as file contents after the fact (§3.2), never the
   login itself.

---

## 5. Build order

1. Confirm §1 prerequisites with the human; collect every `{{…}}`
   value from §3.1. Hit human stops §4.2 and §4.3 here — get them
   done first so nothing waits on a login later.
2. Clone the repo to `{{APP_DIR}}` (§2.1).
3. `brew install portaudio coturn`; run the dependency installs (§2.3).
4. Download the models at the pinned revisions (§2.2) into the
   HuggingFace cache; place the sherpa KWS files in `models/sherpa-kws/`.
5. Write `~/.hermes/.env` (§3.2, `0600`); merge the config fragment
   (§3.3). Confirm the Hermes gateway answers (§6, check 1).
6. Generate the mobile token (§3.4) and the TURN secret (§3.5);
   install `turnserver.conf`.
7. **Warm every model once, in a throwaway process** — load Whisper,
   Pocket TTS, GLiNER, ModernBERT, and the sherpa spotter from the
   local cache with the network off, and run one trivial input through
   each. This proves the cache bytes load before any service depends
   on them, and it is the only model test you run: never bounce a live
   service to "check" a model.
8. Install run scripts + plists (§3.6); create the log dirs.
9. Load the services: `ai.hermes.turn`, then `ai.hermes.voice`, then
   `ai.hermes.voice-mobile` (launchctl bootstrap/load per your macOS).
   Expect human stop §4.1 at the desk bot's first mic open.
10. Configure Tailscale Serve (§3.6) and verify the mapping.
11. Run the full verification sequence (§6).
12. Hand the human the access link (§3.7) and the wake phrases. Done.

---

## 6. Verification — the true checks

Run these in order. Expected results are stated exactly; several
"wrong-looking" results are correct, and chasing them wastes the build.

1. **Gateway alive + auth on:**
   `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8642/v1/models`
   → **401** without a key (correct: alive, auth enforced). With
   `Authorization: Bearer <API_SERVER_KEY>` → 200.
2. **Services loaded:** `launchctl list | grep ai.hermes` shows
   `ai.hermes.voice`, `ai.hermes.voice-mobile`, `ai.hermes.turn`, each
   with a pid. Desk log `{{APP_DIR}}/logs/launchd.out.log` shows the
   pipeline starting; wake gate reports its engine and threshold.
3. **Mobile root:** `curl -s -o /dev/null -w '%{http_code}'
   http://127.0.0.1:8917/` → **200**.
   `/health` → **404, expected** — that route does not exist; do not
   "fix" it. (`/healthz` does exist. `/api/auth-check` → 404 too.)
4. **Token gate:** `curl -X POST http://127.0.0.1:8917/api/offer`
   with no token → **403**. A 200 here means the gate is not wired —
   stop and fix before proceeding.
5. **ICE/TURN:** `GET /api/ice-config` (with the token) returns
   `iceServers` including `turn:` entries for `{{TURN_HOST_IP}}:3478`
   (tcp + udp). If only `stun:` entries appear, the TURN secret file
   is missing or unreadable.
6. **Offer flow (the real transport test):** from the phone, open the
   access link (§3.7) and start a session: `POST /api/offer` returns
   **200** — or **409 Busy** if a session is already live, which is
   also a pass. The PWA connects, and the client shell version in
   `client/sw.js` (`vibherpunk-audio-v5` in the reference) is what the
   phone runs.
7. **Spoken round trip (desk or phone):**
   - Say a wake phrase → the lane answers out loud.
   - Ask a plain conversation question → the lane answers with no
     dispatch.
   - Ask for something needing real data ("check my calendar…") →
     the lane dispatches in one short sentence naming **the back
     office**, and the worker's summary is announced when it lands.
   - Say **"go to sleep"** → the listening window closes at once. A
     result landing while asleep is held and announced first on the
     next wake.
8. **Latency sanity (informational):** on the reference route the
   lane's first token measured a 434 ms median; end-to-end time to
   first sound on the live build is ~1.2 s, of which ~0.6 s is the
   end-of-turn wait. Gross deviations point at route or load, not at
   the install.

---

## 7. Failure lines and branding — preserve verbatim

These strings are behavioral, not decorative. Keep them unless the
human is deliberately rebranding, in which case substitute
`{{ASSISTANT_NAME}}` everywhere consistently (prompt, client title,
service-worker cache name, these lines).

- Worker failure: **"Sorry — {{ASSISTANT_NAME}} couldn't complete that
  task."** (Em dash, exactly.)
- Lane unreachable: **"Sorry, I'm having trouble reaching my
  conversation model right now."**
- The worker is **"the back office"** in all speech. Never its
  framework name, never "Hermes", in anything spoken.
- Sleep commands: **"go to sleep"** / **"stop listening"**.
- Wake phrases (reference): **"hey clanker"**, **"hey viber"**.
- App title: **"Vibherpunk Audio"** (`client/index.html` title +
  `apple-mobile-web-app-title`).
- Service-worker rule: **any change under `client/` must bump the
  `CACHE` name in `client/sw.js`** (`"vibherpunk-audio-v5"` in the
  reference), or phones keep running the old shell forever.

---

## 8. What NOT to do

- **Never bind the install to the reference build's machine, accounts,
  people, phone numbers, tailnet name, or tokens.** Every such value
  is a `{{PLACEHOLDER}}`. If you find a real value from the reference
  machine in your output, stop and redact it.
- **Never print a generated secret** — the mobile token, the TURN
  secret, the gateway key, the provider key — into a terminal log,
  chat, commit, or file other than its designated `0600` home. The
  token reaches the human only inside the §3.7 link.
- **Never inline an API key in a run script.** The reference build
  does this with its OpenCode key; it is a known defect there (its
  test scripts carry plaintext gateway keys too — do not replicate
  the pattern, and do not "clean it up" on the reference machine as
  part of an install; that is the owner's separate call).
- **Never restart the human's Hermes gateway or a live voice service
  to test something.** Warm models in a side process (§5 step 7).
- **Never chase the `/health` 404** — see §6 check 3.
- **Never expose the mobile server or the TURN relay publicly.** The
  server binds loopback; Serve is tailnet-only; coturn peers are
  tailnet-restricted. No port forwarding, no funnel.
- **Never enable the `lane2/` screened backend or the LFM3
  anticipator from the reference tree.** They are shelved experiments
  with a recorded fail, not options in this build.
- **Never edit `client/` without bumping the service-worker cache
  name** (§7).
- **Never let the lane's prompt lose its rules while personalizing
  it** — names change, behavior doesn't: no invented results, dispatch
  announced in one sentence, the worker is the back office.

---

*Package assembled from the live reference build, 2026-10-28.
Companions: `models-manifest.json` (pinned models), `templates/`
(every generated file), `RUNBOOK.md` / `RUNBOOK-mobile.md` (operating
the built system).*
