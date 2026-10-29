# RUNBOOK — Hermes Voice, mobile (WebRTC) transport

> **Branding (2026-10-27, the user):** the app's display name is **Vibherpunk Audio** (page title, heading, PWA manifest name; short_name "Vibherpunk"). All user-visible "Hermes" strings were removed from client/ and the shell cache was bumped to vibherpunk-audio-v4. Internal names (project dir, launchd labels, the window.hermesVoice JS handle) are unchanged working names. Spoken announcements were renamed to Vibherpunk the same evening (the user's call): held results announce "Vibherpunk finished", and the desk lane now speaks as Vibherpunk with the worker referred to as "the back office" (hermes_voice.py LANE_SYSTEM_PROMPT; internal tool name delegate_to_hermes unchanged).


The phone/tablet front end for the Hermes live-voice system. Same brain as
the desk bot (fast lane + Hermes worker, MLX Whisper, Pocket "eve") — this
is a second *transport* on the same pipeline, not a second voice stack.

## The address (how the user connects)

```
https://your-mac.your-tailnet.ts.net:8443/?token=<TOKEN>
```

- `<TOKEN>` lives in `~/.hermes/mobile-voice-token` on the Mac (file mode
  0600). It is created automatically on first server start if missing.
  **Never paste the token into chat, docs, or git.**
- Open the link once per phone: the page stores the token on the device and
  strips it from the address bar. Later visits to the bare address
  (`https://your-mac.your-tailnet.ts.net:8443/`) just work on that
  device.
- Tailnet-only: the address resolves and answers only on the user's tailnet
  (Tailscale Serve). Nothing is exposed to the public internet.
- To rotate the token: delete `~/.hermes/mobile-voice-token`, restart the
  service (below), then open the new link on each phone once.

## Putting it to sleep (spoken — works on the phone too)

While connected, say either phrase as your whole utterance (a polite
lead-in or "that's all, ..." in front is fine):

- **"go to sleep"**
- **"stop listening"**

Hermes answers "Going to sleep — say hey Clanker when you need me." and
the session ends. Matching is deliberately conservative: a request that
merely contains the words ("stop listening to the radio and check the
weather", "tell my kid to go to sleep") does NOT put it to sleep, and
the phrase is never treated as conversation or handed to Hermes as a
task.

**Held results:** if a Hermes job finishes while you are asleep or
disconnected, its result is NOT lost and NOT spoken into the void. It
is held, and the next time you connect it is announced first — "While
you were asleep, Hermes finished: ..." — before anything else, in
completion order if there are several. (The same behavior on the desk
bot, where the next "hey hermes" triggers the announcement.) Holds and
announcements are logged (`held_result_held` /
`held_result_announced` in `logs-mobile/metrics.jsonl`).

Sleep is the soft off. The hard off is stopping the service:
`launchctl bootout gui/501/ai.hermes.voice-mobile` (desk bot:
`launchctl bootout gui/501/ai.hermes.voice`).

## the user's 3-minute check — Android (Chrome)

**PENDING-OWNER-TEST** (needs the physical phone)

1. Phone on Tailscale (app connected), open the link in Chrome.
2. Tap **Add to Home screen** if you want it as an app icon (optional).
3. Tap **Connect**. Allow the microphone prompt.
   While you're connected, the page holds a **screen wake lock** — the
   screen should stay awake for the whole conversation (the page log
   shows "Wake lock: acquired", and "Wake lock: released" when you
   disconnect or say "go to sleep"). If the screen still auto-locks
   mid-conversation, this browser doesn't support the wake-lock API —
   the fallback is the phone's own auto-lock setting (set it longer,
   or Never, while using voice).
4. Status should read **Listening**. Say: "What is the capital of France?"
   Expect "Paris." within ~2–3 seconds, and a "You: / Hermes:" caption.
5. Say: "Hermes, check what version of macOS this machine is running and
   report back." Expect an immediate acknowledgment, the status to show
   Hermes working, then the spoken answer when the job finishes.
6. Say **"go to sleep."** Expect the goodbye line ("Going to sleep —
   say hey Clanker when you need me.") and the session to end.
   Reconnect and talk again — should just work.

Pass: all of the above, no errors. Anything else: note what the status line
said and tell Muse.

## the user's 3-minute check — iPhone (Safari)

**PENDING-OWNER-TEST** (needs the physical phone)

1. Phone on Tailscale (app connected), open the link in Safari.
2. Optional: Share → **Add to Home Screen** for the app icon.
3. Tap **Connect** — on iOS this tap is also what unlocks audio playback;
   do not skip straight to talking. Allow the microphone prompt.
   While you're connected, the page holds a **screen wake lock** — the
   screen should stay awake for the whole conversation (the page log
   shows "Wake lock: acquired"). The wake lock prevents *auto*-lock
   only; it cannot stop a manual lock with the side button — that's
   what step 5 tests. If the screen still auto-locks while connected,
   this browser lacks the wake-lock API and the fallback is the
   phone's auto-lock setting (Settings → Display & Brightness →
   Auto-Lock → Never, while using voice).
4. Status should read **Listening**. Ask the same two questions as the
   Android check; expect the same behavior.
5. While connected, lock the phone for 30 seconds, unlock, and talk again.
   If it dropped, tap Connect — any background job Hermes was running
   still finishes; its result is written to the Mac log, not spoken into
   the new session.
6. Say **"go to sleep."** Expect the goodbye line and the session to
   end (or tap **Disconnect**).

Pass: same as Android. iOS background/lock behavior is the main unknown —
that is what step 5 is for.

## How it runs (ops)

| Piece | What |
|---|---|
| Mobile server | launchd `ai.hermes.voice-mobile` → `run_mobile.sh` → `mobile_server.py`, binds **127.0.0.1:8917** only |
| TURN relay | launchd `ai.hermes.turn` → coturn, config `~/.hermes/turnserver.conf` (0600), listens **127.0.0.1 + the host's tailnet IP :3478** (UDP+TCP) only — never public. The phone gets short-lived credentials from the token-gated `GET /api/ice-config`; the shared secret is `~/.hermes/turn-auth-secret` (0600, never printed). Restart: `launchctl kickstart -k gui/<uid>/ai.hermes.turn` |
| Serving | `tailscale serve --https=8443 http://127.0.0.1:8917` (background serve entry; the :443, :7443 and :10000 entries belong to other services — leave them) |
| Logs | `logs-mobile/` in the project (`metrics.jsonl`, `launchd.*.log`, `server.out.log`) |
| Session state | `GET /api/session` with header `x-voice-token: <TOKEN>` (events, spoken lines, turn timings) |
| Health | `curl http://127.0.0.1:8917/healthz` → `{"status":"ok"}` |

Restart the mobile service:

```
launchctl kickstart -k gui/501/ai.hermes.voice-mobile
```

Check it:

```
launchctl list | grep ai.hermes.voice-mobile
curl -s http://127.0.0.1:8917/healthz
tailscale serve status
```

Ground rules:

- The mobile server **never opens the Mac's microphone or speakers** and
  runs as its own process. It cannot disturb the desk bot
  (`ai.hermes.voice`, local transport) or the gateway.
- One phone session at a time: a new connection replaces the old one.
- Connecting **is** the session start — there is no wake word on mobile.
  The desk bot's wake gate is desk-only (phrases are now "hey clanker" / "hey viber" via the sherpa engine — see RUNBOOK.md, "Wake word").
- If the phone drops mid-job, the job still finishes; the result is logged
  (`result_logged_disconnected` in `logs-mobile/metrics.jsonl`), never
  replayed into a later session.
- Known trait: the **desk bot** exits after ~5 minutes with no interaction
  (Pipecat idle timeout; its launchd job has KeepAlive off). It is not a
  crash — `launchctl kickstart gui/501/ai.hermes.voice` brings it back.
  The mobile service does not have this trait (KeepAlive is on).

## Standing rule — shipping client changes

**Any change under `client/` MUST bump the service-worker cache version** (`const CACHE` in `client/sw.js`). Installed phones serve the shell from the service-worker cache; an unbumped version means devices keep running the old page indefinitely and server-side fixes appear to "not work." The server also sends `Cache-Control: no-cache` for `/` and `/sw.js`, documents are network-first in the worker, and the page reloads itself once when a new worker takes over — but the cache-name bump is still required: it is what forces the old cache to be discarded on activation. (Added 2026-10-27, the fix2 incident in Troubleshooting below.)

## Troubleshooting

| Symptom | Check |
|---|---|
| Page won't load on the phone | Tailscale connected on the phone? `tailscale serve status` on the Mac shows the :8443 entry? |
| "Add ?token=… to the address" | Open the full link (with token) once on that device; the page remembers it after that. |
| Connects, but Hermes never answers | `curl 127.0.0.1:8917/healthz`; check `logs-mobile/metrics.jsonl` for `lane_error` (conversation model unreachable) — Hermes will have *said* that out loud too. |
| Status stuck on "Connecting…" | **Diagnosed 2026-10-27 (mobile fix1):** phone browsers send ICE checks from per-interface sockets that cannot use the tailnet route, so host-candidate media never reaches the server (server logs show the session start, then a timeout at exactly 60 s, no `mobile_client_connected`). Fixed by the TURN relay above: the client now fetches `/api/ice-config` and connects through a TURN-over-TCP allocation across the tunnel. If it recurs: (1) confirm `ai.hermes.turn` is running and the page log shows "ICE: relay configured" plus a `relay` candidate line; (2) restart the TURN service, then the mobile service; (3) read the page log tail to Muse — it now records candidate types, ICE states, and candidate-pair states after 9 s. |
| Status stuck on "Connecting…" right after a client fix shipped | **Diagnosed 2026-10-27 (mobile fix2):** the phone is running the PRE-fix page from its service-worker cache — e.g. the fix1 TURN client never executed (the page log shows no "ICE: relay configured" line). Proof: check the access log around the attempt — `grep "ice-config" logs-mobile/launchd.out.log` (access lines live in **launchd.out.log, NOT server.out.log**) shows no `/api/ice-config` request near the device's session, while the session still starts and dies at exactly +60 s. Cure: ship the client with a bumped SW cache version (standing rule above); devices pick it up on the next open with no manual cache clearing — the first open can still show the old page while the new worker activates, so reopen once if it sticks. |
| Screen auto-locks mid-conversation and the session drops | **Wake lock added 2026-10-27 (mobile fix3, SW `hermes-voice-v3`):** while connected, the client holds a screen wake lock (page log: "Wake lock: acquired" on connect, "Wake lock: released" on disconnect/sleep end; locks auto-release when the page is hidden and are re-acquired when it becomes visible again mid-session). If the page log instead says "Wake lock: not supported" or "Wake lock: request failed", that browser lacks the Screen Wake Lock API — the device's auto-lock setting governs; lengthen it (or set Never) while using voice. A manual lock (side button) always overrides the wake lock and still drops the session — reconnect after unlocking; held results are announced on the next connect. |
| Hermes desktop app also wakes at the desk | The desktop app runs its own "hey hermes" listener. Disable `wake_word.enabled` in Hermes settings if the double-wake bothers you (desk bot unaffected either way). |


## PWA first-run access (2026-10-27 fix)

An installed PWA (iPhone/iPad home-screen app) has its OWN storage, separate from the
browser. If the app was installed without the token link ever being opened inside the
app itself, its storage is empty and Connect used to dead-end at "Missing access token -
open the full URL from the runbook" - an instruction that cannot be followed from inside
a PWA. Fixed 2026-10-27 (client cache vibherpunk-audio-v5): when no token is stored, the
app shows an Access panel - paste the full access link (or just the token) once and tap
"Save access". The token is extracted from the link and stored in the app's own storage;
it persists across launches on that device. "Access settings" (bottom of the page)
reopens the panel; "Remove saved access" clears it. First-run steps per device:
open the installed app, copy the full link from chat, paste it into the panel, Save,
then Tap to talk. The ?token= browser bootstrap still works for plain browser use.
