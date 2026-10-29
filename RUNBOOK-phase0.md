# Hermes Live Voice — 2-Minute Live Test

A Pipecat front-end that lets you TALK to Hermes and hear spoken replies:
your mic → local Whisper (what you said) → Hermes (the brain, trimmed
"voice" profile) → local Pocket TTS (eve's voice) → your speakers.

## Run it

Open Terminal on the Mac Studio and paste this one line:

```
cd ~/workspace/hermes-voice && API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/profiles/voice/.env | head -1 | cut -d= -f2- | tr -d '"') .venv/bin/python hermes_voice.py
```

Wait ~15 seconds after you run it — the speech models load first
(Whisper, Pocket TTS, VAD). Nothing bad happens if you talk early;
Hermes just won't hear the first seconds.

If macOS asks for microphone permission for Terminal, click **Allow**.
(That prompt is expected — it's the same permission Hermes needed.)

## What you will hear

- Nothing at startup. Hermes does not greet you — it listens.
- After you finish a sentence, there's a pause (expect **roughly
  2–4 seconds** at this stage — measuring and shrinking that pause is
  what this test is for), then Hermes answers out loud in eve's voice.
- Replies are spoken-style: short sentences, no lists read aloud.

## Say these lines (in order)

1. "What is the capital of France?" — sanity check, quick factual reply.
2. "Tell me a story about a dragon and a lighthouse keeper, at least
   ten sentences." — let it get going, then do the interruption test.
3. Anything you actually want to ask Hermes.

## Test 1 — barge-in (interruption)

While Hermes is mid-story (step 2 above), start talking over it:
say **"Stop — just give me the short version."**

What should happen: Hermes stops talking within a fraction of a
second, listens to you, and answers the new request.
What we're measuring: how long the voice keeps going after you start
speaking. Target: under ~150 ms of extra speech. Note what you perceive:
instant stop / short tail / kept talking.

## Test 2 — no-headphones echo

Do this run **without headphones**, speakers on, normal volume.
Ask line 1 again. Watch for: Hermes hearing its own voice and
interrupting itself, answering its own questions, or looping.
The bot has no echo cancellation in this spike — if it misbehaves,
that result is exactly what we need to know. Note what happened.

## Stop it

Click the Terminal window and press **Ctrl-C**. That's it — nothing
keeps running in the background.

## What this test settles

- Real end-to-end first-audio delay with a live mic (headless tests
  can't measure this — no speakers/mic in a script).
- Perceived barge-in stop time (machinery verified headlessly;
  your ears are the verdict).
- The no-headphones echo case: usable or unusable without a headset.

Everything else — speech recognition accuracy, the voice, the brain —
was already measured in the spike report.
