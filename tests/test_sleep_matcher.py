"""Unit tests for is_sleep_command — especially the false positives."""
import os
import sys

os.environ.setdefault("HERMES_VOICE_LOGDIR", "/tmp/hv-sleep-matcher-logs")
sys.path.insert(0, "/path/to/hermes-voice")
from hermes_voice import is_sleep_command  # noqa: E402

SHOULD_SLEEP = [
    "Go to sleep",
    "go to sleep.",
    "Go to sleep now",
    "Go to sleep now please",
    "please go to sleep",
    "Hermes, go to sleep",
    "Hey Hermes, go to sleep.",
    "hey hermes go to sleep please",
    "Stop listening",
    "stop listening.",
    "Please stop listening now",
    "That's all, go to sleep",
    "That is all. Stop listening.",
    "Okay that's all, go to sleep now",
    "GO TO SLEEP",
]

SHOULD_NOT = [
    "Stop listening to the radio and check the weather",
    "stop listening to music",
    "Tell my kid to go to sleep",
    "Tell the kids a story about a bear who would not go to sleep",
    "Go to sleep and set an alarm for seven",
    "What time should I go to sleep tonight?",
    "I'm going to sleep early tonight",
    "I can't go to sleep without my podcast",
    "listening",
    "sleep",
    "What is the capital of France?",
    "Please have Hermes check its memory and tell me what city the user lives in",
    "",
    "go",
    "Stop",
    "Don't stop listening",
    "Never go to sleep before midnight",
]

fails = []
for t in SHOULD_SLEEP:
    if not is_sleep_command(t):
        fails.append(("should sleep", t))
for t in SHOULD_NOT:
    if is_sleep_command(t):
        fails.append(("should NOT sleep", t))
for kind, t in fails:
    print("FAIL", kind, repr(t))
print(f"MATCHER {len(SHOULD_SLEEP) + len(SHOULD_NOT) - len(fails)}/{len(SHOULD_SLEEP) + len(SHOULD_NOT)} passed")
sys.exit(1 if fails else 0)
