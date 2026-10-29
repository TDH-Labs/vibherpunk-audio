"""Vibherpunk Audio v2 lane (architect verdict v1 + v2 delta).

Components:
  events    — monotonic event spine (Step 1)
  registry  — JobRegistry: buffered write path + mirror (Step 1, v2 §2.2/I.2)
  arbiter   — Orchestrator Arbiter: commitment lane + speech lane (Step 2)
  context   — raw-text context builder, session log, structured memo (Steps 3-4)
  facts     — FactLedger + write-time extraction (Step 4)
  talker    — raw-text MLX talker client + boot golden check (Step 3)
  screen    — §2.7 output screen C0-C6 (Step 6)
  guard     — topic_guard, blocking (Step 6, Part D)
  speech    — SpeechPlan templates + sliced-speech player (Step 5)
  executor  — back-office worker executor (gateway + stub)
"""

__version__ = "2.0.0"
