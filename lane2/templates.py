"""Arbiter-owned speech templates (v1 §2.5/§2.8, v2 Parts C/D/E).

Templates are the only voice of job state. They are rendered from
registry slots, never model-generated, and are linted at build time
against the §2.7 screen lexicons (expected matches whitelisted per
template ID, §2.7.1 item 2). Variants rotate per session so repeats
do not stack on one phrasing; selection is deterministic per
(template, job, turn) with last-used avoidance in the speech lane.
"""
import hashlib

# slot defaults: a missing slot must never crash speech; defaults are
# the neutral forms the specs themselves use.
DEFAULTS = {
    "label": "that", "verb": "do", "obj": "that", "question": "",
    "options": "", "recipient": "", "draft_head": "", "alt_verb": "do",
    "alt_obj": "that", "summary": "", "capability": "that capability",
    "subject": "", "value": "", "as_of": "", "n": "a few",
    "label_1": "the first", "label_2": "the second",
    "state_phrase_1": "in progress", "state_phrase_2": "in progress",
    "queue_pos_phrase": "in line", "progress_phrase": "no estimate yet",
    "age_phrase": "just now", "result_headline": "",
    "failure_reason_short": "", "by_whom_phrase": "",
    "failure_sentence": "", "eta": "",
}

TEMPLATES = {
    # ---- acknowledgements (Step 5: 6-8 variants for the hot ones)
    "T-ACK": [
        "On it — {label}.",
        "Got it. Passing {label} to the back office.",
        "Okay — {label}, handing that over now.",
        "Sure. {label} is with the back office.",
        "Right — sending {label} over.",
        "Yep, {label} is going to the back office.",
        "Passing that on — {label}.",
    ],
    "T-ACK-HOLD": [
        "Sure — I've got the {label} request. Go ahead.",
        "Okay, holding on {label}. Go ahead.",
    ],
    "HOLD_OK": [
        "Sure.", "Okay.", "Go ahead.", "Mm-hm, go ahead.",
        "Right.", "Sure thing.",
    ],
    "T-PROMPT-GO-ON": ["Go ahead.", "I'm listening."],
    # ---- clarifies
    "T-CLARIFY-HOLD": ["Which one did you mean?"],
    "T-CLARIFY-CANCEL": ["Which one should I cancel?"],
    "T-CLARIFY-SLOT": ["Quick question on {label}: {question}"],
    # ---- cancel (§2.8 table)
    "T-CANCEL-QUEUED": ["Cancelling {label} — it hasn't gone out yet."],
    "T-CANCEL-NEVER": ["Cancelled — I hadn't started {label} yet."],
    "T-CANCEL-ASKING": [
        "I've asked to stop {label}. I'll confirm once it's stopped."],
    "T-CANCEL-CONFIRMED": ["{label} is stopped. Nothing went through."],
    "T-CANCEL-PARTIAL": [
        "I stopped {label}, but part of it had already gone through."],
    "T-CANCEL-DONE": ["{label} already finished."],
    "T-CANCEL-ALREADY": ["{label} isn't running anymore."],
    # ---- confirms (Part E §2.5.1/§2.5.3/§2.5.4)
    "T-CONFIRM": ["{verb} {obj} — confirm?"],
    "T-CONFIRM-ALT": ["{verb} {obj}, or {alt_verb} {alt_obj}?"],
    "T-CONFIRM-DRAFT": [
        "Send to {recipient}, starting '{draft_head}' — confirm?"],
    "T-CONFIRM-EXPIRED": ["Didn't hear a confirm — I didn't {verb} {obj}."],
    "T-CONFIRM-REJECTED": ["Cancelled."],
    "T-OA-QUESTION": ["{question} Options: {options}."],
    "T-REPROMPT": [
        "Didn't catch that — did you want {verb} {obj}?",
        "Didn't catch that.",
    ],
    "T-DUP-CONFIRM": ["You already asked for {label} — send it again?"],
    "T-DUP-STILL": ["Still on the earlier one — {label}."],
    "T-DUP-DONE": ["Already done."],
    # ---- outcomes / results
    "T-RESULT": ["Here's what came back on {label}: {summary}"],
    "T-RESULT-FACT": [
        "Here's what came back on {label}: {subject} is {value}."],
    "T-RESULT-MULTI": [
        "The back office came back on {label} with {n} items. "
        "The first: {subject} is {value}. Want the rest?"],
    "T-RESULT-NULL": ["The back office came back on {label}: {summary}"],
    # Lane3 H1 fallback: spoken when a result summary carries typed
    # values not backed by registry facts — announces completion
    # without repeating the unbacked values.
    "T-RESULT-BRIEF": ["{label} is done.", "{label} finished."],
    "T-SCOPE-KILLED": [
        "That needed {capability}, which it didn't have — stopped."],
    "T-DISPATCH-FAILED": [
        "I couldn't get {label} started — the back office didn't take it."],
    "T-PREFLIGHT-REFUSAL": [
        "I can't do that right now — the back office connection is down."],
    # ---- status (screen replacements, §2.7.3)
    "T-STATUS-PENDING": [
        "The back office is still working on {label}. "
        "I don't have a result yet.",
        "Still working on {label} — nothing back yet.",
        "{label} is still with the back office. I'll tell you when it's back.",
    ],
    "T-PENDING": ["Still working on that — I'll tell you when it's back."],
    "T-STATUS-NOETA": [
        "The back office is still on {label}, and I don't have a time "
        "estimate yet."],
    "T-STATUS-ETA": [
        "The back office is still on {label}. Their estimate is {eta}."],
    "T-STATUS-FAILED": [
        "The back office couldn't complete {label}.{failure_sentence}"],
    "T-STATUS-CANCELLED": ["{label} was cancelled."],
    "T-STATUS-SUBMITTING": [
        "I'm still getting {label} over to the back office."],
    # ---- screen fallbacks (§2.7.4)
    "T-UNVERIFIED": ["I can't confirm that from what I have."],
    "T-QUOTE": ["{label} says, quote: {summary}. End quote."],
    "T-DISAMBIG": ["Which one do you mean — {label_1} or {label_2}?"],
    "T-FB-LOOKUP-NONE": ["I don't have anything from the back office on that."],
    "T-FB-ANSWER": ["I don't have a reliable answer for that one."],
    "T-FB-FAULT": ["Sorry — give me a second. Could you say that again?"],
    "T-FB-FAULT-MID": ["Sorry, I lost my train of thought there."],
    "T-LANE-DOWN": [
        "Sorry — Vibherpunk couldn't complete that task."],
    # ---- guard STATUS.v2 (Part D.1.6), keyed by job state at render
    "T-GUARD-QUEUED": [
        "{label} hasn't started yet — it's {queue_pos_phrase}. "
        "I'll tell you when it's done."],
    "T-GUARD-RUNNING": [
        "{label} is still running — {progress_phrase}. "
        "I'll tell you when it's done."],
    "T-GUARD-DONE": ["{label} finished {age_phrase}. {result_headline}"],
    "T-GUARD-FAILED": [
        "{label} didn't finish — {failure_reason_short}. "
        "Want me to try again?"],
    "T-GUARD-CANCELLED": ["{label} was cancelled {age_phrase}{by_whom_phrase}."],
    "T-GUARD-PLURAL": [
        "{label_1} is {state_phrase_1}, and {label_2} is {state_phrase_2}."],
    "T-GUARD-TAIL-LOW": [" — or were you asking something else?"],
    "T-HELD-BATCH": ["While you were away: {summary}"],
}

GUARD_BY_STATE = {
    "QUEUED": "T-GUARD-QUEUED", "BAND_HELD": "T-GUARD-QUEUED",
    "DRAFT": "T-GUARD-QUEUED", "HELD_DUPLICATE": "T-GUARD-QUEUED",
    "DISPATCHING": "T-GUARD-RUNNING", "RUNNING": "T-GUARD-RUNNING",
    "NEEDS_INPUT": "T-GUARD-RUNNING", "CANCEL_REQUESTED": "T-GUARD-RUNNING",
    "DONE": "T-GUARD-DONE", "FAILED": "T-GUARD-FAILED",
    "DISPATCH_FAILED": "T-GUARD-FAILED", "CANCELLED": "T-GUARD-CANCELLED",
}

# templates with no slots at all are pre-renderable statics
STATIC = {tid for tid, variants in TEMPLATES.items()
          if all("{" not in v for v in variants)}


def pick_variant(template_id, seed=""):
    variants = TEMPLATES[template_id]
    if len(variants) == 1:
        return variants[0], 0
    h = int(hashlib.sha256(
        f"{template_id}|{seed}".encode()).hexdigest()[:8], 16)
    idx = h % len(variants)
    return variants[idx], idx


def render(template_id, slots=None, seed=""):
    slots = {**DEFAULTS, **(slots or {})}
    fmt, idx = pick_variant(template_id, seed)
    return fmt.format(**slots), idx


def all_template_ids():
    return sorted(TEMPLATES)
