#!/usr/bin/env python3
"""Cut-and-rig avatar animator.

Takes a single artwork image, cuts rig pieces (jaw, eyelids, ears) from the
artwork itself -- nothing is redrawn or remodelled -- and animates them:

  * the jaw is driven by a viseme cue track (Rhubarb-style letters A-H + X)
    mapped to (open, width, round) mouth parameters, exponentially smoothed
    (fast attack, slower release) and modulated by the audio amplitude
    envelope;
  * eyelids blink on a seeded, randomised-but-reproducible schedule;
  * ears drift with slow noise and perk briefly at speech onsets;
  * the whole frame breathes and sways gently, at idle and while speaking.

Usage:
    python3 avatar.py --art ART.png --audio CLIP.wav --track TRACK.json \
        --clip sentence1 --out OUTDIR [--fps 30] [--size 800] [--frames]

    python3 avatar.py --art ART.png --out OUTDIR --idle 4.5   # silent idle

Outputs in OUTDIR:
    <clip>.mp4       H.264/AAC video (frames muxed with the audio, if any)
    frame-strip.png  four 200px frames: closed / mid / open / round
    frames/          individual PNG frames (only with --frames)

Requirements: Python 3, Pillow, numpy, ffmpeg (on PATH).
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

# --------------------------------------------------------------------------
# Rig geometry, as fractions of the square canvas size S.
# Jaw / eyes follow the values proven in the original proof render; the ear
# boxes were verified against the artwork with labelled test crops.
# --------------------------------------------------------------------------
JAW_BOX = (0.345, 0.520, 0.655, 0.628)          # x0, y0, x1, y1
NOTCH_Y = 0.553                                  # mouth notch centre in the art
EYES = [(0.412, 0.338), (0.588, 0.338)]         # (cx, cy) per eye
EAR_BOXES = {
    "left": (0.09, 0.09, 0.38, 0.41),
    "right": (0.60, 0.07, 0.88, 0.39),
}
EAR_PIVOTS = {                                   # top-inner base of each ear
    "left": (0.33, 0.14),
    "right": (0.66, 0.12),
}
MAXDROP_AT_800 = 40.0                            # jaw drop in px at S=800
BLINK_SEED = 20261105

# Rhubarb letter -> (open, width, round)
LETTER_PARAMS = {
    "X": (0.00, 1.00, 0.00),
    "A": (0.02, 0.98, 0.00),
    "B": (0.28, 0.95, 0.05),
    "C": (0.52, 1.05, 0.00),
    "D": (0.92, 1.10, 0.00),
    "E": (0.50, 0.80, 0.55),
    "F": (0.32, 0.62, 1.00),
    "G": (0.38, 0.90, 0.10),
    "H": (0.60, 1.00, 0.00),
}
CLOSURES = {"A", "X"}


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def decode_to_wav(audio_path, tmpdir):
    """Return path to a 16-bit PCM wav for audio_path (via ffmpeg if needed)."""
    if audio_path.lower().endswith(".wav"):
        return audio_path
    out = os.path.join(tmpdir, "decoded.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", audio_path,
         "-ar", "22050", "-ac", "1", out],
        check=True)
    return out


def read_pcm(wav_path):
    """Read a wav file -> (mono float32 samples in [-1, 1], sample rate)."""
    with wave.open(wav_path) as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported wav sample width: {sw} bytes")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def amplitude_envelope(pcm, sr, fps, n_frames):
    """Per-frame RMS envelope, normalised to the 97th percentile, gated."""
    env = np.zeros(n_frames, dtype=np.float64)
    per = max(1, int(round(sr / fps)))
    for i in range(n_frames):
        seg = pcm[i * per:(i + 1) * per]
        if len(seg):
            env[i] = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.percentile(env, 97)) if n_frames else 0.0
    if peak > 1e-6:
        env /= peak
    env = np.clip(env, 0.0, 1.0)
    env[env < 0.07] = 0.0
    return env


def synthesize_track(env, fps):
    """Build a cue track from an amplitude envelope (used when no track given).

    Envelope level -> letter: silence X, quiet B, mid C, loud D. Consecutive
    identical letters are merged into single cues.
    """
    letters = []
    for e in env:
        if e < 0.08:
            letters.append("X")
        elif e < 0.30:
            letters.append("B")
        elif e < 0.55:
            letters.append("C")
        else:
            letters.append("D")
    cues = []
    for i, v in enumerate(letters):
        s, e = i / fps, (i + 1) / fps
        if cues and cues[-1]["v"] == v:
            cues[-1]["e"] = e
        else:
            cues.append({"s": s, "e": e, "v": v})
    dur = len(letters) / fps if letters else 0.0
    return dur, cues


# --------------------------------------------------------------------------
# Cue tracks
# --------------------------------------------------------------------------
def load_track(track_path, clip_name):
    """Load a cue track JSON.

    Accepts either a single track {"dur": ..., "cues": [{s, e, v}, ...]} or a
    map of named tracks {"name": {"dur":..., "cues":[...]}, ...}; with a map,
    --clip selects the entry (required when the map has several entries).
    Returns (duration, cues).
    """
    with open(track_path) as f:
        data = json.load(f)
    if "cues" in data:
        track = data
    else:
        if clip_name is not None:
            if clip_name not in data:
                raise ValueError(f"clip {clip_name!r} not in track file; "
                                 f"available: {sorted(data)}")
            track = data[clip_name]
        elif len(data) == 1:
            track = next(iter(data.values()))
        else:
            raise ValueError("track file holds several named tracks; "
                             "pass --clip NAME to select one")
    cues = [{"s": float(c["s"]), "e": float(c["e"]), "v": str(c["v"])}
            for c in track["cues"]]
    dur = float(track.get("dur", cues[-1]["e"] if cues else 0.0))
    return dur, cues


def letter_at(cues, t):
    """Viseme letter active at time t (X outside the tracked range)."""
    for c in cues:
        if c["s"] <= t < c["e"]:
            return c["v"]
    return "X"


def speech_onsets(cues):
    """Times where speech starts after a closure / silence."""
    onsets = []
    prev = "X"
    for c in cues:
        if c["v"] not in CLOSURES and prev in CLOSURES:
            onsets.append(c["s"])
        prev = c["v"]
    return onsets


# --------------------------------------------------------------------------
# Blinks (seeded -> reproducible renders)
# --------------------------------------------------------------------------
def build_blink_schedule(duration, seed=BLINK_SEED):
    """Return a list of blink start times.

    Intervals are randomised in [2.2, 5.5] s from a seeded RNG; ~18% of
    blinks are doubles (a second blink ~0.3 s later).
    """
    rng = random.Random(seed)
    starts = []
    t = rng.uniform(0.9, 1.8)
    while t < duration:
        starts.append(t)
        if rng.random() < 0.18 and t + 0.32 < duration:
            starts.append(t + 0.32)
        t += rng.uniform(2.2, 5.5)
    return starts


def _smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def blink_amount(t, starts):
    """0 = eyes open, 1 = fully closed. Ease down 90 ms, hold 40 ms, up 130 ms."""
    amount = 0.0
    for bt in starts:
        d = t - bt
        if d < 0.0:
            continue
        if d < 0.09:
            amount = max(amount, _smoothstep(d / 0.09))
        elif d < 0.13:
            amount = max(amount, 1.0)
        elif d < 0.26:
            amount = max(amount, 1.0 - _smoothstep((d - 0.13) / 0.13))
    return amount


# --------------------------------------------------------------------------
# Rig: pieces cut from the artwork
# --------------------------------------------------------------------------
def _feathered_mask(size, radius, blur):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return m.filter(ImageFilter.GaussianBlur(blur))


class Rig:
    def __init__(self, art_path, size):
        self.S = size
        self.scale = size / 800.0
        base = Image.open(art_path).convert("RGB")
        if base.size != (size, size):
            base = base.resize((size, size), Image.LANCZOS)
        self.base = base

        # Jaw piece + feathered mask.
        S = size
        self.jx0, self.jy0 = int(JAW_BOX[0] * S), int(JAW_BOX[1] * S)
        self.jx1, self.jy1 = int(JAW_BOX[2] * S), int(JAW_BOX[3] * S)
        self.jaw = base.crop((self.jx0, self.jy0, self.jx1, self.jy1))
        self.jaw_mask = _feathered_mask(self.jaw.size, int(26 * self.scale),
                                        max(1.0, 8 * self.scale))

        # Base with the original mouth softened away inside the jaw box:
        # when the jaw piece drops, the revealed base must not show a
        # second, unmoved mouth. A strong local blur melts the smile/notch
        # into the surrounding fur; at rest the jaw piece covers the area
        # exactly, so the artwork is unchanged.
        erased = base.copy()
        ex0 = max(0, self.jx0 - int(4 * self.scale))
        ey0 = max(0, self.jy0 - int(4 * self.scale))
        ex1 = min(S, self.jx1 + int(4 * self.scale))
        ey1 = min(S, self.jy1 + int(4 * self.scale))
        # Fill source: the beard fur BELOW the jaw box, flipped and
        # stretched upward. (Rows above the box are unusable at the centre:
        # they contain the nose, and stretching them down paints a dark
        # nose-coloured column into the gap -- caught in test renders.)
        src_h = int(36 * self.scale)
        src = base.crop((ex0, ey1, ex1, min(S, ey1 + src_h)))
        src = src.transpose(Image.FLIP_TOP_BOTTOM)
        region = src.resize((ex1 - ex0, ey1 - ey0), Image.BICUBIC)
        region = Image.fromarray((np.asarray(region).astype(np.float32)
                                  * 0.95).astype(np.uint8))
        emask = _feathered_mask(region.size, int(30 * self.scale),
                                max(1.0, 12 * self.scale))
        erased.paste(region, (ex0, ey0), emask)
        self.base_erased = erased

        # Ears: the ear pieces, plus a "filled" base used underneath them.
        # The fill is a blurred copy of the whole frame, masked to ONLY the
        # thin ring the ear piece does not cover (inverse of the piece's
        # own alpha, softened). At rest the composite is therefore identical
        # to the original art; when a piece rotates, the few pixels it
        # vacates show blurred surroundings instead of a rectangular patch.
        blurred_full = erased.filter(
            ImageFilter.GaussianBlur(int(16 * self.scale)))
        self.filled_base = erased.copy()
        self.ears = {}
        for name, (fx0, fy0, fx1, fy1) in EAR_BOXES.items():
            x0, y0, x1, y1 = int(fx0 * S), int(fy0 * S), int(fx1 * S), int(fy1 * S)
            piece = base.crop((x0, y0, x1, y1)).convert("RGBA")
            pmask = _feathered_mask(piece.size, int(46 * self.scale),
                                    max(1.0, 5 * self.scale))
            piece.putalpha(pmask)
            alpha_full = Image.new("L", (S, S), 0)
            alpha_full.paste(pmask, (x0, y0))
            ring = alpha_full.filter(
                ImageFilter.GaussianBlur(max(1.0, 5 * self.scale)))
            ring = ring.point(lambda a: 255 - a)
            pad = int(0.03 * S)
            region_mask = Image.new("L", (S, S), 0)
            rd = ImageDraw.Draw(region_mask)
            rd.rectangle([max(0, x0 - pad), max(0, y0 - pad),
                          min(S, x1 + pad), min(S, y1 + pad)], fill=255)
            region_mask = region_mask.filter(
                ImageFilter.GaussianBlur(max(1.0, 6 * self.scale)))
            ring = ImageChops.multiply(ring, region_mask)
            self.filled_base = Image.composite(blurred_full,
                                               self.filled_base, ring)
            pvx, pvy = EAR_PIVOTS[name]
            pivot = (int(pvx * S) - x0, int(pvy * S) - y0)
            self.ears[name] = {"img": piece, "pos": (x0, y0), "pivot": pivot}

        # Eyelids: a small piece of the fur directly above each eye, slid
        # down over the eye to blink. Real art texture (including the brow
        # line) reads as a lid; a flat fur-coloured ellipse does not.
        self.eyes = [(int(fx * S), int(fy * S)) for fx, fy in EYES]
        self.lids = []
        hw = int(47 * self.scale)
        lh = int(64 * self.scale)
        for cx, cy in self.eyes:
            piece = base.crop((cx - hw, cy - lh, cx + hw, cy)).convert("RGBA")
            lmask = _feathered_mask(piece.size, int(20 * self.scale),
                                    max(1.0, 4 * self.scale))
            piece.putalpha(lmask)
            self.lids.append({"img": piece, "x": cx - hw, "y": cy - lh,
                              "h": lh})

    # -- one frame ---------------------------------------------------------
    def render(self, open_, width, round_, blink, ear_angles, t):
        S = self.S
        sc = self.scale
        frame = self.filled_base.copy()

        # Ears (behind everything else that moves).
        for name in ("left", "right"):
            ear = self.ears[name]
            ang = ear_angles[name]
            if abs(ang) > 0.01:
                piece = ear["img"].rotate(ang, resample=Image.BICUBIC,
                                          center=ear["pivot"])
            else:
                piece = ear["img"]
            frame.paste(piece, ear["pos"], piece)

        # Soft shadow in the strip the dropping jaw exposes: darkest at
        # the jaw's new top edge, fading upward -- sells the recess.
        drop0 = open_ * MAXDROP_AT_800 * sc * (1.0 - 0.30 * round_)
        dy0 = int(round(drop0))
        if dy0 > 2:
            arr = np.asarray(frame).astype(np.float32)
            y0s, y1s = self.jy0, min(S, self.jy0 + dy0)
            xs = np.arange(arr.shape[1], dtype=np.float32)
            win = np.clip((xs - (self.jx0 - 12 * sc)) / (12 * sc), 0, 1) * \
                np.clip(((self.jx1 + 12 * sc) - xs) / (12 * sc), 0, 1)
            fac = np.linspace(0.96, 0.84, max(1, y1s - y0s),
                              dtype=np.float32)[:, None]
            arr[y0s:y1s] *= (1.0 - win[None, :] * (1.0 - fac))[..., None]
            frame = Image.fromarray(
                np.clip(arr, 0, 255).astype(np.uint8))

        # Jaw piece first, then the mouth interior ON TOP of it, centred on
        # the dropped mouth notch: one dark opening where the mouth is,
        # growing with the drop. (An interior drawn under the piece is
        # hidden by it; one drawn at the unmoved notch double-prints the
        # mouth. Both failure modes were caught in test renders.)
        drop = open_ * MAXDROP_AT_800 * sc * (1.0 - 0.30 * round_)
        dy = int(round(drop))
        # Round narrows the jaw itself a little. (Width is NOT multiplied
        # in here: width already narrows the interior, and scaling the jaw
        # piece by e.g. F's 0.62 would shrink the chin to half width and
        # leave the un-narrowed original visible as a ghost box around it.)
        jaw_w_scale = 1.0 - 0.18 * round_
        if abs(jaw_w_scale - 1.0) > 0.01:
            nw = max(8, int(self.jaw.size[0] * jaw_w_scale))
            jaw = self.jaw.resize((nw, self.jaw.size[1]), Image.BICUBIC)
            jmask = self.jaw_mask.resize((nw, self.jaw.size[1]), Image.BICUBIC)
            jx = self.jx0 + (self.jaw.size[0] - nw) // 2
        else:
            jaw, jmask, jx = self.jaw, self.jaw_mask, self.jx0
        frame.paste(jaw, (jx, self.jy0 + dy), jmask)
        if dy > 2:
            cx = S // 2
            wpx = int(0.145 * S * width * (1.0 - 0.45 * round_))
            cy = int(NOTCH_Y * S) + dy
            hh = int(24 * sc + dy)
            layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            ld.ellipse([cx - wpx // 2, cy - hh // 2,
                        cx + wpx // 2, cy + hh // 2], fill=(62, 32, 20, 255))
            if dy > 16 * sc:  # tongue hint low in the opening
                ld.ellipse([cx - wpx // 4, cy + hh // 2 - int(dy * 0.42),
                            cx + wpx // 4, cy + hh // 2 + 2],
                           fill=(165, 84, 62, 255))
            layer = layer.filter(ImageFilter.GaussianBlur(max(1.0, 2 * sc)))
            frame = Image.alpha_composite(frame.convert("RGBA"),
                                          layer).convert("RGB")

        # Eyelids: slide each supra-eye fur piece down over its eye.
        if blink > 0.001:
            cover = int(44 * sc * blink)
            for lid in self.lids:
                frame.paste(lid["img"], (lid["x"], lid["y"] + cover),
                            lid["img"])

        # Whole-frame life: overscan, breathing scale, sway rotation + drift.
        zoom = 1.03 * (1.0 + 0.006 * math.sin(2.0 * math.pi * t / 4.1))
        big_size = int(round(S * zoom)) + 4
        big = frame.resize((big_size, big_size), Image.BICUBIC)
        ang = 0.35 * math.sin(2.0 * math.pi * t / 5.2)
        if abs(ang) > 0.001:
            big = big.rotate(ang, resample=Image.BICUBIC)
        tx = int(round(2 * sc * math.sin(2.0 * math.pi * t / 7.3)))
        left = (big_size - S) // 2 + tx
        top = (big_size - S) // 2
        left = min(max(0, left), big_size - S)
        return big.crop((left, top, left + S, top + S))


# --------------------------------------------------------------------------
# Animation driver
# --------------------------------------------------------------------------
def animate(rig, cues, env, duration, fps):
    """Yield (frame_image, params_dict) for each frame.

    Cue targets are exponentially smoothed (attack tau 45 ms, release tau
    80 ms). The amplitude envelope scales openness -- except closures
    (A / X), whose targets are already ~0 and stay closed regardless.
    """
    n_frames = max(1, int(round(duration * fps)))
    onsets = speech_onsets(cues)
    blinks = build_blink_schedule(duration)
    cur = [0.0, 1.0, 0.0]  # open, width, round -- start at rest (X)
    dt = 1.0 / fps
    for i in range(n_frames):
        t = i / fps
        letter = letter_at(cues, t) if cues else "X"
        tgt = list(LETTER_PARAMS[letter])
        if letter not in CLOSURES:
            e = float(env[i]) if env is not None and i < len(env) else 1.0
            tgt[0] = tgt[0] * (0.30 + 0.70 * e)
        for k in range(3):
            tau = 0.045 if tgt[k] > cur[k] else 0.080
            cur[k] += (tgt[k] - cur[k]) * (1.0 - math.exp(-dt / tau))
        open_, width, round_ = cur

        # Ears: slow idle noise + a brief outward perk at speech onsets.
        perk = 0.0
        for ot in onsets:
            d = t - ot
            if 0.0 <= d < 0.45:
                perk = max(perk, math.exp(-d / 0.14))
        n_l = math.sin(2.0 * math.pi * t / 6.7 + 1.3) + 0.55 * math.sin(2.0 * math.pi * t / 2.9)
        n_r = math.sin(2.0 * math.pi * t / 7.9 + 4.1) + 0.55 * math.sin(2.0 * math.pi * t / 3.4 + 0.7)
        ear_angles = {
            "left": 0.75 * n_l + 1.5 * perk,    # + = tip swings outward (CCW)
            "right": 0.75 * n_r - 1.5 * perk,   # - = tip swings outward (CW)
        }
        blink = blink_amount(t, blinks)
        frame = rig.render(open_, width, round_, blink, ear_angles, t)
        yield frame, {"t": t, "letter": letter, "open": open_,
                      "width": width, "round": round_, "blink": blink}


def pick_strip_frames(records):
    """Choose (closed, mid, open, round) frame indices from render records.

    Frames caught mid-blink are avoided (a blink in the "closed mouth"
    panel reads as a rig error, not as life).
    """
    awake = [i for i, r in enumerate(records) if r.get("blink", 0.0) < 0.25]
    pool = awake if len(awake) >= 4 else list(range(len(records)))
    used = []

    def best(pred, key):
        # Keep panels visually distinct: stay >= 12 frames from picks
        # already made, unless that leaves no candidate at all.
        for source in (pool, list(range(len(records)))):
            cand = [i for i in source if pred(records[i])
                    and all(abs(i - u) >= 12 for u in used)]
            if not cand:
                cand = [i for i in source if pred(records[i])]
            if cand:
                pick = max(cand, key=key) if key else cand[len(cand) // 2]
                used.append(pick)
                return pick
        return None

    opened = best(lambda r: r["letter"] == "D", lambda i: records[i]["open"])
    if opened is None:
        opened = max(range(len(records)), key=lambda i: records[i]["open"])
        used.append(opened)
    rounded = best(lambda r: r["letter"] in ("F", "E"),
                   lambda i: records[i]["round"])
    if rounded is None:
        rounded = max(range(len(records)), key=lambda i: records[i]["round"])
        used.append(rounded)
    mid = best(lambda r: r["letter"] in ("C", "H"), None)
    if mid is None:
        mid = min(range(len(records)),
                  key=lambda i: abs(records[i]["open"] - 0.5))
        used.append(mid)
    closed = best(lambda r: r["letter"] in CLOSURES, None)
    if closed is None:
        closed = min(range(len(records)), key=lambda i: records[i]["open"])
        used.append(closed)
    return [closed, mid, opened, rounded]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Animate artwork with a cut-and-rig jaw/eyelid/ear rig "
                    "driven by a viseme cue track.")
    ap.add_argument("--art", required=True, help="artwork image (square)")
    ap.add_argument("--audio", help="audio clip (.wav, or anything ffmpeg reads)")
    ap.add_argument("--track", help="cue-track JSON (single track or named map)")
    ap.add_argument("--clip", help="track name (selects an entry in a named "
                                   "map; also names the outputs)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--frames", action="store_true",
                    help="also write individual PNG frames to OUTDIR/frames/")
    ap.add_argument("--idle", type=float, default=4.5,
                    help="seconds of silent idle to render when no --audio "
                         "is given (default 4.5)")
    args = ap.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    clip_name = args.clip
    cues = None
    track_dur = None
    if args.track:
        track_dur, cues = load_track(args.track, args.clip)

    tmpdir = tempfile.mkdtemp(prefix="avatar-")
    try:
        env = None
        duration = None
        audio_for_mux = None
        if args.audio:
            wav_path = decode_to_wav(args.audio, tmpdir)
            pcm, sr = read_pcm(wav_path)
            audio_dur = len(pcm) / sr
            duration = track_dur if track_dur is not None else audio_dur
            n_frames = max(1, int(round(duration * args.fps)))
            env = amplitude_envelope(pcm, sr, args.fps, n_frames)
            audio_for_mux = args.audio
            if clip_name is None:
                clip_name = os.path.splitext(os.path.basename(args.audio))[0]
        else:
            duration = args.idle
            n_frames = max(1, int(round(duration * args.fps)))
            env = np.zeros(n_frames)
            if clip_name is None:
                clip_name = "idle"
        if cues is None:
            _, cues = synthesize_track(env, args.fps)

        rig = Rig(args.art, args.size)
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)
        records = []
        keep = {}
        strip_idx = None
        all_frames = []
        for i, (frame, rec) in enumerate(
                animate(rig, cues, env, duration, args.fps)):
            frame.save(os.path.join(frames_dir, f"f{i:04d}.png"))
            records.append(rec)
            all_frames.append(frame)
        strip_idx = pick_strip_frames(records)
        keep = {i: all_frames[i] for i in strip_idx}

        # Video.
        mp4_path = os.path.join(args.out, f"{clip_name}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(args.fps),
               "-i", os.path.join(frames_dir, "f%04d.png")]
        if audio_for_mux:
            cmd += ["-i", audio_for_mux]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(args.fps)]
        if audio_for_mux:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [mp4_path]
        subprocess.run(cmd, check=True)

        # Frame strip: closed / mid / open / round at 200 px each.
        strip = Image.new("RGB", (200 * 4, 200), "white")
        for k, fi in enumerate(strip_idx):
            strip.paste(keep[fi].resize((200, 200), Image.LANCZOS), (200 * k, 0))
        strip_path = os.path.join(args.out, "frame-strip.png")
        strip.save(strip_path)

        if args.frames:
            dest = os.path.join(args.out, "frames")
            os.makedirs(dest, exist_ok=True)
            for name in sorted(os.listdir(frames_dir)):
                shutil.copy(os.path.join(frames_dir, name),
                            os.path.join(dest, name))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"wrote {mp4_path} and {strip_path} "
          f"({len(records)} frames @ {args.fps} fps, strip frames {strip_idx})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
