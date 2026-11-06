# tools/avatar — cut-and-rig avatar animator

Animates a single piece of character artwork so it can speak an audio clip.
The face is never redrawn or remodelled: the jaw, eyelid and ear pieces are
**cut out of the artwork itself** and moved. What you get back is the same
artwork, rigged.

## What it does

`avatar.py` builds a 2.5D rig from one square artwork image:

- **Jaw** — a piece cut from the muzzle/chin area, dropped by an amount
  driven by a viseme cue track. The original mouth inside the jaw area of
  the base image is melted into the surrounding fur (a flipped, stretched
  copy of the beard below the jaw), so a dropped jaw never reveals a
  second, unmoved mouth. A soft shadow is painted into the exposed strip,
  and the dark mouth interior (with a tongue hint at wide openings) is
  drawn at the dropped mouth position.
- **Eyelids** — the fur directly above each eye, slid down over the eye to
  blink. Blink timing is randomised (2.2–5.5 s intervals, occasional double
  blinks) from a fixed seed, so renders are reproducible.
- **Ears** — two ear pieces rotate a degree or two around their base:
  slow idle drift plus a small outward perk at speech onsets. Under each
  ear the base is filled with blurred surroundings, masked to only the
  thin ring the piece does not cover, so there is no hole and no patch.
- **Body life** — the whole frame breathes (scale, ~4.1 s period) and
  sways (rotation ±0.35°, ~5.2 s period; drift ~7.3 s period), at idle and
  while speaking.

Mouth shape comes from the cue track: each Rhubarb-style letter (A–H, X)
maps to `(open, width, round)` parameters, which are exponentially
smoothed (attack τ ≈ 45 ms, release τ ≈ 80 ms) at render rate, then
modulated by the clip's amplitude envelope — except closures (A, X),
which stay closed regardless of envelope.

## Requirements

- Python 3 with Pillow and numpy
- `ffmpeg` on PATH (audio decoding + video muxing)

## Usage

```sh
python3 avatar.py --art artwork.png --audio clip.wav \
    --track tracks.json --clip sentence1 --out out/
```

Outputs in the output directory:

- `<clip>.mp4` — H.264 video with the audio muxed in (AAC).
- `frame-strip.png` — four 200 px frames: closed / mid / open / round,
  picked from the render by cue letter (mid-blink frames are avoided).
- `frames/` — every rendered PNG frame (only with `--frames`).

Options: `--fps 30`, `--size 800` (working canvas; the artwork is resized
to it), `--idle SECONDS` (when no `--audio` is given, render a silent
idle loop of that length instead).

### Cue-track format

Either a single track:

```json
{"dur": 5.46, "cues": [{"s": 0.0, "e": 0.149, "v": "B"}, ...]}
```

or a map of named tracks, with `--clip NAME` selecting the entry:

```json
{"sentence1": {"dur": 5.46, "cues": [...]}, "serious": {"dur": 3.69, "cues": [...]}}
```

`s`/`e` are seconds, `v` is a Rhubarb viseme letter: X (rest/silence),
A (closed), B, C, D (widest open), E, F (rounded/pursed), G, H.

If `--track` is omitted, a track is synthesised from the audio amplitude
envelope (level thresholds mapped to X/B/C/D letters). It speaks, but a
real cue track reads noticeably better — the synthesised one has no
rounded shapes.

## Honest limits

This is 2.5D cut-and-rig, not a 3D model: expect about four readable mouth
shapes (closed / mid / open / round), not a full viseme set. At phone size
(≈200 px) the speech reads plainly; at full 800 px the jaw seam and the
softened strip above a dropped jaw are visible on close inspection, and
the ear pieces carry a faint soft halo at their edges where the blurred
fill shows through the feathered mask. The rig geometry (jaw box, eye and
ear positions) is tuned to one reference artwork; other artwork will need
the constants at the top of `avatar.py` adjusted.
