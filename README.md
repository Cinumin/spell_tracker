# SpellTracker

Draw Chinese characters in the air with your fingertip. SpellTracker tracks a pinch gesture through a webcam, records the stroke you trace, and compares it against reference stroke data to recognize the character — triggering a 3D fire effect rendered live over the camera feed when you correctly draw "火" (fire).

## How it works

1. **Hand tracking** — [MediaPipe](https://developers.google.com/mediapipe)'s `HandLandmarker` runs on each webcam frame (asynchronously, in live-stream mode) to detect 21 hand landmarks per frame.
2. **Pinch detection** — A pinch gesture (index finger and thumb tips close together) is detected from the relative distance between the fingertip and thumb-tip landmarks. Two-level hysteresis thresholds (a stricter one to *start* a pinch, a looser one to *sustain* it) prevent a single noisy frame from fragmenting a stroke mid-draw.
3. **Point smoothing** — Raw pinch positions are smoothed frame-to-frame with a [One Euro Filter](https://cristal.univ-lille.fr/~casiez/1euro/) to reduce jitter without adding noticeable lag, then the finished stroke is rendered as a smooth curve using Catmull-Rom spline interpolation.
4. **Stroke buffering** — Points are buffered into the current stroke while pinching. A short grace period after the pinch ends (to absorb brief detection dropouts) closes out the stroke and starts a new one on the next pinch.
5. **Character recognition** — Completed strokes are compared against reference stroke data from the [Make Me a Hanzi](https://github.com/skishore/makemeahanzi) dataset:
   - Both the drawn and reference strokes are normalized into a shared coordinate space (translated, scaled, centered) so absolute position/size on screen doesn't matter.
   - Each stroke is resampled to a fixed number of points by arc length, so the comparison isn't skewed by how many frames were captured while drawing.
   - Strokes are matched to reference strokes via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), then compared with Dynamic Time Warping (DTW) to score shape similarity.
   - Characters with a different stroke count than the drawing are skipped; among the remaining candidates, the best (lowest-distance) match is returned.
6. **Rendering** — The webcam feed and drawn strokes are rendered to an OpenGL texture via [moderngl](https://github.com/moderngl/moderngl) and [moderngl-window](https://github.com/moderngl/moderngl-window). Recognizing "火" triggers an additively-blended fireball shader overlay on top of the live feed.

## Controls

| Key | Action |
|-----|--------|
| Draw a stroke | Pinch your index finger and thumb, then move your hand |
| `R` | Attempt to identify the drawn character against the reference set |
| `C` | Clear all drawn strokes and reset the fire effect |
| `q` / close window | Quit |

Currently recognized characters: **火** (fire) and **土** (earth).

## Installation

Requires Python 3.10+ and a webcam.

```bash
pip install -r requirements.txt
```

> **Note:** The webcam capture currently opens via OpenCV's AVFoundation backend (`cv.CAP_AVFOUNDATION`), so out of the box this runs on macOS. On other platforms, swap the capture backend in `spell_tracker/hand_tracker.py`.

## Usage

```bash
python -m spell_tracker
```

This opens a window showing your webcam feed. Pinch and move your fingertip to draw a stroke; release the pinch to end it. Press `R` to check your drawing against the reference characters, or `C` to start over.

## Project structure

```
spell_tracker/
├── spell_tracker/
│   ├── hand_tracker.py   # Pinch detection, stroke buffering/smoothing, OpenGL render loop
│   ├── geometry.py       # Stroke normalization, resampling, DTW + Hungarian matching, recognition
│   ├── shader.py         # Fireball fragment shader (GLSL)
│   ├── utils.py          # Hand landmark visualization helpers
│   └── data/
│       └── characters.json  # Reference stroke data (Make Me a Hanzi)
├── models/
│   └── hand_landmarker.task  # MediaPipe hand-landmark model
├── test/                 # pytest unit tests (geometry + hand tracking logic)
└── experiments/           # Scratch scripts used while prototyping DTW and the One Euro Filter
```

## Testing

```bash
pytest
```

Tests cover the geometry/matching pipeline (`test_geometry.py`) and the stroke-tracking/pinch-detection logic (`test_hand_tracker.py`), independent of any webcam or live MediaPipe session.

## Known limitations

- Recognition currently requires the drawn stroke count to exactly match a candidate character's stroke count — no partial credit for close-but-not-quite attempts.
- Only two characters are wired up to reference data and effects so far ("火" triggers the fire effect; "土" is recognized but has no visual effect yet).
- The One Euro Filter's `beta` parameter is tuned as a starting point for hand-tuning on a real webcam, not empirically calibrated.

## Credits

- Reference character stroke data from [Make Me a Hanzi](https://github.com/skishore/makemeahanzi) (`skishore/makemeahanzi`).
- Hand tracking via [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
