# CyberVision

A sci-fi "body scanner" for your webcam. CyberVision runs four YOLOv8 models on the live image and turns the result into a neon HUD, a hologram, a ghost and a time echo, built with Ultralytics and Streamlit.


## The four YOLO skills

| Skill | Model | Job in the app | How often |
|---|---|---|---|
| Pose + tracking | yolov8n-pose | Skeleton, stable person IDs, angles, posture, motion energy, gestures | every frame |
| Segmentation | yolov8n-seg | Exact person outline for the effects | every frame |
| Detection | yolov8n | Objects near the hands ("CARRYING: cup") | every 2nd frame |
| Classification | yolov8n-cls | Scene label ("ENV: office 72%") | every 2 seconds |

## Modes

- **HUD**: glowing outline, neon skeleton, per-person data panel.
- **HOLOGRAM**: cyan tint, scan lines, glitch.
- **GHOST**: you turn see-through over a background the app learns by itself.
- **TIME ECHO**: fading copies of your last positions.

## Gestures

- Both wrists above the nose: next mode.
- T-pose: save a screenshot to `screenshots/`.

Each gesture needs a short hold, then pauses, so it never fires twice.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Model weights download automatically on first run (or put them in `models/`).
On macOS, allow camera access for your terminal in System Settings > Privacy & Security > Camera.

## Structure

```
app.py                  Streamlit UI and camera loop
cybervision/engine.py   model loading and per-frame scheduling
cybervision/gestures.py angles, posture, energy, gesture checks
cybervision/effects.py  HUD, hologram, ghost, time echo
tests/                  pytest
```

Run the tests with `pytest -q`.

Built for a computer vision course.
