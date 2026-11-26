import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from cybervision import effects, gestures
from cybervision.camera import Camera
from cybervision.engine import Engine

MODES = ["HUD", "HOLOGRAM", "GHOST", "TIME ECHO"]
SHOT_DIR = Path(__file__).resolve().parent / "screenshots"  # fixed folder, never user-supplied
WIDTH = 640

st.set_page_config(page_title="CyberVision", layout="wide")


def save_screenshot(img):
    SHOT_DIR.mkdir(exist_ok=True)
    path = SHOT_DIR / f"cybervision_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), img)
    return path.name


ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("mode", MODES[0])
ss.setdefault("up", gestures.GestureTrigger())
ss.setdefault("tpose", gestures.GestureTrigger())
ss.setdefault("bg", effects.BackgroundModel())
ss.setdefault("echo", effects.TimeEcho())


def toggle():
    ss.running = not ss.running


with st.sidebar:
    st.title("CyberVision")
    st.button("Stop" if ss.running else "Start", on_click=toggle, use_container_width=True)
    cam = st.number_input("Camera index", 0, 9, 0)
    conf = st.slider("Confidence", 0.05, 0.95, 0.4, 0.05)
    # no widget key: gestures change ss.mode mid-loop without a rerun (which would reopen the camera)
    ss.mode = st.selectbox("Mode", MODES, index=MODES.index(ss.mode))
    mode_label = st.empty()
    st.caption("Hands up: next mode. T-pose: screenshot.")
    skills = {
        "pose": st.checkbox("Pose + tracking", True),
        "seg": st.checkbox("Segmentation", True),
        "detect": st.checkbox("Detection", True),
        "cls": st.checkbox("Classification", True),
    }
    stats = st.empty()

view = st.empty()
if not ss.running:
    view.info("Press Start in the sidebar.")
    st.stop()

cap = Camera(int(cam))
try:
    if not cap.isOpened():
        ss.running = False
        st.error(f"Cannot open camera {int(cam)}. Check macOS camera permission "
                 "(System Settings > Privacy & Security > Camera) or try another index.")
        st.stop()
    if "engine" not in ss:  # per session: the tracker and motion history must not mix between browser tabs
        ss.engine = Engine()
    engine = ss.engine
    fps, last = 0.0, time.perf_counter()
    fails = dark = seq = 0
    last_new = time.perf_counter()
    while True:
        ok, frame, new = cap.read()
        if not ok:
            fails += 1  # macOS cameras often fail the first reads while warming up or while the permission prompt is open
            if fails < 30:
                time.sleep(0.1)
                continue
            ss.running = False
            st.error("Lost the camera: no frames for 3 seconds. Check camera permission, "
                     "close other apps using it, or try another camera index.")
            break
        fails = 0
        if new == seq:  # no new camera frame yet
            if time.perf_counter() - last_new > 3:  # camera died mid-stream (unplugged, taken by another app)
                ss.running = False
                st.error("Lost the camera: no new frames for 3 seconds.")
                break
            time.sleep(0.005)
            continue
        seq, last_new = new, time.perf_counter()
        dark = dark + 1 if frame.mean() < 2 else 0  # e.g. an asleep iPhone (Continuity Camera) sends black frames
        if dark == 50:
            st.warning(f"Camera {int(cam)} only sends black frames. Try another camera index in the sidebar.")
        h, w = frame.shape[:2]
        frame = cv2.flip(cv2.resize(frame, (WIDTH, int(h * WIDTH / w))), 1)
        res = engine.process(frame, {"conf": conf, **skills})
        people = res["people"]

        mode = ss.mode
        if mode == "HOLOGRAM":
            out = effects.hologram(frame, people)
        elif mode == "GHOST":
            out = ss.bg.ghost(frame, people)
        elif mode == "TIME ECHO":
            out = ss.echo.apply(frame, people)
        else:
            out = frame
        out = effects.draw_hud(out, people, mode, fps, res["env"]) if mode == "HUD" else out
        view.image(out, channels="BGR", output_format="JPEG")

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
        last = now
        ms = res["ms"]
        stats.markdown(f"**FPS** {fps:.1f}  \n" + "  \n".join(f"{k}: {v:.0f} ms" for k, v in ms.items()))

        if ss.tpose.update(any(gestures.is_t_pose(p["kp"], p["conf"]) for p in people)):
            st.toast(f"Saved screenshots/{save_screenshot(out)}")
        if ss.up.update(any(gestures.is_hands_up(p["kp"], p["conf"]) for p in people)):
            ss.mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            mode_label.success(f"Gesture → {ss.mode}")
finally:
    cap.stop()
