"""Pure numpy/OpenCV drawing. All functions return a new uint8 image of the input's shape."""
from collections import deque

import cv2
import numpy as np

from .gestures import MIN_CONF

CYAN, MAGENTA, DARK = (255, 255, 0), (255, 0, 255), (24, 14, 10)  # BGR
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
_rng = np.random.default_rng()


def _ok(frame):
    return frame is not None and frame.ndim == 3 and frame.size > 0


def _mask(m, shape):
    """Bool mask resized to frame (h, w), or None if absent/empty."""
    if m is None:
        return None
    m = np.asarray(m)
    if m.size == 0 or m.ndim != 2:
        return None
    h, w = shape[:2]
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def _union(people, shape):
    out = np.zeros(shape[:2], bool)
    for p in people:
        m = _mask(p.get("mask"), shape)
        if m is not None:
            out |= m
    return out


def _clip_box(box, shape):
    """Integer box clipped to the frame, or None if nothing is inside."""
    h, w = shape[:2]
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
    y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _skeleton(img, kp, conf, color, thick=2):
    if kp is None or conf is None or len(kp) < 17:
        return
    ok = [conf[i] >= MIN_CONF and np.all(np.isfinite(kp[i])) for i in range(17)]
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), color, thick, cv2.LINE_AA)
    for i in range(17):
        if ok[i]:
            cv2.circle(img, tuple(int(v) for v in kp[i]), 3, MAGENTA, -1, cv2.LINE_AA)


def _text(img, s, org, color=CYAN, scale=0.45):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _bar(img, x, y, w, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + 7), (80, 60, 40), 1)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(frac, 0, 1))), y + 7), color, -1)


def _fmt(v):
    return "--" if v is None else f"{v:.0f}"


def _panel(img, p, box):
    x1, y1 = box[0], box[1]
    pw, ph = 150, 92
    h, w = img.shape[:2]
    x = int(np.clip(x1, 0, max(w - pw, 0)))
    y = int(np.clip(y1 - ph - 4, 24, max(h - ph, 24)))  # 24: stay below the top bar
    roi = img[y:y + ph, x:x + pw]
    roi[:] = (roi * 0.3 + np.array(DARK) * 0.7).astype(np.uint8)  # translucent panel
    cv2.rectangle(img, (x, y), (min(x + pw, w) - 1, min(y + ph, h) - 1), CYAN, 1)
    tid = p.get("id")
    el, kn = p.get("elbows") or (None, None), p.get("knees") or (None, None)
    carrying = ", ".join(p.get("carrying") or []) or "-"
    _text(img, f"SUBJECT #{tid:02d}" if tid is not None else "SUBJECT #--", (x + 5, y + 14), MAGENTA, 0.5)
    _text(img, f"ELBOW {_fmt(el[0])}/{_fmt(el[1])}  KNEE {_fmt(kn[0])}/{_fmt(kn[1])}", (x + 5, y + 29), scale=0.35)
    _text(img, "POSTURE", (x + 5, y + 43), scale=0.35)
    _bar(img, x + 60, y + 36, 80, p.get("posture", 0) / 100, CYAN)
    _text(img, "ENERGY", (x + 5, y + 57), scale=0.35)
    _bar(img, x + 60, y + 50, 80, p.get("energy", 0) / 0.2, MAGENTA)
    _text(img, f"CARRY: {carrying}"[:26], (x + 5, y + 76), scale=0.35)


def draw_hud(frame, people, mode="HUD", fps=0.0, env=None):
    if not _ok(frame):
        return frame
    out = frame.copy()
    glow = np.zeros_like(out)
    union = _union(people, out.shape)
    if union.any():
        cnts, _ = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(glow, cnts, -1, CYAN, 2)
        out = cv2.add(out, cv2.GaussianBlur(glow, (0, 0), 6))
        out = cv2.add(out, glow)
    for p in people:
        _skeleton(out, p.get("kp"), p.get("conf"), CYAN)
        box = _clip_box(p.get("box"), out.shape)
        if box:
            _panel(out, p, box)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), DARK, -1)
    envs = f"{env[0].upper()} {env[1]:.0%}" if env else "--"
    _text(out, f"CYBERVISION  MODE:{mode}  FPS:{fps:.0f}  ENV:{envs}", (8, 15), CYAN, 0.5)
    return out


def hologram(frame, people):
    if not _ok(frame):
        return frame
    m = _union(people, frame.shape)
    if not m.any():
        return frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    holo = np.stack([gray * 1.0, gray * 0.95, gray * 0.15], axis=-1) + np.array([40, 30, 0])  # cyan tint
    holo[::3] *= 0.55  # scanlines
    holo = np.clip(holo, 0, 255).astype(np.uint8)
    rows = np.where(m.any(axis=1))[0]
    for y in _rng.choice(rows, size=min(len(rows), 6), replace=False):  # glitch: shift a band sideways
        band = slice(y, min(y + int(_rng.integers(2, 8)), frame.shape[0]))
        holo[band] = np.roll(holo[band], int(_rng.integers(-12, 13)), axis=1)
    out = frame.copy()
    out[m] = (frame[m] * 0.25 + holo[m] * 0.75).astype(np.uint8)  # slight transparency
    return out


class BackgroundModel:
    def __init__(self, rate=0.05):
        self.rate, self.bg = rate, None

    def update(self, frame, people):
        if self.bg is None or self.bg.shape != frame.shape:
            self.bg = frame.astype(np.float32)  # pixels under a person at start only clear once they move
            return
        free = ~cv2.dilate(_union(people, frame.shape).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.bg[free] += self.rate * (frame[free].astype(np.float32) - self.bg[free])

    def ghost(self, frame, people):
        if not _ok(frame):
            return frame
        self.update(frame, people)
        out = frame.copy()
        m = _union(people, frame.shape)
        out[m] = (frame[m] * 0.25 + self.bg[m] * 0.75).astype(np.uint8)
        for p in people:
            _skeleton(out, p.get("kp"), p.get("conf"), CYAN, 1)
        return out


class TimeEcho:
    def __init__(self, n=6, every=3):
        self.every, self.copies, self._i = every, deque(maxlen=n), 0

    def apply(self, frame, people):
        if not _ok(frame):
            return frame
        m = _union(people, frame.shape)
        if self._i % self.every == 0 and m.any():
            self.copies.append((frame.copy(), m))
        self._i += 1
        out = frame.copy()
        for i, (f, cm) in enumerate(self.copies):
            if f.shape != frame.shape:
                continue
            a = 0.5 * (i + 1) / (len(self.copies) + 1)  # older copies fade out
            tint = (f * 0.6 + np.array(MAGENTA) * 0.4).astype(np.uint8)
            out[cm] = (out[cm] * (1 - a) + tint[cm] * a).astype(np.uint8)
        out[m] = frame[m]
        return out
