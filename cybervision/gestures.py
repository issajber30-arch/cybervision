"""Pure functions on COCO 17-keypoint arrays: kp (17,2) xy and conf (17,)."""
import time

import numpy as np

MIN_CONF = 0.3
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = (
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


def _pt(kp, conf, i):
    """Keypoint i as a float array, or None if missing/low confidence/invalid."""
    try:
        if kp is None or conf is None or len(kp) <= i or len(conf) <= i or conf[i] < MIN_CONF:
            return None
        p = np.asarray(kp[i], dtype=float)
        return p if p.shape == (2,) and np.all(np.isfinite(p)) else None
    except (TypeError, ValueError):
        return None


def _shoulder_width(kp, conf):
    a, b = _pt(kp, conf, L_SH), _pt(kp, conf, R_SH)
    if a is None or b is None:
        return None
    w = float(np.linalg.norm(a - b))
    return w if w > 1e-6 else None


def angle(a, b, c):
    """Angle ABC in degrees (at b), or None for missing/degenerate points."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = np.asarray(a, float) - np.asarray(b, float), np.asarray(c, float) - np.asarray(b, float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def _angles(kp, conf, triples):
    return tuple(angle(*(_pt(kp, conf, i) for i in t)) for t in triples)


def elbow_angles(kp, conf):
    """(left, right) elbow angles in degrees; None where unknown."""
    return _angles(kp, conf, [(L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)])


def knee_angles(kp, conf):
    return _angles(kp, conf, [(L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)])


def posture_score(kp, conf):
    """0-100: level shoulders and head centered over them. 0 if unknown."""
    ls, rs, nose = _pt(kp, conf, L_SH), _pt(kp, conf, R_SH), _pt(kp, conf, NOSE)
    sw = _shoulder_width(kp, conf)
    if sw is None or nose is None:
        return 0
    tilt = abs(ls[1] - rs[1]) / sw
    off = abs(nose[0] - (ls[0] + rs[0]) / 2) / sw
    return int(round(100 - 50 * min(tilt / 0.5, 1) - 50 * min(off / 0.5, 1)))


def motion_energy(prev_kp, prev_conf, kp, conf):
    """Mean keypoint displacement between frames, in shoulder widths. 0 if unknown."""
    sw = _shoulder_width(kp, conf)
    if sw is None or prev_kp is None:
        return 0.0
    d = []
    for i in range(min(len(kp), 17)):
        a, b = _pt(prev_kp, prev_conf, i), _pt(kp, conf, i)
        if a is not None and b is not None:
            d.append(np.linalg.norm(a - b))
    return float(np.mean(d) / sw) if d else 0.0


def is_hands_up(kp, conf):
    nose, lw, rw = _pt(kp, conf, NOSE), _pt(kp, conf, L_WR), _pt(kp, conf, R_WR)
    if nose is None or lw is None or rw is None:
        return False
    return bool(lw[1] < nose[1] and rw[1] < nose[1])  # image y grows downward


def is_t_pose(kp, conf):
    ls, rs, lw, rw = (_pt(kp, conf, i) for i in (L_SH, R_SH, L_WR, R_WR))
    sw = _shoulder_width(kp, conf)
    if sw is None or lw is None or rw is None:
        return False
    line_y = (ls[1] + rs[1]) / 2
    level = abs(lw[1] - line_y) <= 0.25 * sw and abs(rw[1] - line_y) <= 0.25 * sw
    return bool(level and abs(lw[0] - rw[0]) > 2.2 * sw)


class GestureTrigger:
    """Fires once after the condition is held hold_s, then ignores input for cooldown_s."""

    def __init__(self, hold_s=0.5, cooldown_s=1.5, clock=time.monotonic):
        self.hold_s, self.cooldown_s, self.clock = hold_s, cooldown_s, clock
        self._since = None
        self._blocked_until = -float("inf")

    def update(self, active):
        now = self.clock()
        if now < self._blocked_until:
            self._since = None
            return False
        if not active:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if now - self._since >= self.hold_s:
            self._since = None
            self._blocked_until = now + self.cooldown_s
            return True
        return False
