import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from . import gestures as g

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WRIST_IDS = (g.L_WR, g.R_WR)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Engine:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        def load(name):
            p = MODELS_DIR / f"{name}.pt"
            return YOLO(str(p) if p.exists() else f"{name}.pt")  # plain name auto-downloads

        self.models = {"pose": load("yolov8n-pose"), "seg": load("yolov8n-seg"),
                       "detect": load("yolov8n"), "cls": load("yolov8n-cls")}
        self._n = 0
        self._objects, self._env, self._cls_t = [], None, -float("inf")
        self._ms = {k: 0.0 for k in self.models}
        self._prev = {}

    def _timed(self, key, fn):
        t = time.perf_counter()
        out = fn()
        self._ms[key] = (time.perf_counter() - t) * 1000
        return out

    def _pose(self, frame, conf):
        r = self.models["pose"].track(frame, persist=True, classes=[0], conf=conf,
                                      device=self.device, imgsz=480, verbose=False)[0]
        people = []
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return people
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else range(1, len(boxes) + 1)
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        for i, tid in enumerate(ids):
            people.append({"id": int(tid), "box": boxes[i], "kp": kxy[i], "conf": kc[i], "mask": None})
        return people

    def _seg(self, frame, conf):
        r = self.models["seg"](frame, classes=[0], conf=conf, device=self.device, imgsz=480, verbose=False)[0]
        out = []
        if r.masks is None:
            return out
        h, w = frame.shape[:2]
        for poly, box in zip(r.masks.xy, r.boxes.xyxy.cpu().numpy()):
            m = np.zeros((h, w), np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            out.append((box, m.astype(bool)))
        return out

    def _detect(self, frame, conf):
        r = self.models["detect"](frame, conf=conf, device=self.device, verbose=False)[0]
        names = r.names
        return [{"name": names[int(c)], "box": b, "conf": float(s)}
                for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())
                if int(c) != 0]

    def _classify(self, frame):
        r = self.models["cls"](frame, device=self.device, verbose=False)[0]
        return r.names[r.probs.top1].replace("_", " "), float(r.probs.top1conf)

    def process(self, frame, settings):
        """settings: conf, pose, seg, detect, cls. Returns dict with people, objects, env, ms."""
        conf = settings.get("conf", 0.4)
        self._n += 1

        people = self._timed("pose", lambda: self._pose(frame, conf)) if settings.get("pose") else []
        segs = self._timed("seg", lambda: self._seg(frame, conf)) if settings.get("seg") else []

        if people:
            for p in people:
                best = max(segs, key=lambda s: _iou(p["box"], s[0]), default=None)
                if best is not None and _iou(p["box"], best[0]) > 0.3:
                    p["mask"] = best[1]
        else:  # pose off: still give mask-driven effects something to work with
            people = [{"id": i + 1, "box": b, "kp": np.zeros((17, 2)), "conf": np.zeros(17), "mask": m}
                      for i, (b, m) in enumerate(segs)]

        if settings.get("detect"):
            if self._n % 2 == 1:
                self._objects = self._timed("detect", lambda: self._detect(frame, conf))
        else:
            self._objects = []
        if settings.get("cls"):
            if time.monotonic() - self._cls_t >= 2:
                self._env = self._timed("cls", lambda: self._classify(frame))
                self._cls_t = time.monotonic()
        else:
            self._env, self._cls_t = None, -float("inf")

        prev, self._prev = self._prev, {}
        for p in people:
            kp, c = p["kp"], p["conf"]
            p["elbows"], p["knees"] = g.elbow_angles(kp, c), g.knee_angles(kp, c)
            p["posture"] = g.posture_score(kp, c)
            old = prev.get(p["id"])
            p["energy"] = g.motion_energy(old[0], old[1], kp, c) if old else 0.0
            self._prev[p["id"]] = (kp, c)
            p["carrying"] = self._carrying(p)

        ms = {k: self._ms[k] if settings.get(k) else 0.0 for k in self._ms}
        return {"people": people, "objects": self._objects, "env": self._env, "ms": ms}

    def _carrying(self, p):
        x1, y1, x2, y2 = p["box"]
        reach = 0.6 * (y2 - y1)
        wrists = [pt for pt in (g._pt(p["kp"], p["conf"], i) for i in WRIST_IDS) if pt is not None]
        names = []
        person_area = (x2 - x1) * (y2 - y1)
        for o in self._objects:
            ox1, oy1, ox2, oy2 = o["box"]
            if (ox2 - ox1) * (oy2 - oy1) > person_area:  # can't hold something bigger than yourself (bus, couch)
                continue
            cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
            if wrists:
                near = any(np.hypot(cx - w[0], cy - w[1]) < reach for w in wrists)
            else:
                near = x1 <= cx <= x2 and y1 <= cy <= y2
            if near and o["name"] not in names:
                names.append(o["name"])
        return names
