import threading
import time

import cv2


class Camera:
    """Reads the camera in a background thread and keeps only the newest frame."""

    def __init__(self, source):
        # source: a camera index, or an already-open capture object (used by tests)
        self.cap = cv2.VideoCapture(source) if isinstance(source, int) else source
        self._lock = threading.Lock()
        self._frame, self._seq = None, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def isOpened(self):
        return self.cap.isOpened()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def read(self):
        """Returns (ok, newest frame, frame counter) without blocking."""
        with self._lock:
            return self._frame is not None, self._frame, self._seq

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()
