import time

from cybervision.camera import Camera


class FakeCap:
    def __init__(self, start_after=0.0):
        self.n, self.released, self.t0, self.delay = 0, False, time.time(), start_after

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.005)
        if time.time() - self.t0 < self.delay:
            return False, None
        self.n += 1
        return True, self.n

    def release(self):
        self.released = True


def test_read_before_first_frame():
    cam = Camera(FakeCap(start_after=5))
    assert cam.read()[0] is False
    cam.stop()


def test_newest_frame_and_seq_advance():
    cam = Camera(FakeCap())
    time.sleep(0.1)
    _, f1, s1 = cam.read()
    time.sleep(0.1)
    _, f2, s2 = cam.read()
    cam.stop()
    assert s2 > s1 and f2 > f1


def test_stop_releases_and_ends_thread():
    cap = FakeCap()
    cam = Camera(cap)
    cam.stop()
    assert cap.released and not cam._thread.is_alive()
