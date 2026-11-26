import numpy as np
import pytest

from cybervision import effects as fx

SHAPE = (120, 160, 3)


def frame():
    return np.random.default_rng(0).integers(0, 255, SHAPE, dtype=np.uint8)


def person(mask="ok", box=(40, 20, 100, 110), pid=1):
    kp = np.zeros((17, 2))
    kp[5], kp[6], kp[0] = (60, 50), (90, 50), (75, 30)
    conf = np.zeros(17)
    conf[[0, 5, 6]] = 1
    m = None
    if mask == "ok":
        m = np.zeros(SHAPE[:2], bool)
        m[20:110, 40:100] = True
    elif mask == "small":
        m = np.ones((30, 40), bool)
    elif mask == "empty":
        m = np.zeros((0, 0), bool)
    return {"id": pid, "box": box, "kp": kp, "conf": conf, "mask": m, "carrying": ["cup"],
            "elbows": (90.0, None), "knees": (None, None), "posture": 80, "energy": 0.1}


CASES = {
    "none": [], "ok": [person()], "no_mask": [person(None)], "empty_mask": [person("empty")],
    "small_mask": [person("small")], "box_out": [person(box=(-50, -50, 400, 300))],
    "box_fully_out": [person(box=(500, 500, 600, 600))], "no_id": [{**person(), "id": None}],
}


def check(out, f):
    assert out.shape == f.shape and out.dtype == np.uint8


@pytest.mark.parametrize("case", CASES)
def test_all_modes(case):
    f, people = frame(), CASES[case]
    check(fx.draw_hud(f, people, "HUD", 30.0, ("office", 0.7)), f)
    check(fx.hologram(f, people), f)
    check(fx.BackgroundModel().ghost(f, people), f)
    echo = fx.TimeEcho(every=1)
    for _ in range(3):
        check(echo.apply(f, people), f)


def test_empty_frame_and_none():
    e = np.zeros((0, 0, 3), np.uint8)
    for fn in (lambda x: fx.draw_hud(x, []), lambda x: fx.hologram(x, []),
               lambda x: fx.BackgroundModel().ghost(x, []), lambda x: fx.TimeEcho().apply(x, [])):
        assert fn(e).shape == e.shape
        assert fn(None) is None


def test_no_people_returns_unchanged_content():
    f = frame()
    assert np.array_equal(fx.hologram(f, []), f)


def test_ghost_learns_background_and_fades_person():
    bg = fx.BackgroundModel(rate=1.0)
    empty = np.full(SHAPE, 50, np.uint8)
    bg.update(empty, [])
    f = np.full(SHAPE, 250, np.uint8)
    out = bg.ghost(f, [person()])
    # inside the person: 25% frame + 75% background (50) = 100
    assert 95 <= out[60, 70, 0] <= 105


def test_echo_does_not_mutate_input_and_resizes_mask_shape_change():
    f = frame()
    before = f.copy()
    echo = fx.TimeEcho(every=1)
    echo.apply(f, [person()])
    echo.apply(np.zeros((60, 80, 3), np.uint8), [person()])  # different size: old copies skipped
    assert np.array_equal(f, before)
