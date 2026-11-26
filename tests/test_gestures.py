import numpy as np

from cybervision import gestures as g


def person(**over):
    """Standing person facing camera; shoulder width 100, image y grows downward."""
    kp = np.zeros((17, 2))
    kp[g.NOSE] = (200, 80)
    kp[g.L_SH], kp[g.R_SH] = (150, 150), (250, 150)
    kp[g.L_EL], kp[g.R_EL] = (140, 220), (260, 220)
    kp[g.L_WR], kp[g.R_WR] = (140, 290), (260, 290)
    kp[g.L_HIP], kp[g.R_HIP] = (160, 300), (240, 300)
    kp[g.L_KN], kp[g.R_KN] = (160, 400), (240, 400)
    kp[g.L_AN], kp[g.R_AN] = (160, 500), (240, 500)
    for k, v in over.items():
        kp[getattr(g, k)] = v
    return kp, np.ones(17)


def test_angle():
    assert abs(g.angle((1, 0), (0, 0), (0, 1)) - 90) < 1e-6
    assert abs(g.angle((-1, 0), (0, 0), (1, 0)) - 180) < 1e-6
    assert g.angle(None, (0, 0), (1, 1)) is None
    assert g.angle((0, 0), (0, 0), (1, 1)) is None


def test_elbow_knee_straight():
    kp, c = person(L_EL=(150, 220), L_WR=(150, 290))
    assert abs(g.elbow_angles(kp, c)[0] - 180) < 1e-6
    assert g.knee_angles(kp, c)[0] is not None


def test_hands_up():
    kp, c = person(L_WR=(150, 20), R_WR=(250, 20))
    assert g.is_hands_up(kp, c)
    kp, c = person(L_WR=(150, 20))
    assert not g.is_hands_up(kp, c)


def test_t_pose():
    kp, c = person(L_WR=(20, 150), R_WR=(380, 150))
    assert g.is_t_pose(kp, c)
    assert not g.is_t_pose(*person())
    kp, c = person(L_WR=(20, 150), R_WR=(380, 300))  # one arm down
    assert not g.is_t_pose(kp, c)


def test_missing_keypoints():
    kp, c = person()
    c[:] = 0.1
    assert g.elbow_angles(kp, c) == (None, None)
    assert not g.is_hands_up(kp, c) and not g.is_t_pose(kp, c)
    assert g.posture_score(kp, c) == 0
    assert g.motion_energy(kp, c, kp, c) == 0.0


def test_zero_shoulder_width_and_empty():
    kp, c = person(R_SH=(150, 150), L_WR=(20, 150), R_WR=(380, 150))
    assert not g.is_t_pose(kp, c)
    assert g.posture_score(kp, c) == 0
    assert g.motion_energy(kp, c, kp, c) == 0.0
    for args in [(None, None), (np.zeros((0, 2)), np.zeros(0))]:
        assert not g.is_hands_up(*args) and not g.is_t_pose(*args)
        assert g.posture_score(*args) == 0
        assert g.elbow_angles(*args) == (None, None)


def test_posture_and_energy():
    assert g.posture_score(*person()) == 100
    assert g.posture_score(*person(L_SH=(150, 200))) < 100
    kp, c = person()
    moved = kp + 10
    assert abs(g.motion_energy(kp, c, moved, c) - np.hypot(10, 10) / 100) < 1e-6
    assert g.motion_energy(None, None, kp, c) == 0.0


def test_trigger_hold_and_cooldown():
    t = [0.0]
    trig = g.GestureTrigger(0.5, 1.5, clock=lambda: t[0])
    assert not trig.update(True)
    t[0] = 0.3
    assert not trig.update(True)
    t[0] = 0.6
    assert trig.update(True)
    t[0] = 1.0
    assert not trig.update(True)  # cooldown
    t[0] = 2.2
    assert not trig.update(True)  # hold restarts after cooldown
    t[0] = 2.8
    assert trig.update(True)


def test_trigger_released_resets_hold():
    t = [0.0]
    trig = g.GestureTrigger(0.5, 1.5, clock=lambda: t[0])
    trig.update(True)
    t[0] = 0.4
    trig.update(False)
    t[0] = 0.7
    assert not trig.update(True)
