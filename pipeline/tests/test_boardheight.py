import math

import numpy as np
import pytest

from board_pipeline import boardheight


def _board(tilt_x_deg=0.3, z=0.005):
    """A board pose tilted about base y (its top rises along base x)."""
    t = math.radians(tilt_x_deg)
    b = np.eye(4)
    b[:3, :3] = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
    b[:3, 3] = [0.5, 0.0, z]
    return b


def test_vision_height_and_slope():
    b = _board()
    s = boardheight.vision_slope(b)
    z0 = boardheight.vision_height(b, 0.0, (0.5, 0.0))
    z1 = boardheight.vision_height(b, 0.0, (0.51, 0.02))
    assert z0 == pytest.approx(0.005)
    assert z1 - z0 == pytest.approx(s[0] * 0.01 + s[1] * 0.02)
    assert s[0] == pytest.approx(math.tan(math.radians(0.3)))


def _touches(rng, noise=2e-5):
    """Runs of 3 touches, robot - vision = 1.5 mm + (4, 1) um/mm + a peg offset."""
    offsets = {"triangle": -0.0004, "l_shape": 0.0001, "rectangle": 0.0}
    out = []
    spots = [("triangle", 0.50, -0.02), ("triangle", 0.55, -0.02), ("rectangle", 0.55, -0.07), ("l_shape", 0.55, -0.13),
             ("l_shape", 0.50, -0.13), ("l_shape", 0.45, -0.13)]
    for k, (shape, x, y) in enumerate(spots):
        for dx, dy in ((0.0, 0.0), (-0.022, 0.0), (-0.022, -0.022)):
            xy = np.array([x + dx, y + dy])
            zv = 0.006
            dz = 0.0015 + 0.004 * (xy[0] - 0.5) + 0.001 * (xy[1] + 0.07) + offsets[shape] + rng.normal(0, noise)
            out.append({"run": f"r{k}", "shape": shape, "xy": xy.tolist(), "z": zv + dz, "z_vision": zv,
                        "board_sha256": "s"})
    return out


def test_fit_recovers_slope_and_offsets():
    t = _touches(np.random.default_rng(0))
    m = boardheight.fit(t + [{**t[0], "board_sha256": "other", "z": 1.0}], "s")
    assert m["touches"] == len(t) and m["runs"] == 6
    assert m["slope"][0] == pytest.approx(0.004, abs=3e-4)
    assert m["slope"][1] == pytest.approx(0.001, abs=3e-4)
    diff = m["shape_offset"]["triangle"] - m["shape_offset"]["l_shape"]
    assert diff == pytest.approx(-0.0005, abs=5e-5)
    assert boardheight.slope_known(m)
    z, known = boardheight.predict(m, (0.52, -0.05), "triangle", 0.006)
    true = 0.006 + 0.0015 + 0.004 * 0.02 + 0.001 * 0.02 - 0.0004
    assert known and z == pytest.approx(true, abs=8e-5)
    _, known = boardheight.predict(m, (0.52, -0.05), "hexagon", 0.006)
    assert not known


def test_slope_unknown_from_one_run():
    t = [x for x in _touches(np.random.default_rng(1)) if x["run"] == "r0"]
    assert not boardheight.slope_known(boardheight.fit(t, "s"))


def test_leave_one_run_out_with_one_touch_is_tight():
    rows = boardheight.leave_one_run_out(_touches(np.random.default_rng(2)), "s")
    assert len(rows) == 6
    assert max(r["rms_one_m"] for r in rows) < 1e-4


def test_record_and_load(tmp_path):
    boardheight.record(tmp_path, "run1", "round", "s", [(0.5, 0.1, 0.007)], [0.0055])
    boardheight.record(tmp_path, "run1", "round", "s", [(0.5, 0.1, 0.0071)], [0.0055])  # the run's own replaced
    boardheight.record(tmp_path, "run2", "round", "s", [(0.52, 0.1, 0.007)], [0.0055])
    t = boardheight.load(tmp_path)
    assert [x["run"] for x in t] == ["run1", "run2"] and t[0]["z"] == pytest.approx(0.0071)


def test_resolve_tip_prefers_flag_then_session_then_cad(tmp_path):
    pytest.importorskip("cv2")  # grasp -> panda needs OpenCV (the pipeline's interpreter has it)
    from board_pipeline import grasp
    tool = grasp.Tool(shape="l_shape", rotation=np.eye(3), tip_offset=np.array([0.0045, 0.0170, 0.039]),
                      point=np.zeros(2), peg_yaw=0.0, grasp_width_mm=19.4, turn=0.0, period=0.0) \
        if "turn" in grasp.Tool.__dataclass_fields__ else None
    if tool is None:
        pytest.skip("Tool fields changed")
    assert grasp.resolve_tip(tool, None, tmp_path)[1] == "CAD"
    tips = grasp.session_tip_path(tmp_path, "l_shape")
    tips.parent.mkdir()
    tips.write_text("frame: panda_hand_tcp\nshape: l_shape\ntip_in_tcp_m: [0.004732, 0.018022, 0.039]\n")
    t, source = grasp.resolve_tip(tool, None, tmp_path)
    assert t.tip_offset[1] == pytest.approx(0.018022) and "tips/l_shape.yaml" in source
    t, source = grasp.resolve_tip(tool, "cad", tmp_path)
    assert t.tip_offset[1] == pytest.approx(0.0170) and source.startswith("CAD")
    other = tmp_path / "x.yaml"
    other.write_text("frame: panda_hand_tcp\nshape: l_shape\ntip_in_tcp_m: [0.0045, 0.0175, 0.039]\n")
    assert grasp.resolve_tip(tool, str(other), tmp_path)[0].tip_offset[1] == pytest.approx(0.0175)
