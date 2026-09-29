"""src/viz/viewer.py: CST-style viewer (phase / cut slices on a synthetic box mode)."""
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("skfem")

from src.viz.viewer import ModeViewer  # noqa: E402
from tests.maxwell3d_synth import box_geometry  # noqa: E402


@pytest.fixture(scope="module")
def viewer():
    geom, lam, Y = box_geometry(dims=(1.0, 0.8, 0.6), n=4, jitter=0.1, n_modes=3, seed=1)
    K = Y.shape[1]
    pred = {"geom_id": 0, "shape_type": "box", "X": geom["X"], "tets": geom["tets"], "edges": geom["edges"],
            "scale": 0.1, "center": np.zeros(3), "true": Y, "pred": 0.9 * Y,
            "f_true": np.ones(K), "f_pred": np.ones(K), "rel_l2": np.array([0.1, np.nan, 0.2]),
            "split": np.array([False, True, False])}
    return ModeViewer(pred, res=31)


def test_phase_standing_wave(viewer):
    *_, t0, p0, e0, _, _ = viewer.slice(0, "H", "x", "y", phase=0)
    *_, t90, _, _, _, _ = viewer.slice(0, "H", "x", "y", phase=90)
    *_, t180, _, _, _, _ = viewer.slice(0, "H", "x", "y", phase=180)
    m = np.isfinite(t0)
    assert m.mean() > 0.8
    np.testing.assert_allclose(t90[m], 0, atol=1e-12)            # H zero at 90°
    np.testing.assert_allclose(t180[m], -t0[m], atol=1e-12)      # sign flip at 180°
    np.testing.assert_allclose(p0[m], 0.9 * t0[m], atol=1e-12)
    *_, E0, _, _, _, _ = viewer.slice(0, "E", "abs", "y", phase=0)
    np.testing.assert_allclose(E0[np.isfinite(E0)], 0, atol=1e-12)  # E zero at 0°


def test_cut_position_and_limits(viewer):
    lo, hi = viewer.range_mm("z")
    U, V, t, *_ = viewer.slice(1, "H", "abs", "z", pos_mm=0.5 * (lo + hi))
    assert U.shape == t.shape == (31, 31)
    assert np.nanmax(t) <= viewer.limit(1, "H", "abs") + 1e-12
    _, _, t_out, *_ = viewer.slice(1, "H", "abs", "z", pos_mm=hi + 10.0)   # beyond the mesh
    assert np.isnan(t_out).all()


def test_draw_and_widget(viewer, tmp_path):
    fig = viewer.draw(1, "E", "z", "x", None, 45)
    assert "split pair" in fig._suptitle.get_text()
    fig.savefig(tmp_path / "v.png")
    pytest.importorskip("ipywidgets")
    viewer.widget()


@pytest.fixture(scope="module")
def viewer_e():
    geom, lam, Y = box_geometry(dims=(1.0, 0.8, 0.6), n=4, jitter=0.1, n_modes=2, seed=1, field="E")
    K = Y.shape[1]
    pred = {"field": "E", "geom_id": 0, "shape_type": "box", "X": geom["X"], "tets": geom["tets"],
            "edges": geom["edges"], "scale": 0.1, "center": np.zeros(3), "true": Y, "pred": 0.9 * Y,
            "f_true": np.ones(K), "f_pred": np.ones(K), "rel_l2": np.array([0.1, 0.2]),
            "split": np.array([False, False])}
    return ModeViewer(pred, res=31)


def test_e_primary_phase_and_curl(viewer_e):
    """E primary: E(t) = E cos φ (0° = E max), H(t) ∝ −curl E sin φ (0 at 0°, max at 90°)."""
    from src.viz.nedelec import vertex_field
    v = viewer_e
    assert v.fields == ("E", "H")
    *_, e0, _, _, _, _ = v.slice(0, None, "x", "y", phase=0)        # default field = primary E
    *_, e90, _, _, _, _ = v.slice(0, "E", "x", "y", phase=90)
    *_, h0, _, _, _, _ = v.slice(0, "H", "abs", "y", phase=0)
    *_, h90, _, _, _, _ = v.slice(0, "H", "x", "y", phase=90)
    m = np.isfinite(e0)
    assert np.abs(e0[m]).max() > 0
    np.testing.assert_allclose(e90[m], 0, atol=1e-12)
    np.testing.assert_allclose(h0[np.isfinite(h0)], 0, atol=1e-12)
    curl = vertex_field(v.X, v.tets, v.p["edges"], v.p["true"], curl=True)
    np.testing.assert_allclose(v.nodal[("H", "true")], curl)
    assert np.nanmax(np.abs(h90)) > 0
    fig = v.draw(0, None, "abs", "y", None, 90)
    assert fig._suptitle is not None
    with pytest.raises(ValueError):
        v.slice(0, "B")
