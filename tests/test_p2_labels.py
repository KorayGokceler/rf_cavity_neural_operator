"""p2 labels on a model-sized mesh (docs/37): model_sized_mesh, label_sample's p2 path, the generator's default, and
the p4 reference tool (src/data_gen/reference.py).

File name: runs after test_data_gen_3d.py (netgen state must not fork the generator's Pool)."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")
from scipy.special import jn_zeros  # noqa: E402

from src.data_gen import field_labels as fl  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C0 = 299792458.0
R, L = 0.040, 0.050
SMALL = {"model_min_elements": 800, "model_target_elements": 1500, "max_ndof": 200000, "tol_f_p2": 1.0}   # tol 1: no refinement


def _brep(path):
    from netgen.occ import Axes, Cylinder, Z
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    cyl.WriteBrep(str(path))
    return str(path)


def test_model_sized_mesh_lands_near_the_target_on_a_pillbox(tmp_path):
    mesh, _, info = fl.model_sized_mesh(_brep(tmp_path / "p.brep"), 0.02, None, SMALL)
    assert SMALL["model_min_elements"] <= mesh.ne <= 2 * SMALL["model_target_elements"]
    assert info["n_tets"] == mesh.ne and info["curvaturesafety"] == 2.0 and info["n_meshes"] >= 1
    assert mesh.levels == 1


def test_model_sized_mesh_on_a_small_feature_shape(tmp_path):
    """TESLA cell with pipes (iris radii, small curvature radii): at least the minimum, curvaturesafety 2, single level."""
    from src.data_gen import cavity_shapes as cs
    from tests import p2_reference as p2
    tesla = dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019, n_cells=1.0, Lpipe=0.08)
    brep = p2.write_brep(lambda occ: cs.build_elliptical(occ, None, dict(tesla)), tmp_path / "tesla.brep")
    s = {"model_min_elements": 1500, "model_target_elements": 3000}
    mesh, _, info = fl.model_sized_mesh(str(brep), 0.03, None, s)   # folds at cs 2: retried at cs 3
    assert mesh.ne >= 1500 and mesh.levels == 1 and info["curvaturesafety"] in (3.0, 4.0)


def _fake_curved_mesh(counts, calls):
    """curved_mesh stand-in: tets = max(C / f³, floor[cs])  (maxh-bound above the curvature floor)."""
    def fake(cad_path, h, deform, s):
        f, cs = s["maxh_factor"], s["curvaturesafety"]
        calls.append((round(f, 4), cs))
        ne = int(max(1.08e6 / f ** 3, counts[cs]))
        if ne > s["max_elements"]:
            from src.data_gen import highorder as ho
            raise ho.TooManyElements("too many")
        return SimpleNamespace(ne=ne, levels=1), "shape"
    return fake


def test_model_sized_mesh_has_no_cap_and_no_curvature_fallback(monkeypatch):
    """Curvature-bound shape (30k tets at curvaturesafety 2 whatever maxh): the 30k mesh is kept, cs is never lowered."""
    calls = []
    monkeypatch.setattr(fl, "curved_mesh", _fake_curved_mesh({2.0: 30000, 1.5: 12000, 1.0: 60000}, calls))
    monkeypatch.setattr(fl.ho, "single_level", lambda m, *a, **k: m)
    mesh, _, info = fl.model_sized_mesh("x.brep", 0.1, None, {})
    assert mesh.ne == 30000 == info["n_tets"] and info["curvaturesafety"] == 2.0
    assert {cs for _, cs in calls} == {2.0} and len(calls) <= 5      # no pointless retries once coarser maxh stops helping
    calls.clear()                                                    # an easy shape lands near the target
    monkeypatch.setattr(fl, "curved_mesh", _fake_curved_mesh({2.0: 100, 1.5: 100, 1.0: 100}, calls))
    mesh, _, info = fl.model_sized_mesh("x.brep", 0.1, None, {})
    assert info["curvaturesafety"] == 2.0 and 8000 <= mesh.ne <= 12500 and {cs for _, cs in calls} == {2.0}


def test_model_sized_mesh_raises_when_only_too_coarse_meshes_exist(monkeypatch):
    calls = []
    monkeypatch.setattr(fl, "curved_mesh", _fake_curved_mesh({2.0: 100, 1.5: 100, 1.0: 100}, calls))
    monkeypatch.setattr(fl.ho, "single_level", lambda m, *a, **k: m)
    with pytest.raises(RuntimeError, match="no curved mesh with >= 5000 tets"):
        fl.model_sized_mesh("x.brep", 0.1, None, {"model_min_elements": 5000, "model_target_elements": 10000,
                                                  "maxh_factor": 1e3, "model_mesh_tries": 2})
    assert {cs for _, cs in calls} == {2.0}


def _fake_refine_env(monkeypatch, ests_by_ne, ndof_per_tet=11):
    """refine_p2_to_tolerance with a fake mesh builder: tets = 1e7 / f³ (so f x 0.9 → 1.37x tets), estimate by tets."""
    built = []

    def single(cad, h, deform, s, f, cs, cap):
        built.append((f, cs))
        return SimpleNamespace(ne=int(1.0e7 / f ** 3), levels=1), "shape"
    monkeypatch.setattr(fl, "_single_level_mesh", single)
    monkeypatch.setattr("ngsolve.HCurl", lambda m, **k: SimpleNamespace(ndof=ndof_per_tet * m.ne))
    return built


def test_refine_p2_to_tolerance_refines_until_the_estimate_is_within_tol(monkeypatch):
    built = _fake_refine_env(monkeypatch, None)
    est_of = lambda m: (4e-3 * (10000 / m.ne) ** (4 / 3), m.ne)        # error ∝ h⁴ ∝ tets^(-4/3)  # noqa: E731
    s = {**fl.DEFAULTS, "tol_f_p2": 1e-3}
    start = SimpleNamespace(ne=10000, levels=1)
    r = fl.refine_p2_to_tolerance("x", 0.1, None, s, start, "shape", 10.0, est_of)
    assert r["stop"] == "converged" and r["est"] <= 1e-3 and r["rounds"] == len(r["tets"]) == len(built) + 1
    assert r["rounds"] == 2 and r["tets"][1] > r["tets"][0] and r["payload"] == r["tets"][-1]
    # (1) both sizes scale by r = (tol/est)^(1/4)·0.9: maxh_factor down, curvaturesafety up by the same factor
    rr = (1e-3 / 4e-3) ** 0.25 * 0.9
    (f1, cs1), = built
    assert f1 == pytest.approx(10.0 * rr) and cs1 == pytest.approx(2.0 / rr) and cs1 > 2.0
    assert r["css"] == pytest.approx([2.0, cs1]) and r["cs"] == pytest.approx(cs1) and r["f"] == pytest.approx(f1)
    # already within tol: one solve, no mesh built
    built.clear()
    r = fl.refine_p2_to_tolerance("x", 0.1, None, s, start, "shape", 10.0, lambda m: (5e-4, None))
    assert r["rounds"] == 1 and r["stop"] == "converged" and not built


def test_refine_p2_to_tolerance_stalls_when_the_mesh_barely_grows(monkeypatch):
    """Curvature-bound shape: refinement changes the tets by < 15 %: stop 'stalled', keep the last solved mesh, never 'converged'."""
    def single(cad, h, deform, s, f, cs, cap):
        return SimpleNamespace(ne=10500, levels=1), "shape"
    monkeypatch.setattr(fl, "_single_level_mesh", single)
    monkeypatch.setattr("ngsolve.HCurl", lambda m, **k: SimpleNamespace(ndof=11 * m.ne))
    start = SimpleNamespace(ne=10000, levels=1)
    r = fl.refine_p2_to_tolerance("x", 0.1, None, {**fl.DEFAULTS}, start, "shape", 10.0, lambda m: (4e-3, None))
    assert r["stop"] == "stalled" and r["rounds"] == 1 and r["mesh"] is start and r["est"] > 1e-3


def test_refine_p2_to_tolerance_retries_a_stalled_mesh_with_a_larger_step(monkeypatch):
    """r (cs 2 → 2.3) grows 5 %, r² (cs 2.6) 10 %, r³ (cs 2.9) 40 %: the third meshing is solved (overshoot is fine); cs is capped."""
    calls = []

    def single(cad, h, deform, s, f, cs, cap):
        calls.append(cs)
        return SimpleNamespace(ne=10500 if cs < 2.5 else 11000 if cs < 2.8 else 14000, levels=1), "shape"
    monkeypatch.setattr(fl, "_single_level_mesh", single)
    monkeypatch.setattr("ngsolve.HCurl", lambda m, **k: SimpleNamespace(ndof=11 * m.ne))
    start = SimpleNamespace(ne=10000, levels=1)
    ests = iter([1.15e-3, 4e-4])
    r = fl.refine_p2_to_tolerance("x", 0.1, None, {**fl.DEFAULTS}, start, "shape", 10.0, lambda m: (next(ests), None))
    assert len(calls) == 3 and calls[0] < calls[1] < calls[2] and r["stop"] == "converged" and r["tets"] == [10000, 14000]
    assert r["css"] == pytest.approx([2.0, calls[2]])
    calls.clear()                                       # cs cap: est 4e-3 → r = 0.64, cs 3.1, 4.9, then 7.7 > 6: two meshings
    monkeypatch.setattr(fl, "_single_level_mesh", lambda *a: (calls.append(a[5]), (SimpleNamespace(ne=10500), "s"))[1])
    r = fl.refine_p2_to_tolerance("x", 0.1, None, {**fl.DEFAULTS, "p2_max_cs": 6.0, "max_ndof": 10**9}, start, "shape",
                                  10.0, lambda m: (4e-3, None))
    assert len(calls) == 2 and max(calls) <= 6.0 and r["stop"] == "stalled"


def test_refine_p2_to_tolerance_retries_a_folded_mesh_at_a_finer_cs(monkeypatch):
    calls = []

    def single(cad, h, deform, s, f, cs, cap):
        calls.append(cs)
        if len(calls) < 3:
            raise RuntimeError("folded elements")
        return SimpleNamespace(ne=20000, levels=1), "shape"
    monkeypatch.setattr(fl, "_single_level_mesh", single)
    monkeypatch.setattr("ngsolve.HCurl", lambda m, **k: SimpleNamespace(ndof=11 * m.ne))
    start = SimpleNamespace(ne=10000, levels=1)
    ests = iter([4e-3, 5e-4])
    r = fl.refine_p2_to_tolerance("x", 0.1, None, {**fl.DEFAULTS}, start, "shape", 10.0, lambda m: (next(ests), None))
    assert calls[1] == calls[0] + 1 and calls[2] == calls[0] + 2 and r["stop"] == "converged" and r["css"][-1] == calls[2]
    calls.clear()
    monkeypatch.setattr(fl, "_single_level_mesh", lambda *a: (_ for _ in ()).throw(RuntimeError("folded")))
    r = fl.refine_p2_to_tolerance("x", 0.1, None, {**fl.DEFAULTS}, start, "shape", 10.0, lambda m: (4e-3, None))
    assert r["stop"] == "mesh_failed" and r["mesh"] is start


def test_model_sized_mesh_raises_cs_on_a_fold_and_never_lowers_it(monkeypatch):
    calls = []

    def fake(cad_path, h, deform, s):
        calls.append(s["curvaturesafety"])
        if s["curvaturesafety"] < 3.0:
            raise RuntimeError("folded elements")
        return SimpleNamespace(ne=int(1.08e6 / s["maxh_factor"] ** 3), levels=1), "shape"
    monkeypatch.setattr(fl, "curved_mesh", fake)
    monkeypatch.setattr(fl.ho, "single_level", lambda m, *a, **k: m)
    mesh, _, info = fl.model_sized_mesh("x.brep", 0.1, None, {})
    assert calls[:2] == [2.0, 3.0] and min(calls) == 2.0 and info["curvaturesafety"] == 3.0 and mesh.ne >= 5000


def test_refine_p2_to_tolerance_budget_and_max_rounds(monkeypatch):
    _fake_refine_env(monkeypatch, None)
    start = SimpleNamespace(ne=10000, levels=1)
    s = {**fl.DEFAULTS, "tol_f_p2": 1e-3, "max_ndof": 150000}           # 10k tets x 11 = 110k; ~1.4x tets exceeds 150k
    r = fl.refine_p2_to_tolerance("x", 0.1, None, s, start, "shape", 10.0, lambda m: (4e-3, None))
    assert r["stop"] == "budget" and r["rounds"] == 1 and r["est"] == 4e-3 and r["mesh"] is start
    s = {**fl.DEFAULTS, "tol_f_p2": 1e-3, "max_ndof": 10**9, "p2_max_refine": 2}   # estimate never improves
    r = fl.refine_p2_to_tolerance("x", 0.1, None, s, start, "shape", 10.0, lambda m: (4e-3, None))
    assert r["stop"] == "max_rounds" and r["rounds"] == 3


@pytest.fixture(scope="module")
def p2_pillbox(tmp_path_factory):
    brep = _brep(tmp_path_factory.mktemp("p2") / "p.brep")
    return fl.label_sample(brep, 0.02, "pillbox", 3, settings=SMALL)


def test_p2_label_sample_attrs_and_projection(p2_pillbox):
    a = p2_pillbox["attrs"]
    assert a["label_order"] == a["model_order"] == 2 and "adapt_iters" not in a
    assert np.isfinite(a["est_rel_f"]) and a["est_rel_f"] > 0 and a["label_ok"] in (0, 1)
    assert a["label_ok"] == int(a["est_rel_f"] <= a["tol_f_p2"]) == 1 and a["tol_f_p2"] == 1.0
    assert a["curvaturesafety_used"] == 2.0 and a["model_meshes"] >= 1 and a["maxh_factor_used"] > 0   # no cs fallback
    assert a["p2_rounds"] == 1 and a["p2_stop"] == "converged"
    assert 800 <= a["n_tets"] <= 3000 and a["n_tets"] == a["n_tets_label"] and a["n_dof_label"] == a["n_dof_model"]
    # same space: the "projection" is the identity (nested, unit-norm shortcut); _check_reload ran inside label_sample
    assert np.abs(p2_pillbox["proj_err"]).max() < 1e-6


def test_p2_label_sample_refines_until_the_estimate_is_within_tol(tmp_path):
    """Pillbox, h = 0.02: start mesh ~1.8k tets has est 1.6e-4; tol 5e-5 forces refinement (numbers in the assert message)."""
    brep = _brep(tmp_path / "p.brep")
    res = fl.label_sample(brep, 0.02, "pillbox", 3, settings={**SMALL, "tol_f_p2": 5e-5})
    a = res["attrs"]
    msg = (a["p2_tets"], a["p2_est"], a["p2_stop"])
    assert a["p2_rounds"] >= 2 and a["p2_stop"] == "converged" and a["label_ok"] == 1 and a["est_rel_f"] <= 5e-5, msg
    assert a["p2_est"][0] > 5e-5 and a["p2_tets"][-1] > a["p2_tets"][0] and a["n_tets"] == a["p2_tets"][-1], msg
    assert a["curvaturesafety_used"] > 2.0 and a["p2_cs"][0] == 2.0 and a["p2_cs"][-1] == a["curvaturesafety_used"] and a["maxh_factor_used"] < 3.0 and np.abs(res["proj_err"]).max() < 1e-6


def test_p2_label_sample_budget_stop_keeps_the_last_mesh_and_flags_it(tmp_path):
    brep = _brep(tmp_path / "p.brep")
    first = fl.label_sample(brep, 0.02, "pillbox", 3, settings=SMALL)["attrs"]
    res = fl.label_sample(brep, 0.02, "pillbox", 3, settings={**SMALL, "tol_f_p2": 5e-5,
                                                               "max_ndof": int(first["n_dof_model"] * 1.1)})
    a = res["attrs"]
    assert a["p2_stop"] == "budget" and a["p2_rounds"] == 1 and a["label_ok"] == 0 and a["est_rel_f"] > 5e-5
    assert a["n_tets"] == first["n_tets"] and a["n_dof_model"] == first["n_dof_model"]


def test_p2_pillbox_frequency_within_the_measured_p2_error(p2_pillbox):
    """TM010 = c·j01/(2πR).  Measured on this 1777-tet mesh: rel. error 3.9e-5 (est_rel_f 1.6e-4, proj_err 3e-8); tolerance 5x."""
    f0 = C0 * jn_zeros(0, 1)[0] / (2 * np.pi * R) / 1e9
    err = abs(p2_pillbox["freqs"][0] / f0 - 1)
    assert err < TOL_TM010, err
    # the estimate is the right order of magnitude (docs/37: estimate / true error 0.8–3)
    est = p2_pillbox["attrs"]["est_rel_f"]
    assert 0.1 < est / max(err, 1e-12) < 30


TOL_TM010 = 2e-4


def test_generator_cli_default_writes_p2_labels(tmp_path):
    import h5py
    out = tmp_path / "d"
    out.mkdir()
    cmd = [sys.executable, os.path.join(REPO, "src/data_gen/dataset_generator_3d.py"), "--mode", "calibration",
           "--n_total", "2", "--n_eigen_modes", "2", "--mesh_size", "0.2", "--n_workers", "1", "--threads", "2",
           "--model_min_elements", "300", "--model_target_elements", "600", "--tol_f_p2", "1",
           "--h5_filename", str(out / "d.h5")]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    files = [os.path.join(d, f) for d, _, fs in os.walk(out) for f in fs if f.endswith(".h5")]
    assert files, r.stdout[-1000:]
    with h5py.File(files[0], "r") as f:
        assert json.loads(f.attrs["metadata"])["generator_args"]["label_order"] == 2
        g = f["sample_0001"]
        assert g.attrs["label_order"] == g.attrs["model_order"] == 2 and "est_rel_f" in g.attrs
        assert 300 <= g.attrs["n_tets"] and g.attrs["p2_stop"] == "converged"


def test_reference_check_on_a_p2_label_file(p2_pillbox, tmp_path):
    """p4 reference on the stored mesh (tiny); p2 labels are within 1e-3 of it, the cross-check (p3 vs p4) bounds p4 itself."""
    import h5py
    from src.data_gen import reference as ref
    path = tmp_path / "ref.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("sample_0001")
        fl.write_field_group(g, p2_pillbox)
        g.attrs["shape_type"] = "pillbox"
    out = tmp_path / "ref.json"
    doc = ref.run([str(tmp_path)], out=str(out), order=4, cross_check=True, log=lambda m: None)
    s = doc["samples"][0]
    assert json.load(open(out))["summary"] == doc["summary"] and s["skipped"] is None
    assert s["family"] == "pillbox" and s["ref_order"] == 4 and len(s["rel_err"]) == 3
    assert s["max_rel_err"] < 1e-3 and s["ref_self_err"] < s["max_rel_err"] * 3 + 1e-4
    assert doc["summary"]["max_rel_err_by_family"]["pillbox"] == s["max_rel_err"]
    big = ref.run([str(path)], order=3, max_ndof=100, log=lambda m: None)
    assert "max_ndof" in big["samples"][0]["skipped"] and big["summary"]["n_skipped"] == 1


def _write_h5(path, res, keys=("sample_0001",), **attrs):
    import h5py
    with h5py.File(path, "w") as f:
        for k in keys:
            g = f.create_group(k)
            fl.write_field_group(g, res)
            g.attrs["shape_type"] = "pillbox"
            for n, v in attrs.items():
                g.attrs[n] = v
    return str(path)


def test_qoi_ok_flags_a_bad_jacobian_and_keeps_the_label_ok(p2_pillbox, tmp_path, capsys):
    a = p2_pillbox["attrs"]
    assert a["qoi_ok"] == 1 and a["jacobian_ratio"] >= a["qoi_min_jacobian"] == 0.3
    brep = _brep(tmp_path / "p.brep")
    res = fl.label_sample(brep, 0.02, "pillbox", 2, settings={**SMALL, "qoi_min_jacobian": 2.0})   # ratio <= 1: flagged
    assert res["attrs"]["qoi_ok"] == 0 and res["attrs"]["label_ok"] == 1
    assert "qoi_ok = 0" in capsys.readouterr().out and np.isfinite(res["qoi"]).all()


def test_dataset_require_label_ok_drops_flagged_samples(p2_pillbox, tmp_path, capsys):
    from src.data.field_dataset import FieldDataset
    bad = _write_h5(tmp_path / "bad.h5", p2_pillbox, ("sample_0001", "sample_0002"), label_ok=0)
    good = _write_h5(tmp_path / "good.h5", p2_pillbox, ("sample_0003",), label_ok=1)
    legacy = _write_h5(tmp_path / "legacy.h5", p2_pillbox)                         # label_ok is in attrs: remove it
    import h5py
    with h5py.File(legacy, "r+") as f:
        for g in f.values():
            del g.attrs["label_ok"]
    kw = dict(train_ratio=1.0, val_ratio=0.0, with_ct=False)
    assert len(FieldDataset([bad, good, legacy], **kw)) == 4                       # default: nothing dropped
    capsys.readouterr()
    ds = FieldDataset([bad, good, legacy], require_label_ok=True, **kw)
    out = capsys.readouterr().out
    assert len(ds) == 2 and "dropped 2 samples" in out and "pillbox: 2" in out
    with pytest.raises(ValueError, match="all dropped"):
        FieldDataset(bad, require_label_ok=True, **kw)


def test_reference_skips_labels_from_another_mesh_and_counts_families_over_files(p2_pillbox, tmp_path):
    from src.data_gen import reference as ref
    a = _write_h5(tmp_path / "a.h5", p2_pillbox, n_tets_label=99999)             # label mesh != model mesh
    b = _write_h5(tmp_path / "b.h5", p2_pillbox, adapt_iters=3)
    doc = ref.run([a, b], order=2, log=lambda m: None)
    assert len(doc["samples"]) == 1                                              # one pillbox over both shards
    assert doc["samples"][0]["skipped"] == "label frequencies from a different (finer) label mesh"
    doc = ref.run([a, b], order=2, per_family=2, allow_other_mesh=True, log=lambda m: None)
    s = doc["samples"]
    assert len(s) == 2 and all(not x["skipped"] and x["est_over_true"] is None for x in s)


def test_reference_self_error_is_null_when_order_minus_one_is_the_label_order(p2_pillbox, tmp_path):
    from src.data_gen import reference as ref
    p = _write_h5(tmp_path / "c.h5", p2_pillbox)
    s = ref.run([p], order=3, cross_check=True, log=lambda m: None)["samples"][0]    # label order 2 == 3 - 1
    assert s["ref_self_err"] is None and "label order" in s["ref_self_err_note"] and s["max_rel_err"] < 1e-3
