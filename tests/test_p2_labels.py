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
SMALL = {"model_min_elements": 800, "model_target_elements": 1500, "model_max_elements": 3000, "max_ndof": 200000}


def _brep(path):
    from netgen.occ import Axes, Cylinder, Z
    cyl = Cylinder(Axes((0, 0, -L / 2), Z), r=R, h=L)
    cyl.faces.name = "wall"
    cyl.WriteBrep(str(path))
    return str(path)


def test_model_sized_mesh_lands_in_range_on_a_pillbox(tmp_path):
    mesh, _, info = fl.model_sized_mesh(_brep(tmp_path / "p.brep"), 0.02, None, SMALL)
    assert SMALL["model_min_elements"] <= mesh.ne <= SMALL["model_max_elements"]
    assert info["n_tets"] == mesh.ne and info["curvaturesafety"] == 2.0 and info["n_meshes"] >= 1
    assert mesh.levels == 1


def test_model_sized_mesh_on_a_small_feature_shape(tmp_path):
    """TESLA cell with pipes (iris radii, small curvature radii): inside the range, single level."""
    from src.data_gen import cavity_shapes as cs
    from tests import p2_reference as p2
    tesla = dict(Req=0.1033, Riris=0.035, L=0.0577, A=0.042, B=0.042, a=0.012, b=0.019, n_cells=1.0, Lpipe=0.08)
    brep = p2.write_brep(lambda occ: cs.build_elliptical(occ, None, dict(tesla)), tmp_path / "tesla.brep")
    s = {"model_min_elements": 1500, "model_target_elements": 3000, "model_max_elements": 6000}
    mesh, _, info = fl.model_sized_mesh(str(brep), 0.03, None, s)
    assert 1500 <= mesh.ne <= 6000 and mesh.levels == 1 and info["curvaturesafety"] in (2.0, 1.5, 1.0)


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


def test_model_sized_mesh_falls_back_to_a_coarser_curvature_rule(monkeypatch):
    calls = []
    monkeypatch.setattr(fl, "curved_mesh", _fake_curved_mesh({2.0: 30000, 1.5: 12000, 1.0: 60000}, calls))
    monkeypatch.setattr(fl.ho, "single_level", lambda m, *a, **k: m)
    mesh, _, info = fl.model_sized_mesh("x.brep", 0.1, None, {})
    assert info["curvaturesafety"] == 1.5 and mesh.ne == 12000 == info["n_tets"]   # cs 2: 30k whatever maxh
    assert {cs for _, cs in calls} == {2.0, 1.5}
    # no pointless retries once coarser maxh stops helping at a curvature rule
    assert len([c for c in calls if c[1] == 2.0]) <= 4
    # a mesh that fits at the first rule never touches the next one
    calls.clear()
    monkeypatch.setattr(fl, "curved_mesh", _fake_curved_mesh({2.0: 100, 1.5: 100, 1.0: 100}, calls))
    mesh, _, info = fl.model_sized_mesh("x.brep", 0.1, None, {})
    assert info["curvaturesafety"] == 2.0 and 5000 <= mesh.ne <= 20000 and {cs for _, cs in calls} == {2.0}


def test_model_sized_mesh_raises_when_nothing_fits(monkeypatch):
    calls = []
    monkeypatch.setattr(fl, "curved_mesh", _fake_curved_mesh({2.0: 90000, 1.5: 90000, 1.0: 90000}, calls))
    monkeypatch.setattr(fl.ho, "single_level", lambda m, *a, **k: m)
    with pytest.raises(RuntimeError, match="no curved mesh with 5000–20000 tets"):
        fl.model_sized_mesh("x.brep", 0.1, None, {})
    assert {cs for _, cs in calls} == {2.0, 1.5, 1.0}


@pytest.fixture(scope="module")
def p2_pillbox(tmp_path_factory):
    brep = _brep(tmp_path_factory.mktemp("p2") / "p.brep")
    return fl.label_sample(brep, 0.02, "pillbox", 3, settings=SMALL)


def test_p2_label_sample_attrs_and_projection(p2_pillbox):
    a = p2_pillbox["attrs"]
    assert a["label_order"] == a["model_order"] == 2 and "adapt_iters" not in a
    assert np.isfinite(a["est_rel_f"]) and a["est_rel_f"] > 0 and a["label_ok"] in (0, 1)
    assert a["label_ok"] == int(a["est_rel_f"] <= a["est_max"]) and a["est_max"] == 1e-3
    assert a["curvaturesafety_used"] == 2.0 and a["model_meshes"] >= 1 and a["maxh_factor_used"] > 0
    assert 800 <= a["n_tets"] <= 3000 and a["n_tets"] == a["n_tets_label"] and a["n_dof_label"] == a["n_dof_model"]
    # same space: the "projection" is the identity (nested, unit-norm shortcut); _check_reload ran inside label_sample
    assert np.abs(p2_pillbox["proj_err"]).max() < 1e-6


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
           "--model_min_elements", "300", "--model_target_elements", "600", "--model_max_elements", "1500",
           "--h5_filename", str(out / "d.h5")]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    files = [os.path.join(d, f) for d, _, fs in os.walk(out) for f in fs if f.endswith(".h5")]
    assert files, r.stdout[-1000:]
    with h5py.File(files[0], "r") as f:
        assert json.loads(f.attrs["metadata"])["generator_args"]["label_order"] == 2
        g = f["sample_0001"]
        assert g.attrs["label_order"] == g.attrs["model_order"] == 2 and "est_rel_f" in g.attrs
        assert 300 <= g.attrs["n_tets"] <= 1500


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
