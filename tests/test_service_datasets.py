"""Development-UI dataset browsing (src/service/datasets.py), model-vs-FE alignment (model.align)
and the dataset / feature / comparison endpoints of src/service/api.py."""
import base64
import pickle
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient                                   # noqa: E402

from src.data_gen import dataset_generator_3d as gen                         # noqa: E402
from src.service import api as A                                            # noqa: E402
from src.service.datasets import H5Dataset, discover                         # noqa: E402
from src.service.model import ModelService, align                           # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dsui")
    (d / "h5" / "pillbox_pipes").mkdir(parents=True)
    h5 = d / "h5" / "pillbox_pipes" / "pillbox_pipes_s00000.h5"
    gen.main(["--n_total", "3", "--n_workers", "1", "--mesh_size", "0.3", "--n_eigen_modes", "4",
              "--families", "pillbox_pipes", "--start_id", "1000", "--h5_filename", str(h5)])
    return d


def f32(s):
    return np.frombuffer(base64.b64decode(s), np.float32)


def test_discover_groups_h5_dirs(data_dir):
    ds = discover(str(data_dir))
    assert [(type(v).__name__, k) for k, v in ds.items()] == [("H5Dataset", "h5/pillbox_pipes/ (h5)")]


def test_h5_items(data_dir):
    h5 = H5Dataset([str(data_dir / "h5" / "pillbox_pipes" / "pillbox_pipes_s00000.h5")], "h")
    assert sorted(r["id"] for r in h5.rows()) == [1000, 1001, 1002]
    a = h5.open(1001)
    assert a["family"] == "pillbox_pipes" and a["U"].shape == (len(a["geom"]["edges"]), 4)
    assert np.all(np.diff(a["f_GHz"]) >= 0) and a["qoi_labels"] is None
    assert np.all(np.einsum("ik,ik->k", a["U"], a["M"] @ a["U"]) > 0)        # nonzero N0 modes
    assert "Req" not in a["params"] and "mesh_h" in a["params"]                # generator attributes


def test_align_recovers_sign_and_degenerate_rotation():
    rng = np.random.default_rng(0)
    n = 40
    A = rng.standard_normal((n, n))
    M = A @ A.T + n * np.eye(n)
    L = np.linalg.cholesky(M)                                                # M-orthonormal modes, like Ritz / FE
    T = np.linalg.solve(L.T, np.linalg.qr(rng.standard_normal((n, 4)))[0])
    f = np.array([1.0, 2.0, 2.0, 3.0])                                       # modes 1, 2 degenerate
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = T.copy()
    P[:, 0] *= -3.0
    P[:, 1:3] = T[:, 1:3] @ R
    P[:, 3] *= 0.5
    out, rl = align(P, T, M, f)
    np.testing.assert_allclose(rl, 0.0, atol=1e-10)
    np.testing.assert_allclose(out, T, atol=1e-10)


@pytest.fixture(scope="module")
def client(data_dir):
    return TestClient(A.create_app(ModelService(None, "cpu"), discover(str(data_dir))))


def _opened(client):
    did = next(d["id"] for d in client.get("/api/datasets").json() if d["kind"] == "h5")
    it = client.get(f"/api/datasets/{did}/items", params={"sort": "f0", "desc": True, "limit": 2}).json()
    return client.post(f"/api/datasets/{did}/items/{it['items'][0]['id']}/open").json()


def test_dataset_endpoints(client):
    L = client.get("/api/datasets").json()
    assert {d["kind"] for d in L} == {"h5"}
    did = L[0]["id"]
    it = client.get(f"/api/datasets/{did}/items", params={"sort": "f0", "desc": True, "limit": 2}).json()
    assert it["total"] == 3 and len(it["items"]) == 2
    assert it["items"][0]["f_GHz"][0] >= it["items"][1]["f_GHz"][0]
    st = client.get(f"/api/datasets/{did}/stats").json()
    assert st["families"]["pillbox_pipes"]["count"] == 3 and st["n_modes"] == 4
    assert client.get(f"/api/datasets/{did}/items", params={"sort": "nope"}).status_code == 422

    o = client.post(f"/api/datasets/{did}/items/{it['items'][0]['id']}/open").json()
    assert o["truth"]["source"] == "fe" and len(o["truth"]["modes"]) == 4
    assert o["truth"]["modes"][0]["Q0"] > 0                                     # figures of merit of the FE modes
    fe = client.get(f"/api/geometries/{o['id']}/features").json()
    assert fe["names"][3] == "dist_to_boundary" and fe["n_edges"] > 0
    fp = client.get(f"/api/geometries/{o['id']}/feature_plane", params={"name": "torsion", "res": 31}).json()
    v = f32(fp["values_b64"])
    assert v.size == fp["n_points"] > 0 and v.min() >= -1e-6 and v.max() <= 1 + 1e-6
    fs = client.get(f"/api/geometries/{o['id']}/feature_surface", params={"name": "dist_to_boundary"}).json()
    assert np.abs(f32(fs["values_b64"])).max() < 1e-6                          # zero on the wall
    assert client.get(f"/api/geometries/{o['id']}/feature_plane", params={"name": "x1"}).status_code == 404
    t = client.get(f"/api/predictions/{o['truth']['id']}/plane", params={"mode": 0, "res": 31}).json()
    assert t["n_points"] > 0


@pytest.mark.skip(reason="model prediction is not wired to the field3d model yet (the N0 eigenspace3d model "
                         "was removed)")
def test_model_vs_fe_comparison(client):
    o = _opened(client)
    p = client.post("/api/predictions", json={"geometry_id": o["id"], "compare_to": o["truth"]["id"]}).json()
    rows = p["comparison"]["rows"]
    assert len(rows) == 4 and all(r["rel_l2"] >= 0 for r in rows)
    assert all(r["f_fe"] == pytest.approx(o["truth"]["modes"][k]["f_GHz"]) for k, r in enumerate(rows))
    d = client.get(f"/api/predictions/{p['comparison']['diff_id']}/modes/0").json()
    assert d["surface"]["E_max"] > 0
    other = client.post("/api/geometries/sample", json={"family": "box", "id": 3, "mesh_size": 0.3}).json()
    bad = client.post("/api/predictions", json={"geometry_id": other["id"], "compare_to": o["truth"]["id"]})
    assert bad.status_code == 422


def test_feature_views(client):
    """Each feature group has its own display: frame grid, wall arrows, torsion iso-lines, cut tets."""
    g = client.post("/api/geometries/sample", json={"family": "pillbox_pipes", "id": 5, "mesh_size": 0.3}).json()
    gid = g["id"]

    def view(kind, **kw):
        r = client.get(f"/api/geometries/{gid}/feature_view", params={"kind": kind, "res": 61, **kw})
        assert r.status_code == 200, r.text
        return r.json()

    pos = view("position")
    roles = {s["role"]: s["n"] for s in pos["segments"]}
    assert pos["neutral"] and roles["grid"] > 0 and roles["axis"] > 0 and roles["marker"] == 2
    wall = view("wall")
    arrows = next(s for s in wall["segments"] if s["role"] == "arrow")
    A = f32(arrows["points_b64"]).reshape(-1, 2, 3)
    n = len(A) // 3                                                           # shafts first, then two heads
    shaft = A[:n]
    assert n > 10 and np.all(np.isfinite(shaft))
    bb = np.array(g["check"]["bbox_mm"])
    assert np.all(shaft[:, 1] >= bb[0] - 1) and np.all(shaft[:, 1] <= bb[1] + 1)   # tips stay in the bbox
    tor = view("torsion")
    assert tor["plane"]["n_points"] > 0 and tor["segments"][0]["role"] == "iso" and tor["range"] == [0.0, 1.0]
    mesh = view("mesh")
    assert mesh["plane"] is None and mesh["cells"]["n_points"] > 0
    assert "tets cut by the plane" in mesh["legend"]
    raw = view("raw", channel="dir_bnd_z")
    assert raw["signed"] and raw["plane"]["n_points"] > 0
    assert client.get(f"/api/geometries/{gid}/feature_view", params={"kind": "raw", "channel": "nope"}).status_code == 404
    assert client.get(f"/api/geometries/{gid}/feature_view", params={"kind": "bad"}).status_code == 422


@pytest.mark.skip(reason="the job's convert step runs convert_3d.py (PKL of the removed N0 model); the service "
                         "rewiring to field3d is pending")
def test_generation_job_writes_family_dataset(tmp_path, monkeypatch):
    """A generation job writes <root>/<TAG>/h5/<family>/<family>_s00000.h5 + pkl/<family>.pkl (ids from the
    family's TRUBA block), shows up in the dataset list, and re-submitting skips finished shards."""
    import time
    root = tmp_path / "drive"
    monkeypatch.setenv("RFCAV_GEN_ROOT", str(root))
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    cfg = c.get("/api/jobs/config").json()
    assert cfg["enabled"] and cfg["blocks"]["pillbox_pipes"] == 3
    body = {"families": ["pillbox_pipes"], "n_total": 3, "mesh_size": 0.3, "n_modes": 4, "shard_size": 16,
            "workers": 1, "deform_prob": 0.0, "tag": "t"}

    def run():
        j = c.post("/api/jobs/generate", json=body).json()[0]
        for _ in range(600):
            j = c.get(f"/api/jobs/{j['id']}").json()
            if j["status"] in ("done", "failed", "cancelled"):
                return j
            time.sleep(0.2)
        raise AssertionError("job did not finish")

    j = run()
    assert j["status"] == "done", j["log"][-10:]
    assert j["progress"]["done"] == 3
    h5 = root / "t" / "h5" / "pillbox_pipes" / "pillbox_pipes_s00000.h5"
    assert h5.exists() and (root / "t" / "pkl" / "pillbox_pipes.pkl").exists()
    import h5py
    with h5py.File(h5) as f:
        assert sorted(f.keys())[0] == f"sample_{3 * 2 ** 19:04d}"
    names = {d["name"] for d in c.get("/api/datasets").json()}
    assert {"generated/t/h5/pillbox_pipes/ (h5)", "generated/t/pkl/pillbox_pipes.pkl"} <= names
    j2 = run()                                                                # same job again: shard skipped
    assert j2["status"] == "done" and any("skipped" in line for line in j2["log"])
    assert c.post("/api/jobs/generate", json=dict(body, families=["nope"])).status_code == 422
    assert c.post("/api/jobs/generate", json=dict(body, n_total=10 ** 9)).status_code == 422


def test_field_generation_jobs_plan_and_settings(tmp_path, monkeypatch):
    """Field labels from the UI: the server plans workers × threads from its cores and memory, writes the
    TRUBA tag, passes every adaptive setting to the generator, and keeps them H5 only (no PKL)."""
    from src.service import jobs as J
    from src.service import truba as TR
    seen = []
    monkeypatch.setattr(J.JobManager, "_run", lambda self, job: seen.append(dict(job.params)))
    monkeypatch.setattr(J, "machine", lambda: {"cpus": 64, "mem_gb": 251.0})
    monkeypatch.setenv("RFCAV_GEN_ROOT", str(tmp_path / "drive"))
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    cfg = c.get("/api/jobs/config").json()
    assert cfg["cpu_count"] == 64 and cfg["mem_gb"] == 251.0
    fd = cfg["field"]["defaults"]
    assert fd["max_ndof"] == 900_000 and fd["adapt"] and fd["round_edges"] and fd["tol_f"] == 1e-6
    assert cfg["field"]["plan"]["workers"] * cfg["field"]["plan"]["threads"] <= 64
    plan = c.get("/api/jobs/plan", params={"max_ndof": 900_000}).json()
    assert plan == J.field_plan(64, 251.0, 900_000) and plan["workers"] == 16 and plan["threads"] == 4
    assert c.get("/api/jobs/plan", params={"threads": 8}).json()["workers"] == 8
    assert "memory" in c.get("/api/jobs/plan", params={"workers": 40}).json()["warnings"]

    body = {"families": ["elliptical", "spoke"], "labels": "field", "n_total": 4, "shard_size": 8,
            "tol_q": 2e-3, "model_max_elements": 8000, "round_edges": False}
    jobs = c.post("/api/jobs/generate", json=body).json()
    assert [j["params"]["family"] for j in jobs] == ["elliptical", "spoke"]
    for _ in range(100):
        if len(seen) == 2:
            break
        import time
        time.sleep(0.05)
    p = seen[0]
    assert p["labels"] == "field" and p["convert"] is False and (p["workers"], p["threads"]) == (16, 4)
    assert p["tag"] == "F_p3p2_ms0.10_k10_mf0.05_v2_sharp"
    tr = TR.build_command(dict(TR.defaults(ROOT)["form"], action="dataset", round_edges=False, tag=""))
    assert tr["tag"] == p["tag"]                                       # same tag as the TRUBA run
    a = J.generator_args(p)
    for flag, val in (("--labels", "field"), ("--threads", "4"), ("--n_workers", "16"), ("--tol_q", "0.002"),
                      ("--model_max_elements", "8000"), ("--max_ndof", "900000"), ("--sample_timeout", "7200")):
        assert a[a.index(flag) + 1] == val, flag
    assert "--no_round_edges" in a and "--no_adapt" not in a and "--no_heal" not in a
    assert "--labels" not in J.generator_args(dict(p, labels="n0"))
    # rejected: a family without a CAD solid, nested orders
    assert c.post("/api/jobs/generate", json=dict(body, families=["freeform"])).status_code == 422
    assert c.post("/api/jobs/generate", json=dict(body, model_order=3)).status_code == 422
    # small machine: the label budget follows the memory, one worker
    monkeypatch.setattr(J, "machine", lambda: {"cpus": 2, "mem_gb": 12.7})
    c2 = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    cfg2 = c2.get("/api/jobs/config").json()
    assert cfg2["field"]["defaults"]["max_ndof"] == 600_000 and cfg2["field"]["plan"]["workers"] == 1


def test_machine_and_field_plan():
    from src.service import jobs as J
    m = J.machine()
    assert m["cpus"] >= 1 and m["mem_gb"] > 0
    p = J.field_plan(64, 100.0, 900_000)                # memory-bound: fewer workers, more threads each
    assert (p["workers"], p["threads"]) == (6, 10) and not p["warnings"]
    assert J.field_plan(64, 0, 300_000)["workers"] == 16  # unknown memory: cores only
    assert "oversubscribed" in J.field_plan(8, 64.0, 300_000, threads=4, workers=4)["warnings"]
    assert J.field_tag_of(0.12, 6, adapt=False) == "F_p3p2_ms0.12_k6_mf0.05_v2_1s"


def test_generation_disabled_without_root(monkeypatch):
    monkeypatch.delenv("RFCAV_GEN_ROOT", raising=False)
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    assert c.get("/api/jobs/config").json()["enabled"] is False
    assert c.post("/api/jobs/generate", json={"families": ["box"]}).status_code == 503


def test_gen_root_inside_a_data_root_is_listed_once(data_dir, monkeypatch):
    """Colab: --data <drive> and --gen_root <drive>/ui_datasets → the generated sets appear only as generated/…"""
    monkeypatch.setenv("RFCAV_DATA", str(data_dir))
    monkeypatch.setenv("RFCAV_GEN_ROOT", str(data_dir / "h5"))
    c = TestClient(A.create_app(ModelService(None, "cpu")))
    names = {d["name"] for d in c.get("/api/datasets").json()}
    assert names == {"generated/pillbox_pipes/ (h5)"}


def test_optional_checkpoint_falls_back_to_untrained(tmp_path):
    bad = tmp_path / "last.ckpt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError, KeyError)):
        ModelService(str(bad), "cpu")
    s = ModelService(str(bad), "cpu", optional=True)
    assert s.untrained and s.load_error and s.info()["load_error"] == s.load_error
