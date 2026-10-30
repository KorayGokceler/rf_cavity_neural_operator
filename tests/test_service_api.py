"""Web API (src/service/api.py) end to end with an untrained model: upload / example geometry,
prediction, mode + plane data, export, limits and input validation."""
import base64
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

from src.service import api as A                                            # noqa: E402
from src.service import geometry as G                                       # noqa: E402
from src.service.model import ModelService, physical_fields                 # noqa: E402


def f32(s):
    return np.frombuffer(base64.b64decode(s), np.float32)


@pytest.fixture(scope="module")
def client():
    return TestClient(A.create_app(ModelService(None, "cpu")))


@pytest.fixture(scope="module")
def pillbox_msh(tmp_path_factory):
    gmsh = pytest.importorskip("gmsh")
    d = tmp_path_factory.mktemp("api")
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.addCylinder(0, 0, -50, 0, 0, 100, 100)        # mm, R = L = 100
    gmsh.model.occ.synchronize()
    gmsh.write(str(d / "pillbox.step"))
    gmsh.option.setNumber("Mesh.MeshSizeMax", 30)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(d / "pillbox.msh"))
    gmsh.clear()
    return d


def test_info(client):
    j = client.get("/api/info").json()
    assert j["model"]["untrained"] is True
    assert "elliptical" in j["families"]["train"] and "box" in j["families"]["ood"]


def _upload(client, pillbox_msh):
    with open(pillbox_msh / "pillbox.msh", "rb") as fh:
        r = client.post("/api/geometries", files={"file": ("cavity.msh", fh)}, data={"unit": "mm"})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload(client, pillbox_msh):
    g = _upload(client, pillbox_msh)
    assert g["check"]["ok"] and g["check"]["betti1"] == 0
    pts = f32(g["surface"]["points_b64"]).reshape(-1, 3)
    assert pts.shape[0] == g["surface"]["n_points"] and abs(pts[:, 0].max() - 100) < 1e-3    # mm


def _upload_step(client, pillbox_msh, mesh_size=0.3):
    with open(pillbox_msh / "pillbox.step", "rb") as fh:
        r = client.post("/api/geometries", files={"file": ("cavity.step", fh)},
                        data={"unit": "mm", "mesh_size": str(mesh_size)})
    assert r.status_code == 200, r.text
    return r.json()


def test_mesh_upload_cannot_be_predicted(client, pillbox_msh):
    """A mesh file has no CAD surface: shown and checked, but the field model (curved mesh) refuses it."""
    g = _upload(client, pillbox_msh)
    assert g["check"]["ok"] and g["can_predict"] is False and "CAD" in g["predict_blocker"]
    r = client.post("/api/predictions", json={"geometry_id": g["id"]})
    assert r.status_code == 422 and r.json()["detail"] == g["predict_blocker"]


def test_predict_fields_export(client, pillbox_msh):
    """STEP upload → curved netgen mesh → untrained field3d prediction: Ritz frequencies, p2 figures of
    merit, η, and the display endpoints on the linear mesh of the same curved mesh."""
    g = _upload_step(client, pillbox_msh)
    assert g["can_predict"] and g["predict_blocker"] is None and g["check"]["ok"]
    r = client.post("/api/predictions", json={"geometry_id": g["id"]})
    assert r.status_code == 200, r.text
    p = r.json()
    assert len(p["modes"]) == 6 and any("untrained" in w for w in p["warnings"])
    assert p["qoi_source"] == "p2_model" and p["n_dof"] > 0 and p["n_tets"] == g["check"]["n_tets"]
    assert all(np.isfinite(m["f_GHz"]) and m["f_GHz"] > 0 and m["eta"] >= 0 for m in p["modes"])
    assert all(m["Q0"] is None or m["Q0"] > 0 for m in p["modes"]) and any(m["Q0"] for m in p["modes"])
    assert {"mesh", "operators", "model", "qoi", "display", "total"} <= set(p["time_s"])
    assert p["model"]["model_type"] == "field3d"
    m = client.get(f"/api/predictions/{p['id']}/modes/0").json()
    E = f32(m["surface"]["E_b64"]).reshape(-1, 3)
    assert E.shape == (g["surface"]["n_points"], 3) and m["surface"]["E_max"] > 0 and np.isfinite(E).all()
    assert len(m["axis"]["z_mm"]) == len(m["axis"]["Ez"])
    q = client.get(f"/api/predictions/{p['id']}/plane", params={"mode": 0, "axis": "y", "res": 41}).json()
    P = f32(q["points_b64"]).reshape(-1, 3)
    assert P.shape[0] == q["n_points"] > 0 and np.ptp(P[:, 1]) < 1e-3 and abs(P[0, 1]) < 5   # y = centroid plane
    assert f32(q["H_b64"]).size == 3 * q["n_points"]
    v = client.get(f"/api/predictions/{p['id']}/volume", params={"mode": 0, "n": 20000}).json()
    nx, ny, nz = v["dims"]
    E = f32(v["E_b64"]).reshape(nz, ny, nx, 3)                         # x fastest (vtkImageData order)
    inside = np.frombuffer(base64.b64decode(v["inside_b64"]), np.uint8).reshape(nz, ny, nx).astype(bool)
    assert 10000 < v["n_points"] == nx * ny * nz < 40000 and 0 < v["n_inside"] == inside.sum() < v["n_points"]
    assert np.all(E[~inside] == 0) and np.abs(E[inside]).max() > 0
    assert v["origin_mm"][0] == pytest.approx(-100, abs=5) and v["spacing_mm"] > 0     # pillbox R = 100 mm
    ctr = inside[nz // 2, ny // 2, nx // 2] and not inside[0, 0, 0]       # the bbox corner is outside the cylinder
    assert ctr
    assert client.get(f"/api/predictions/{p['id']}/volume", params={"n": 10}).status_code == 422
    fb = client.get(f"/api/predictions/{p['id']}/fields").json()            # every mode, at the vertices
    K, nv = fb["n_modes"], fb["n_vertices"]
    E16 = np.frombuffer(base64.b64decode(fb["E_b64"]), np.float16).astype(float).reshape(K, nv, 3)
    E0 = E16[0] * fb["E_max"][0]
    nodes = f32(fb["nodes_b64"]).reshape(nv, 3)
    vid = np.frombuffer(base64.b64decode(fb["vid_b64"]), np.uint32)
    assert K == 6 and nodes.shape == (nv, 3) and np.abs(nodes[:, 0]).max() == pytest.approx(100, abs=5)
    assert np.frombuffer(base64.b64decode(fb["tets_b64"]), np.uint32).size == 4 * fb["n_tets"]
    Es = f32(m["surface"]["E_b64"]).reshape(-1, 3)                          # mode 0 on the surface (above)
    assert np.allclose(E0[vid], Es, rtol=2e-3, atol=2e-3 * fb["E_max"][0])      # = the per-mode surface data
    assert np.linalg.norm(E16, axis=2).max() == pytest.approx(1, abs=1e-3)
    assert len(fb["axis"]["Ez"]) == K and len(fb["axis"]["Ez"][0]) == len(fb["axis"]["z_mm"])
    r = client.get(f"/api/predictions/{p['id']}/fields", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"
    r = client.get(f"/api/predictions/{p['id']}/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert len(r.json()["modes"]) == 6 and r.json()["qoi_source"] == "p2_model"
    p2 = client.post("/api/predictions", json={"geometry_id": g["id"]}).json()      # cached curved mesh + operators
    assert p2["time_s"]["operators"] == 0.0 and [m["f_GHz"] for m in p2["modes"]] == [m["f_GHz"] for m in p["modes"]]


def test_prediction_budget_and_no_cad_family(monkeypatch, pillbox_msh):
    """Above the curved-tet budget the geometry is shown but not predicted (422 with the reason); the discrete
    freeform family has no CAD solid."""
    monkeypatch.setattr(A.Limits, "max_predict_tets", 10)
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    assert c.get("/api/info").json()["limits"]["max_predict_tets"] == 10
    g = _upload_step(c, pillbox_msh)
    assert g["check"]["ok"] and not g["can_predict"] and "budget" in g["predict_blocker"]
    r = c.post("/api/predictions", json={"geometry_id": g["id"]})
    assert r.status_code == 422 and "budget" in r.json()["detail"]
    f = c.post("/api/geometries/sample", json={"family": "freeform", "id": 1, "mesh_size": 0.3}).json()
    assert f["check"]["ok"] and not f["can_predict"] and "CAD" in f["predict_blocker"]
    assert c.post("/api/predictions", json={"geometry_id": f["id"]}).status_code == 422


def test_ritz_frequencies_are_upper_bounds_on_a_box():
    """Untrained field3d on a generator box: the Ritz values lie above the discrete HCurl(p2) eigenvalues of the
    same curved mesh (the kernel projection removes the gradients: no spurious θ ≈ 0), which approximate the
    analytic box spectrum."""
    from src.data_gen import dataset_generator_3d as gen
    from src.data_gen import highorder as ho
    geo = G.run_isolated(G.mesh_family, "box", 3, 0.1, timeout=120)
    assert geo["mesh_src"] is not None and geo["blocker"] is None
    svc = ModelService(None, "cpu")
    geom, M = svc.geometry(geo["nodes"], geo["tets"], "box", geo["params"])
    sol = svc.predict(geo["mesh_src"], geom, M, "box", geo["params"])
    f = np.array([m["f_GHz"] for m in sol["modes"]])
    lab = ho.solve_modes(G.load_mesh_src(geo["mesh_src"]), len(f), order=2)
    f_h = lab["f_hz"][:len(f)] / 1e9
    assert np.all(f >= f_h * (1 - 1e-6)), (f, f_h)
    p = geo["params"]
    f_exact = np.sqrt(gen.box_spectrum(p["a"], p["b"], p["d"])[:len(f)]) * 299792458.0 / (2 * np.pi) / 1e9
    assert np.allclose(f_h, f_exact, rtol=0.01), (f_h, f_exact)          # 154 curved tets: 0.1–0.5 %


def test_load_model_rejects_other_model_types(tmp_path):
    import pytorch_lightning as pl
    import torch
    from src.training.lightning_module import CavityLightning
    m = CavityLightning(val_dim=9, hidden_dim=16, n_heads=2, n_basis=8, num_field_modes=3, rff_dim=8,
                        eigenspace_kwargs={"n_layers": 1}, data_cfg={"field": "E", "feature_indices": None})
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.strategy.connect(m)
    good = tmp_path / "field.ckpt"
    trainer.save_checkpoint(str(good))
    s = ModelService(str(good), "cpu")
    assert not s.untrained and s.n_modes == 3 and s.info()["model_type"] == "field3d"
    ck = torch.load(str(good), map_location="cpu", weights_only=False)
    ck["hyper_parameters"]["model_type"] = "eigenspace3d"                   # the removed N0 model
    bad = tmp_path / "n0.ckpt"
    torch.save(ck, str(bad))
    with pytest.raises(ValueError, match="field3d"):
        ModelService(str(bad), "cpu")
    s = ModelService(str(bad), "cpu", optional=True)
    assert s.untrained and "eigenspace3d" in s.load_error


def test_example_geometry_and_validation(client):
    r = client.post("/api/geometries/sample", json={"family": "pillbox_pipes", "id": 7, "mesh_size": 0.3})
    assert r.status_code == 200, r.text
    assert r.json()["source"]["family"] == "pillbox_pipes" and r.json()["check"]["ok"] and r.json()["can_predict"]
    assert r.json()["params"]["R"] > 0
    assert client.post("/api/geometries/sample", json={"family": "nope"}).status_code == 422
    assert client.post("/api/geometries/sample", json={"family": "box", "mesh_size": 0.01}).status_code == 422
    r = client.post("/api/geometries", files={"file": ("x.exe", b"MZ")}, data={"unit": "mm"})
    assert r.status_code == 415
    r = client.post("/api/geometries", files={"file": ("x.step", b"not a step file")}, data={"unit": "mm"})
    assert r.status_code == 422
    assert client.post("/api/predictions", json={"geometry_id": "0" * 32}).status_code == 404


def test_upload_size_limit(monkeypatch, pillbox_msh):
    monkeypatch.setattr(A.Limits, "max_upload_mb", 0.0001)
    c = TestClient(A.create_app(ModelService(None, "cpu")))
    with open(pillbox_msh / "pillbox.msh", "rb") as fh:
        assert c.post("/api/geometries", files={"file": ("c.msh", fh)}, data={"unit": "mm"}).status_code == 413


def test_run_isolated_timeout_and_errors():
    with pytest.raises(TimeoutError):
        G.run_isolated(_sleep, 5, timeout=0.5)
    with pytest.raises(RuntimeError, match="ValueError: boom"):
        G.run_isolated(_boom, timeout=30)


def _sleep(s):
    import time
    time.sleep(s)


def _boom():
    raise ValueError("boom")


def test_physical_fields_energy_normalisation(pillbox_msh):
    """U = ½ε0∫|E|² = 1 J for the displayed E (nodal average ≈ the N0 field on a fine-enough mesh)."""
    from src.data.dataset_converter_3d import extract_geometry_3d, tet_geometry
    nodes, tets = G.mesh_file(str(pillbox_msh / "pillbox.msh"), "mm")
    geom, M = extract_geometry_3d(nodes, tets)
    rng = np.random.default_rng(0)
    U = rng.standard_normal((len(geom["edges"]), 1))
    U[np.asarray(geom["bnd_edge"], bool)] = 0
    E, H = physical_fields(geom, M, U, np.array([1.0]))
    vol, _ = tet_geometry(nodes, tets)
    e2 = (np.linalg.norm(E[:, :, 0], axis=1) ** 2)[tets].mean(1)            # per-tet mean of |E|² at vertices
    U_J = 0.5 * 8.8541878128e-12 * (vol * e2).sum()
    assert 0.2 < U_J < 1.5                                                  # nodal averaging smooths a random field
    assert np.all(np.isfinite(H))


def test_access_token(monkeypatch):
    """RFCAV_TOKEN: 401 without it; ?token= sets an HttpOnly cookie that authorises the API."""
    monkeypatch.setenv("RFCAV_TOKEN", "s3cret")
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    assert c.get("/api/info").status_code == 401
    assert c.get("/api/info", params={"token": "wrong"}).status_code == 401
    r = c.get("/api/info", params={"token": "s3cret"})
    assert r.status_code == 200 and "rfcav_token" in r.headers.get("set-cookie", "")
    assert "httponly" in r.headers["set-cookie"].lower()
    assert c.get("/api/info").status_code == 200                              # cookie kept by the client
    assert TestClient(c.app).get("/api/info", headers={"X-RFCAV-Token": "s3cret"}).status_code == 200


def test_serve_web_helpers():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import serve_web
    ip = serve_web.lan_ip()
    assert ip.count(".") == 3
