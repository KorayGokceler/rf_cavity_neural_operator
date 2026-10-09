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
    assert j["model"]["untrained"] is True and j["model"]["field"] == "E"
    assert "elliptical" in j["families"]["train"] and "box" in j["families"]["ood"]


def test_upload_predict_fields_export(client, pillbox_msh):
    with open(pillbox_msh / "pillbox.msh", "rb") as fh:
        r = client.post("/api/geometries", files={"file": ("cavity.msh", fh)}, data={"unit": "mm"})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["check"]["ok"] and g["check"]["betti1"] == 0
    pts = f32(g["surface"]["points_b64"]).reshape(-1, 3)
    assert pts.shape[0] == g["surface"]["n_points"] and abs(pts[:, 0].max() - 100) < 1e-3    # mm
    r = client.post("/api/predictions", json={"geometry_id": g["id"]})
    assert r.status_code == 200, r.text
    p = r.json()
    assert len(p["modes"]) == 6 and any("untrained" in w for w in p["warnings"])
    assert all(np.isfinite(m["f_GHz"]) and m["eta"] >= 0 for m in p["modes"])
    m = client.get(f"/api/predictions/{p['id']}/modes/0").json()
    E = f32(m["surface"]["E_b64"]).reshape(-1, 3)
    assert E.shape == (g["surface"]["n_points"], 3) and m["surface"]["E_max"] > 0
    assert len(m["axis"]["z_mm"]) == len(m["axis"]["Ez"])
    q = client.get(f"/api/predictions/{p['id']}/plane", params={"mode": 0, "axis": "y", "res": 41}).json()
    P = f32(q["points_b64"]).reshape(-1, 3)
    assert P.shape[0] == q["n_points"] > 0 and np.ptp(P[:, 1]) < 1e-3 and abs(P[0, 1]) < 5   # y = centroid plane
    assert f32(q["H_b64"]).size == 3 * q["n_points"]
    r = client.get(f"/api/predictions/{p['id']}/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert len(r.json()["modes"]) == 6


def test_example_geometry_and_validation(client):
    r = client.post("/api/geometries/sample", json={"family": "pillbox_pipes", "id": 7, "mesh_size": 0.3})
    assert r.status_code == 200, r.text
    assert r.json()["source"]["family"] == "pillbox_pipes" and r.json()["check"]["ok"]
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
    geom, M = extract_geometry_3d(nodes, tets, "E")
    rng = np.random.default_rng(0)
    U = rng.standard_normal((len(geom["edges"]), 1))
    U[np.asarray(geom["bnd_edge"], bool)] = 0
    E, H = physical_fields(geom, M, U, np.array([1.0]), "E")
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
