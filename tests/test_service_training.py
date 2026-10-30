"""Training from the web UI (src/service/training.py + /api/train/*): two field-label datasets, train a tiny
field3d model (configs/field_3d.yaml), follow progress.json, resume with more epochs, switch the served model."""
import os
import re
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient                                   # noqa: E402

from src.service import api as A                                            # noqa: E402
from src.service import training as T                                       # noqa: E402
from src.service.model import ModelService                                  # noqa: E402
from tests.test_service_datasets import write_field_h5                     # noqa: E402


def test_presets_match_the_cluster_config():
    """cluster/truba/config.sh model_overrides and the UI presets describe the same models."""
    sh = open(os.path.join(ROOT, 'cluster', 'truba', 'config.sh')).read()
    for name in ('small', 'base', 'large', 'xl'):
        line = re.search(rf'^\s*{name}\)\s*echo "([^"]+)"', sh, re.M).group(1)
        assert sorted(line.split()) == sorted(T.preset_overrides(name)), name


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("trainui")
    for name, ids in (("a", range(0, 6)), ("b", range(6, 12))):      # 12 geometries: 9 / 1 / 2 split
        (d / "data" / "F" / name).mkdir(parents=True)
        write_field_h5(d / "data" / "F" / name / f"{name}_s00000.h5", ids)
    return d


def _wait(c, name, until=("finished", "failed", "stopped"), timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/train/runs/{name}").json()
        if r["status"] in until:
            return r
        time.sleep(0.5)
    raise AssertionError(f"run {name} did not finish: {r}")


def test_train_resume_and_load(tree, monkeypatch):
    monkeypatch.setenv("RFCAV_DATA", str(tree / "data"))
    monkeypatch.setenv("RFCAV_RUNS_ROOT", str(tree / "runs"))
    monkeypatch.delenv("RFCAV_GEN_ROOT", raising=False)
    svc = ModelService(None, "cpu")
    c = TestClient(A.create_app(svc))
    cfg = c.get("/api/train/config").json()
    assert cfg["enabled"] and "tiny" in cfg["presets"] and not cfg["busy"] and cfg["model_type"] == "field3d"
    assert cfg["defaults"]["batch_size"] == 2 and "qoi_weight" not in cfg["defaults"]
    ds = c.get("/api/datasets").json()
    ids = [d["id"] for d in ds]
    assert len(ids) == 2 and {d["kind"] for d in ds} == {"field"}
    body = {"name": "ui1", "datasets": ids, "preset": "tiny", "epochs": 1, "batch_size": 2, "n_modes": 4,
            "num_workers": 0}
    r = c.post("/api/train/start", json=body)
    assert r.status_code == 200, r.text
    assert c.post("/api/train/start", json=dict(body, name="ui2")).status_code == 409      # one at a time
    r = _wait(c, "ui1")
    assert r["status"] == "finished", r["log"][-15:]
    assert r["epoch"] == 1 and len(r["history"]) == 1 and "val/field_rel_l2" in r["history"][0]
    assert r["test"] and r["best"]["path"].endswith(".ckpt") and r["resumable"]
    assert r["params"]["sources"] and len(r["params"]["data"]["h5"]) == 2
    assert c.post("/api/train/start", json=body).status_code == 409                       # name taken

    r = c.post("/api/train/runs/ui1/resume", json={"epochs": 2})
    assert r.status_code == 200, r.text
    r = _wait(c, "ui1")
    assert r["status"] == "finished" and [h["epoch"] for h in r["history"]] == [0, 1]

    assert svc.untrained
    info = c.post("/api/model/load", json={"run": "ui1"}).json()
    assert info["untrained"] is False and info["run"] == "ui1" and info["n_modes"] == 4
    assert c.post("/api/model/load", json={"run": ".."}).status_code == 422
    assert [x["name"] for x in c.get("/api/train/runs").json()] == ["ui1"]

    # stop a long run after its first epoch: stopped, resumable from last.ckpt
    assert c.post("/api/train/start", json=dict(body, name="ui3", epochs=500)).status_code == 200
    t0 = time.time()
    while c.get("/api/train/runs/ui3").json()["epoch"] < 1 and time.time() - t0 < 300:
        time.sleep(0.3)
    assert c.post("/api/train/stop").json()["stopped"] == "ui3"
    r = _wait(c, "ui3", until=("stopped",))
    assert r["resumable"] and 1 <= r["epoch"] < 500
    assert not c.get("/api/train/config").json()["busy"]


def test_training_off_without_runs_root(monkeypatch):
    monkeypatch.delenv("RFCAV_RUNS_ROOT", raising=False)
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    assert c.get("/api/train/config").json()["enabled"] is False
    assert c.post("/api/train/start", json={"name": "x", "datasets": ["d"]}).status_code == 503
